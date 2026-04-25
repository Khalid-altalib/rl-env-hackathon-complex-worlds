"""End-to-end test: Claude Agent SDK plays Catan via in-process MCP tools.

Uses claude-agent-sdk, which authenticates through the user's Claude Code
subscription (no ANTHROPIC_API_KEY needed). The env's tools are exposed as an
in-process MCP server. The full message stream is saved as a trajectory JSON.

Also exposes `run_catan_episode(...)` as a reusable helper that scripts
(e.g. scripts/experience_smoketest.py) import to run a single episode with a
configurable system-prompt extension.
"""

import json
import time
from pathlib import Path
from typing import Annotated, Optional

import pytest

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    query,
    tool,
)

from catan_env.env import CatanEnv, PlayActionParams


TRAJ_DIR = Path(__file__).parent.parent / "trajectories"


def _block_to_dict(b):
    if isinstance(b, TextBlock):
        return {"type": "text", "text": b.text}
    if isinstance(b, ThinkingBlock):
        return {"type": "thinking", "thinking": b.thinking}
    if isinstance(b, ToolUseBlock):
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    if isinstance(b, ToolResultBlock):
        content = b.content
        if isinstance(content, list):
            content = [c if isinstance(c, dict) else getattr(c, "__dict__", str(c)) for c in content]
        return {"type": "tool_result", "tool_use_id": b.tool_use_id, "is_error": b.is_error, "content": content}
    return {"type": getattr(b, "type", b.__class__.__name__), "repr": repr(b)}


def _message_to_dict(m):
    if isinstance(m, AssistantMessage):
        return {"role": "assistant", "model": m.model, "content": [_block_to_dict(b) for b in m.content]}
    if isinstance(m, UserMessage):
        content = m.content
        if isinstance(content, list):
            content = [_block_to_dict(b) for b in content]
        return {"role": "user", "content": content}
    if isinstance(m, SystemMessage):
        return {"role": "system", "subtype": m.subtype, "data": m.data}
    if isinstance(m, ResultMessage):
        return {
            "role": "result",
            "subtype": m.subtype,
            "duration_ms": m.duration_ms,
            "num_turns": m.num_turns,
            "is_error": m.is_error,
            "total_cost_usd": m.total_cost_usd,
            "usage": m.usage,
            "result": m.result,
        }
    return {"role": "unknown", "repr": repr(m)}


def _make_tools(env: CatanEnv):
    @tool("get_state", "Return a textual summary of the current Catan board state.", {})
    async def get_state(args):
        out = await env.get_state()
        return {"content": [{"type": "text", "text": out.blocks[0].text}]}

    @tool("list_legal_actions", "Return numbered list of legal actions for the agent's current decision.", {})
    async def list_legal_actions(args):
        out = await env.list_legal_actions()
        return {"content": [{"type": "text", "text": out.blocks[0].text}]}

    @tool(
        "play_action",
        "Apply one legal action by its index (from list_legal_actions) and advance the game.",
        {"index": Annotated[int, "Index into the most recent list_legal_actions result."]},
    )
    async def play_action(args):
        out = await env.play_action(PlayActionParams(index=int(args["index"])))
        return {
            "content": [{"type": "text", "text": out.blocks[0].text}],
            "is_error": False,
        }

    return [get_state, list_legal_actions, play_action]


BASE_SYSTEM_PROMPT = (
    "You are an agent playing Settlers of Catan. Win by reaching the victory-point "
    "target stated in the initial user message. Use the catan tools to query legal "
    "actions and play moves; keep responses brief and favor tool calls. Stop only "
    "when the game is over (a play_action result will say so) or after many turns of play."
)


async def run_catan_episode(
    task_spec: Optional[dict] = None,
    max_turns: int = 80,
    model: str = "claude-sonnet-4-5",
    extra_system_prompt: str = "",
    label: str = "",
    visualize: bool = False,
) -> dict:
    """Run one Claude-vs-Catan episode and save its trajectory.

    Returns a dict with: ticks, winner, agent_vp_final, cum_reward,
    trajectory_path, system_prompt, finished, cli_error.
    """
    spec = dict(task_spec or {})
    spec.setdefault("seed", 7)
    spec.setdefault("agent_color", "RED")
    spec.setdefault("max_ticks", 4000)
    spec.setdefault("visualize", visualize)

    env = CatanEnv(task_spec=spec)
    env.setup()
    if env.last_view_url:
        print(f"\n>>> Watch live: {env.last_view_url} <<<\n")

    tools = _make_tools(env)
    server = create_sdk_mcp_server(name="catan", version="0.1.0", tools=tools)

    allowed = [
        "mcp__catan__get_state",
        "mcp__catan__list_legal_actions",
        "mcp__catan__play_action",
    ]

    system_prompt = BASE_SYSTEM_PROMPT
    if extra_system_prompt.strip():
        system_prompt = (
            BASE_SYSTEM_PROMPT
            + "\n\n--- Lessons from past games (experience library) ---\n"
            + extra_system_prompt.strip()
            + "\n--- end lessons ---"
        )

    initial_text = (
        env.get_prompt()[0].text
        + "\n\nNow start playing. Call mcp__catan__list_legal_actions, choose an index, "
        "then call mcp__catan__play_action with that index. Repeat until the game ends."
    )

    stderr_lines: list[str] = []
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        mcp_servers={"catan": server},
        allowed_tools=allowed,
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        model=model,
        stderr=lambda line: stderr_lines.append(line),
    )

    trajectory: list[dict] = [{
        "kind": "config",
        "system_prompt": system_prompt,
        "initial_user_prompt": initial_text,
        "allowed_tools": allowed,
        "task_spec": env.task_spec,
        "max_turns": max_turns,
        "model": model,
        "label": label,
    }]

    finished_marker_seen = False
    cli_error: Optional[str] = None
    last_result_subtype: Optional[str] = None
    try:
        async for msg in query(prompt=initial_text, options=options):
            trajectory.append({"kind": "message", **_message_to_dict(msg)})
            if isinstance(msg, ResultMessage):
                last_result_subtype = msg.subtype
            if env.game.winning_color() is not None or env._ticks >= env.max_ticks:
                finished_marker_seen = True
    except Exception as e:
        cli_error = f"{type(e).__name__}: {e}"
        trajectory.append({"kind": "error", "error": cli_error, "stderr_tail": stderr_lines[-50:]})

    winner = env.game.winning_color()
    summary = {
        "kind": "summary",
        "ticks": env._ticks,
        "winner": winner.value if winner else None,
        "finished": finished_marker_seen,
        "agent_vp_final": env._agent_vp_prev,
        "cum_reward": env._cum_reward,
        "last_result_subtype": last_result_subtype,
    }
    trajectory.append(summary)

    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    name = f"claude_catan_{label}_{int(time.time())}.json" if label else f"claude_catan_{int(time.time())}.json"
    out_path = TRAJ_DIR / name
    out_path.write_text(json.dumps(trajectory, indent=2, default=str))
    print(f"\ntrajectory saved to {out_path}")
    print(f"summary: {summary}")
    if stderr_lines:
        print("--- last stderr lines ---")
        for line in stderr_lines[-30:]:
            print(line.rstrip())
    if cli_error:
        print(f"CLI error: {cli_error}")

    return {
        "ticks": env._ticks,
        "winner": winner.value if winner else None,
        "agent_vp_final": env._agent_vp_prev,
        "cum_reward": env._cum_reward,
        "finished": finished_marker_seen,
        "trajectory_path": str(out_path),
        "system_prompt": system_prompt,
        "cli_error": cli_error,
        "last_result_subtype": last_result_subtype,
        "stderr_tail": stderr_lines[-30:],
    }


@pytest.mark.asyncio
async def test_claude_plays_catan():
    result = await run_catan_episode(
        task_spec={"seed": 7, "agent_color": "RED", "max_ticks": 4000},
        max_turns=30,
        visualize=True,
    )
    benign = {"success", "error_max_turns"}
    if result["cli_error"] is not None and result["last_result_subtype"] not in benign:
        raise AssertionError(f"CLI failed: {result['cli_error']}")
    assert result["ticks"] > 0, "no ticks executed — agent never called play_action"
