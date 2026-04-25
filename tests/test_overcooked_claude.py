"""End-to-end test: Claude Agent SDK plays Overcooked via in-process MCP tools.

Uses claude-agent-sdk authenticated through the user's Claude Code subscription.
Saves the full message trajectory to trajectories/claude_overcooked_{timestamp}.json.
"""

import json
import time
from pathlib import Path
from typing import Annotated

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

from overcooked_env.env import OvercookedEnv, PlayActionParams


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


def _make_tools(env: OvercookedEnv):
    @tool("get_state", "Return the current ASCII grid of the Overcooked kitchen.", {})
    async def get_state(args):
        out = await env.get_state()
        return {"content": [{"type": "text", "text": out.blocks[0].text}]}

    @tool("list_legal_actions", "Return the 6 always-available indexed actions.", {})
    async def list_legal_actions(args):
        out = await env.list_legal_actions()
        return {"content": [{"type": "text", "text": out.blocks[0].text}]}

    @tool(
        "play_action",
        "Apply one action by index (0=NORTH, 1=SOUTH, 2=EAST, 3=WEST, 4=STAY, 5=INTERACT) and advance one timestep.",
        {"index": Annotated[int, "Action index 0-5."]},
    )
    async def play_action(args):
        out = await env.play_action(PlayActionParams(index=int(args["index"])))
        return {
            "content": [{"type": "text", "text": out.blocks[0].text}],
            "is_error": False,
        }

    return [get_state, list_legal_actions, play_action]


@pytest.mark.asyncio
async def test_claude_plays_overcooked():
    env = OvercookedEnv(task_spec={"layout": "cramped_room", "horizon": 400})
    env.setup()

    tools = _make_tools(env)
    server = create_sdk_mcp_server(name="overcooked", version="0.1.0", tools=tools)

    allowed = [
        "mcp__overcooked__get_state",
        "mcp__overcooked__list_legal_actions",
        "mcp__overcooked__play_action",
    ]

    system_prompt = (
        "You are an agent playing Overcooked as player 0, cooperating with an AI partner (player 1). "
        "Your goal is to deliver as many soups as possible within 400 timesteps. "
        "Use the overcooked tools to see the kitchen state, pick an action, and advance the game. "
        "Keep reasoning brief and favor tool calls. Stop only when the episode is over."
    )

    initial_text = (
        env.get_prompt()[0].text
        + "\n\nNow start playing. Call mcp__overcooked__list_legal_actions to see options, "
        "choose an action index, then call mcp__overcooked__play_action with that index. "
        "Repeat each timestep until the episode ends (play_action will say so)."
    )

    stderr_lines: list[str] = []
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        mcp_servers={"overcooked": server},
        allowed_tools=allowed,
        permission_mode="bypassPermissions",
        max_turns=30,
        model="claude-sonnet-4-5",
        stderr=lambda line: stderr_lines.append(line),
    )

    trajectory: list[dict] = [{
        "kind": "config",
        "system_prompt": system_prompt,
        "initial_user_prompt": initial_text,
        "allowed_tools": allowed,
        "task_spec": env.task_spec,
    }]

    finished_marker_seen = False
    cli_error: str | None = None
    last_result_subtype: str | None = None
    try:
        async for msg in query(prompt=initial_text, options=options):
            trajectory.append({"kind": "message", **_message_to_dict(msg)})
            if isinstance(msg, ResultMessage):
                last_result_subtype = msg.subtype
            if env._done:
                finished_marker_seen = True
    except Exception as e:
        cli_error = f"{type(e).__name__}: {e}"
        trajectory.append({"kind": "error", "error": cli_error, "stderr_tail": stderr_lines[-50:]})

    summary = {
        "kind": "summary",
        "steps": env._step_count,
        "soups_delivered": env._soups_delivered,
        "finished": finished_marker_seen,
        "layout": env.layout,
        "horizon": env.horizon,
    }
    trajectory.append(summary)

    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRAJ_DIR / f"claude_overcooked_{int(time.time())}.json"
    out_path.write_text(json.dumps(trajectory, indent=2, default=str))
    print(f"\ntrajectory saved to {out_path}")
    print(f"summary: {summary}")
    if stderr_lines:
        print("--- last stderr lines ---")
        for line in stderr_lines[-30:]:
            print(line.rstrip())
    if cli_error:
        print(f"CLI error: {cli_error}")

    benign = {"success", "error_max_turns"}
    if cli_error is not None and last_result_subtype not in benign:
        raise AssertionError(f"CLI failed: {cli_error}")
    assert env._step_count > 0, "no steps executed — agent never called play_action"
