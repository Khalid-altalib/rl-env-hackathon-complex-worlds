"""End-to-end test: Claude Agent SDK plays Diplomacy via in-process MCP tools.

Uses claude-agent-sdk authenticated through the user's Claude Code subscription.
Saves the full message trajectory to trajectories/claude_diplomacy_{timestamp}.json.
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

from diplomacy_env.env import DiplomacyEnv, PlayActionParams


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


def _make_tools(env: DiplomacyEnv):
    @tool("get_state", "Return the current Diplomacy board state.", {})
    async def get_state(args):
        out = await env.get_state()
        return {"content": [{"type": "text", "text": out.blocks[0].text}]}

    @tool("list_legal_actions", "Return numbered legal orders for the current orderable location.", {})
    async def list_legal_actions(args):
        out = await env.list_legal_actions()
        return {"content": [{"type": "text", "text": out.blocks[0].text}]}

    @tool(
        "play_action",
        "Choose an order by index for the current orderable location and advance the env.",
        {"index": Annotated[int, "Index from list_legal_actions output."]},
    )
    async def play_action(args):
        out = await env.play_action(PlayActionParams(index=int(args["index"])))
        return {
            "content": [{"type": "text", "text": out.blocks[0].text}],
            "is_error": False,
        }

    return [get_state, list_legal_actions, play_action]


@pytest.mark.asyncio
async def test_claude_plays_diplomacy():
    env = DiplomacyEnv(task_spec={"seed": 17, "agent_power": "FRANCE", "max_year": 1903})
    env.setup()

    tools = _make_tools(env)
    server = create_sdk_mcp_server(name="diplomacy", version="0.1.0", tools=tools)

    allowed = [
        "mcp__diplomacy__get_state",
        "mcp__diplomacy__list_legal_actions",
        "mcp__diplomacy__play_action",
    ]

    system_prompt = (
        "You are an agent playing Diplomacy as FRANCE on the standard map. "
        "The other 6 powers play random legal orders. Your goal is to capture supply centers "
        "(win at 18; you currently have 3). The episode truncates after year 1903.\n\n"
        "Each phase, every one of your units needs an order. The env asks you for ONE order at a time:\n"
        "  - mcp__diplomacy__get_state: see board, current orderable location, orders chosen so far this phase\n"
        "  - mcp__diplomacy__list_legal_actions: numbered legal orders for the current location\n"
        "  - mcp__diplomacy__play_action(index): submit that order, advance env\n"
        "Reward arrives only when a phase fully processes (after your last unit's order).\n\n"
        "Play efficiently: prefer tool calls over long reasoning. Keep going until the episode ends."
    )

    initial_text = (
        env.get_prompt()[0].text
        + "\n\nNow start playing. Call mcp__diplomacy__list_legal_actions, choose an index, "
        "call mcp__diplomacy__play_action with that index, and repeat until finished."
    )

    stderr_lines: list[str] = []
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        mcp_servers={"diplomacy": server},
        allowed_tools=allowed,
        permission_mode="bypassPermissions",
        max_turns=80,
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

    cli_error: str | None = None
    last_result_subtype: str | None = None
    try:
        async for msg in query(prompt=initial_text, options=options):
            trajectory.append({"kind": "message", **_message_to_dict(msg)})
            if isinstance(msg, ResultMessage):
                last_result_subtype = msg.subtype
    except Exception as e:
        cli_error = f"{type(e).__name__}: {e}"
        trajectory.append({"kind": "error", "error": cli_error, "stderr_tail": stderr_lines[-50:]})

    summary = {
        "kind": "summary",
        "play_action_calls": env._play_action_calls,
        "phases_processed": env._phases_processed,
        "agent_power": env.agent_power,
        "agent_sc_final": len(env.game.get_centers(env.agent_power)) if env.game else 0,
        "all_sc_final": {p: len(env.game.get_centers(p)) for p in [
            "AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"
        ]} if env.game else {},
        "phase_final": env.game.get_current_phase() if env.game else None,
        "is_game_done": env.game.is_game_done if env.game else False,
        "winner": env._winner(),
        "terminal": env._terminal(),
        "max_year": env.max_year,
    }
    trajectory.append(summary)

    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRAJ_DIR / f"claude_diplomacy_{int(time.time())}.json"
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
    assert env._play_action_calls > 0, "no play_action calls — agent never moved"
