"""Analyze a Catan trajectory and update the experience library.

Reads a trajectory JSON written by tests/test_claude_agent.py:run_catan_episode,
plus the current experiences/catan_experience.md, and asks Claude (no tools) to
extract 1-3 portable, prescriptive lessons and merge them into the library.
The experience file is rewritten in place.

Usage:
    python scripts/analyze_trajectory.py <trajectory.json> [--experience PATH]
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIENCE_PATH = REPO_ROOT / "experiences" / "catan_experience.md"

START_MARK = "===EXPERIENCE_LIBRARY_START==="
END_MARK = "===EXPERIENCE_LIBRARY_END==="


def _compact_trajectory(traj: list[dict]) -> str:
    """Produce a compact text view of a trajectory: config, action log, summary.

    Drops thinking/cache fields and keeps the parts most relevant for extracting
    lessons (initial prompt, every applied action with its step reward + post-state,
    final summary).
    """
    config = next((e for e in traj if e.get("kind") == "config"), {})
    summary = next((e for e in traj if e.get("kind") == "summary"), {})

    lines: list[str] = []
    lines.append("## Config")
    lines.append(f"- task_spec: {json.dumps(config.get('task_spec', {}))}")
    lines.append(f"- max_turns: {config.get('max_turns')}")
    lines.append(f"- model: {config.get('model')}")
    lines.append(f"- label: {config.get('label')}")
    lines.append("- system_prompt:")
    sp = config.get("system_prompt", "")
    for ln in sp.splitlines():
        lines.append(f"    {ln}")

    lines.append("\n## Action log (agent's reasoning + applied actions)")
    step = 0
    for entry in traj:
        if entry.get("kind") != "message":
            continue
        role = entry.get("role")
        if role == "assistant":
            for c in entry.get("content", []) or []:
                if isinstance(c, dict) and c.get("type") == "text":
                    txt = (c.get("text") or "").strip()
                    if txt:
                        lines.append(f"[reasoning] {txt[:400]}")
                elif isinstance(c, dict) and c.get("type") == "tool_use":
                    name = c.get("name", "")
                    inp = c.get("input", {})
                    if "play_action" in name:
                        step += 1
                        lines.append(f"[tool_use #{step}] play_action(index={inp.get('index')})")
                    elif "list_legal_actions" in name:
                        lines.append("[tool_use] list_legal_actions")
                    elif "get_state" in name:
                        lines.append("[tool_use] get_state")
        elif role == "user":
            content = entry.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "tool_result":
                        sub = c.get("content")
                        if isinstance(sub, list):
                            for tc in sub:
                                if isinstance(tc, dict) and tc.get("type") == "text":
                                    text = tc.get("text", "")
                                    # Keep play_action results (which start with "Applied:")
                                    # and trim get_state/list_legal_actions noise.
                                    if text.startswith("Applied:"):
                                        head = "\n".join(text.split("\n")[:6])
                                        lines.append(f"[result] {head}")

    lines.append("\n## Summary")
    for k in ("ticks", "winner", "finished", "agent_vp_final", "cum_reward", "last_result_subtype"):
        if k in summary:
            lines.append(f"- {k}: {summary[k]}")

    return "\n".join(lines)


ANALYZER_SYSTEM_PROMPT = f"""You are a coach for an LLM that plays Settlers of Catan.

You will be given:
  1. The CURRENT experience library (a markdown bullet list of lessons).
  2. A COMPACT TRAJECTORY of one episode the LLM just played.

Your job: extract 1-3 NEW lessons from this trajectory that would help a future
LLM playing Catan, then merge them into the existing library and emit the
fully updated library.

Rules for good lessons:
  - Portable and prescriptive: "Prefer X when Y" / "Avoid Z if W" / "When stuck for N+ turns, try ..."
  - About Catan strategy or this particular tool interface, NOT narration of one run.
  - No references like "in run X" or "the agent did Y once" — strip episode-specific details.
  - Concrete and actionable. One sentence per bullet.
  - Dedupe with existing bullets (semantically). If a similar lesson exists, refine the wording instead of adding a near-duplicate.
  - Keep the total list to AT MOST 20 bullets. Drop the weakest if you exceed.

Output format — emit EXACTLY this, with no other text outside the markers:

{START_MARK}
# Catan agent experience library

- <bullet 1>
- <bullet 2>
- ...
{END_MARK}
"""


async def analyze_trajectory(
    trajectory_path: Path,
    experience_path: Path = DEFAULT_EXPERIENCE_PATH,
    model: str = "claude-sonnet-4-5",
) -> str:
    traj = json.loads(Path(trajectory_path).read_text())
    compact = _compact_trajectory(traj)
    current = (
        experience_path.read_text()
        if experience_path.exists()
        else "# Catan agent experience library\n\n(empty)"
    )

    user_prompt = (
        "## Current experience library\n"
        + current
        + "\n\n## New episode trajectory (compact)\n"
        + compact
        + "\n\nReturn the fully updated library between the markers, nothing else."
    )

    options = ClaudeAgentOptions(
        system_prompt=ANALYZER_SYSTEM_PROMPT,
        permission_mode="bypassPermissions",
        max_turns=1,
        model=model,
    )

    chunks: list[str] = []
    async for msg in query(prompt=user_prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for b in msg.content:
                if isinstance(b, TextBlock):
                    chunks.append(b.text)
        elif isinstance(msg, ResultMessage):
            if msg.is_error:
                print(f"[analyzer] result error subtype={msg.subtype}", file=sys.stderr)

    raw = "\n".join(chunks)
    m = re.search(re.escape(START_MARK) + r"\s*(.*?)\s*" + re.escape(END_MARK), raw, re.DOTALL)
    if not m:
        print("[analyzer] WARNING: markers not found in Claude output. Raw response:", file=sys.stderr)
        print(raw, file=sys.stderr)
        print("[analyzer] leaving experience library unchanged.", file=sys.stderr)
        return current

    new_library = m.group(1).strip() + "\n"
    experience_path.parent.mkdir(parents=True, exist_ok=True)
    experience_path.write_text(new_library)
    print(f"[analyzer] updated {experience_path} ({len(new_library)} chars)")
    return new_library


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectory", type=Path, help="Path to a trajectory JSON file.")
    ap.add_argument(
        "--experience",
        type=Path,
        default=DEFAULT_EXPERIENCE_PATH,
        help=f"Path to experience markdown file (default: {DEFAULT_EXPERIENCE_PATH}).",
    )
    ap.add_argument("--model", default="claude-sonnet-4-5")
    args = ap.parse_args()

    asyncio.run(analyze_trajectory(args.trajectory, args.experience, args.model))


if __name__ == "__main__":
    main()
