"""Synthesis agent: turn an arbitrary Python game repo into an OpenReward env.

Driver around `claude_agent_sdk.query` that:
  1. Clones the target game repo (depth-1) into a workdir.
  2. Embeds `adapter_prompt/SKILL.md` verbatim as the system prompt.
  3. Hands Claude the built-in Read/Write/Edit/Glob/Grep/Bash/WebFetch/TodoWrite
     tools and asks it to follow SKILL.md Phases 1-5 to write a new
     `src/<env_name>/` package + matching tests.

Auth path: claude-agent-sdk shells out to the local `claude` CLI, which uses
the user's existing Claude Code subscription. No ANTHROPIC_API_KEY required.

Example:
    python scripts/synthesis_agent.py \
        --repo-url https://github.com/niklasf/python-chess \
        --env-name chess_env \
        --game-name chess \
        --extra-url https://python-chess.readthedocs.io/en/latest/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

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
    query,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_PATH = REPO_ROOT / "adapter_prompt" / "SKILL.md"
TRAJ_DIR = REPO_ROOT / "trajectories"
OPENREWARD_DOCS_URL = "https://docs.openreward.ai/environments/building-agentic-environments"


# --- message serializers (copied verbatim from tests/test_claude_agent.py) ---

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


# --- live console echo (so the user can watch the synthesis in real time) ---

def _echo(msg) -> None:
    if isinstance(msg, AssistantMessage):
        for b in msg.content:
            if isinstance(b, TextBlock):
                print(b.text, flush=True)
            elif isinstance(b, ThinkingBlock):
                print(f"[thinking] {b.thinking[:200]}{'...' if len(b.thinking) > 200 else ''}", flush=True)
            elif isinstance(b, ToolUseBlock):
                preview = json.dumps(b.input, default=str)
                if len(preview) > 200:
                    preview = preview[:200] + "..."
                print(f"[tool_use] {b.name}({preview})", flush=True)
    elif isinstance(msg, UserMessage):
        if isinstance(msg.content, list):
            for b in msg.content:
                if isinstance(b, ToolResultBlock):
                    txt = ""
                    if isinstance(b.content, list):
                        for c in b.content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                txt = c.get("text", "")
                                break
                    if txt:
                        head = txt.splitlines()[0] if txt.splitlines() else txt
                        print(f"[tool_result{' ERROR' if b.is_error else ''}] {head[:200]}", flush=True)
    elif isinstance(msg, ResultMessage):
        cost = f"${msg.total_cost_usd:.4f}" if msg.total_cost_usd is not None else "n/a"
        print(f"\n[result] subtype={msg.subtype} turns={msg.num_turns} cost={cost}", flush=True)


# --- repo clone helper ---

def clone_repo(repo_url: str, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    name = repo_url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    target = workdir / name
    if target.exists():
        print(f"[clone] reusing existing checkout at {target}", flush=True)
        return target
    print(f"[clone] git clone --depth 1 {repo_url} -> {target}", flush=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(target)],
        check=True,
    )
    return target


# --- prompt construction ---

SYSTEM_SUFFIX = """

---

You are a code-generation agent operating inside the
`rl-env-hackathon-complex-worlds` repo. You have these tools:
  Read, Write, Edit, Glob, Grep, Bash, WebFetch, TodoWrite.

Use the recipe above (SKILL.md). Modifications for this run:
  * SKIP Phase 0 — clarifying questions are already answered with these
    defaults: full game per episode, tool-use (function-calling) interface,
    dense per-step reward, no visualization (no Phase 7).
  * DO Phases 1-5 — learn both APIs (WebFetch the OpenReward docs URL
    given to you; probe the cloned game repo with Bash/Read/Grep), design
    the env, lay out the package, and write the scripted + Claude SDK tests.
  * SKIP Phase 6-7 — do NOT run pytest, do NOT run the Claude SDK, do NOT
    set up any web UI. Generation only — the user will verify manually.

Mirror the existing reference implementation `src/catan_env/` structurally
(same file names, same Environment subclass shape, same OpenRewardPlayer
pattern with `__reduce__`, same Server entry point). Read those files
before writing yours.

When you finish, print a one-paragraph summary of:
  - the env class you created and its task_spec keys,
  - the three @tool methods,
  - how dense reward is computed,
  - the test file paths,
  - any new pyproject.toml dependency you added.
Then stop. Do not run anything.
"""


def build_system_prompt() -> str:
    skill = SKILL_PATH.read_text()
    return skill + SYSTEM_SUFFIX


def build_user_prompt(
    *,
    game_name: str,
    env_name: str,
    cloned_path: Path,
    extra_urls: list[str],
) -> str:
    extras = "\n".join(f"  - {u}" for u in extra_urls) if extra_urls else "  (none)"
    return f"""Build an OpenReward environment package for the game: **{game_name}**.

Inputs you have:
  * Cloned game source (read-only reference): {cloned_path}
  * Reference env to mirror structurally:    {REPO_ROOT}/src/catan_env/
  * Reference scripted test:                 {REPO_ROOT}/tests/test_scripted_agent.py
  * Reference Claude SDK test:               {REPO_ROOT}/tests/test_claude_agent.py
  * OpenReward docs (fetch this first):      {OPENREWARD_DOCS_URL}
  * Extra context URLs to fetch:
{extras}

Output files to write (create the directory first):
  * {REPO_ROOT}/src/{env_name}/__init__.py
  * {REPO_ROOT}/src/{env_name}/agent_player.py
  * {REPO_ROOT}/src/{env_name}/env.py
  * {REPO_ROOT}/src/{env_name}/server.py
  * {REPO_ROOT}/tests/test_{env_name}_scripted.py
  * {REPO_ROOT}/tests/test_{env_name}_claude.py

Also: if the new env requires a Python dependency that is NOT already in
{REPO_ROOT}/pyproject.toml, add it to the `dependencies` list. Do not touch
any other existing files in this repo.

Recommended workflow:
  1. WebFetch the OpenReward docs URL and any extra URLs above.
  2. Read src/catan_env/env.py, src/catan_env/agent_player.py,
     src/catan_env/server.py, tests/test_scripted_agent.py,
     tests/test_claude_agent.py to internalize the reference shape.
  3. Use Bash + Glob + Grep + Read to probe the cloned repo at
     {cloned_path}: find Player/Agent base classes, the decide() signature,
     how to enumerate legal actions, how to advance one ply, how to detect
     terminal state and read the winner/score.
  4. Design the env per SKILL.md Phase 2 (three @tool methods, dense reward).
  5. Write the six output files. Generation only — DO NOT run pytest, DO NOT
     run the env, DO NOT execute the Claude SDK test.
  6. Print the summary paragraph and stop.
"""


# --- main async loop ---

async def run(args: argparse.Namespace) -> int:
    if not SKILL_PATH.exists():
        print(f"ERROR: SKILL.md not found at {SKILL_PATH}", file=sys.stderr)
        return 2

    workdir = Path(args.workdir) if args.workdir else Path(f"/tmp/synth_{args.env_name}")
    cloned = clone_repo(args.repo_url, workdir)

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        game_name=args.game_name or args.env_name,
        env_name=args.env_name,
        cloned_path=cloned,
        extra_urls=list(args.extra_url or []),
    )

    allowed = ["Read", "Write", "Edit", "Glob", "Grep", "Bash", "WebFetch", "TodoWrite"]
    stderr_lines: list[str] = []
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=allowed,
        permission_mode="bypassPermissions",
        max_turns=args.max_turns,
        model=args.model,
        cwd=str(REPO_ROOT),
        stderr=lambda line: stderr_lines.append(line),
    )

    trajectory: list[dict] = [{
        "kind": "config",
        "repo_url": args.repo_url,
        "env_name": args.env_name,
        "game_name": args.game_name or args.env_name,
        "extra_urls": list(args.extra_url or []),
        "cloned_path": str(cloned),
        "model": args.model,
        "max_turns": args.max_turns,
        "allowed_tools": allowed,
        "system_prompt": system_prompt,
        "initial_user_prompt": user_prompt,
    }]

    cli_error: Optional[str] = None
    last_result_subtype: Optional[str] = None
    total_cost_usd: Optional[float] = None
    num_turns: Optional[int] = None

    print(f"\n=== synthesis agent starting ({args.env_name}) ===\n", flush=True)
    try:
        async for msg in query(prompt=user_prompt, options=options):
            _echo(msg)
            trajectory.append({"kind": "message", **_message_to_dict(msg)})
            if isinstance(msg, ResultMessage):
                last_result_subtype = msg.subtype
                total_cost_usd = msg.total_cost_usd
                num_turns = msg.num_turns
    except Exception as e:
        cli_error = f"{type(e).__name__}: {e}"
        trajectory.append({"kind": "error", "error": cli_error, "stderr_tail": stderr_lines[-50:]})
        print(f"\n[ERROR] {cli_error}", file=sys.stderr, flush=True)

    summary = {
        "kind": "summary",
        "env_name": args.env_name,
        "last_result_subtype": last_result_subtype,
        "total_cost_usd": total_cost_usd,
        "num_turns": num_turns,
        "cli_error": cli_error,
    }
    trajectory.append(summary)

    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRAJ_DIR / f"synthesis_{args.env_name}_{int(time.time())}.json"
    out_path.write_text(json.dumps(trajectory, indent=2, default=str))
    print(f"\ntrajectory saved to {out_path}", flush=True)
    print(f"summary: {summary}", flush=True)

    benign = {"success", "error_max_turns"}
    if cli_error and last_result_subtype not in benign:
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-url", required=True, help="git URL of the target game repo to wrap")
    p.add_argument("--env-name", required=True, help="python package name for the new env, e.g. chess_env")
    p.add_argument("--game-name", default=None, help="human-readable game label (defaults to --env-name)")
    p.add_argument("--extra-url", action="append", default=[], help="additional URL for the agent to WebFetch (repeatable)")
    p.add_argument("--workdir", default=None, help="where to clone the game repo (default: /tmp/synth_<env_name>)")
    p.add_argument("--max-turns", type=int, default=80, help="max Claude turns (default 80)")
    p.add_argument("--model", default="claude-sonnet-4-5", help="Claude model id (default claude-sonnet-4-5)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if shutil.which("git") is None:
        print("ERROR: git is required on PATH for repo cloning.", file=sys.stderr)
        sys.exit(2)
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
