---
name: cli-game-to-openreward
description: Wrap a Python CLI/library game (e.g. catanatron, chess engine, gymnasium env) as an OpenReward agentic environment with @tool methods, dense reward, scripted smoke test, and a Claude Agent SDK end-to-end test that uses the user's Claude subscription (no ANTHROPIC_API_KEY needed). Verifies itself by running both tests and saving a JSON trajectory of the Claude rollout. Use when the user asks to build a new openreward env from an existing Python game/library or links to its docs.
---

# CLI Game → OpenReward Environment

End-to-end recipe to convert a turn-based Python game into an OpenReward `Environment` with two self-verifying tests.

This is condensed from the catanatron build at `/Users/julie3399/rl-env-hackathon-complex-worlds/`. Read that repo when you want a working reference.

## Phase 0 — Clarify before coding

Always confirm with the user (use AskUserQuestion):

1. **Game scope** — full game vs single decision per episode vs custom subset.
2. **Agent interface** — tool-use (function calling) is the default; alternative is parsed-text replies.
3. **Reward shape** — sparse win/loss vs dense per-step (default to dense; users almost always want it).
4. **Visualization** — does the game have a frontend? If yes, plan path 1 (live web UI) or path 2 (replay-only).

Don't guess on any of these. Skipping the questions adds rework.

## Phase 1 — Learn both APIs in parallel

WebFetch in parallel:
- The **OpenReward** docs: https://docs.openreward.ai/environments/building-agentic-environments
- The **game's** docs (whatever URL the user gave) — ask "How do I create a game/state programmatically? What are Player/Agent classes? Is `decide()` sync or async? How do I get legal actions and apply one?"

Then probe the installed packages directly — docs lie or lag the source:

```bash
source venv/bin/activate && python -c "
import openreward; print([x for x in dir(openreward.environments) if not x.startswith('_')])
from openreward.environments import Environment, tool, ToolOutput, TextBlock
import inspect
print(inspect.getsource(Environment)[:3000])
"
```

```bash
source venv/bin/activate && python -c "
import <game_lib>
# create a game, dump state attrs, find legal-actions and apply-action APIs
"
```

Specifically discover:
- The game's **Player base class** and whether `decide()` is sync (`def`) or async.
- How to get **legal actions** at the current state (attribute? function call?).
- How to **advance one ply** without playing the whole game (e.g. `play_tick()`).
- How to detect **terminal** state and read the **winner**.
- How **VPs / score / reward proxies** are stored on state (for dense reward).

## Phase 2 — Design the env

Single `Environment` subclass. Three @tool methods are usually enough:

| Tool | Args | Purpose |
|------|------|---------|
| `get_state()` | — | Return compact text rendering of the board from the agent's perspective. |
| `list_legal_actions()` | — | Return numbered list of legal actions cached on the player. |
| `play_action(index: int)` | `PlayActionParams` | Apply chosen action, advance engine through opponents, return new state + step reward. |

### Custom Player pattern (no async queues needed when game.decide is sync)

```python
from <game_lib> import Player, RandomPlayer  # use real upstream class names

class OpenRewardPlayer(Player):
    def __init__(self, color):
        super().__init__(color)
        self.pending_action = None
        self.last_playable = []

    def decide(self, game, playable_actions):
        self.last_playable = list(playable_actions)
        if self.pending_action is None:
            raise RuntimeError("decide() called with no pending_action")
        a, self.pending_action = self.pending_action, None
        return a

    # CRITICAL: pickle-as-RandomPlayer so external consumers (e.g. a Docker
    # web server with no access to our package) can unpickle Game objects.
    def __reduce__(self):
        return (RandomPlayer, (self.color,))
```

The `__reduce__` trick saved hours when integrating catanatron's web UI — without it the upstream server crashed with `ModuleNotFoundError: No module named 'catan_env'` while unpickling. Always include it if the upstream library serializes the Game.

### Driving the engine from inside an async tool

Because `decide()` is sync, **no queues, no threads**:

1. `play_action` validates index, sets `self.player.pending_action = chosen`.
2. Calls `game.play_tick()` once — engine calls our `decide()` synchronously, gets the action, advances one ply.
3. Loops `play_tick()` while next decider is an opponent (their bots run synchronously too).
4. Stops when next decider is the agent OR `winning_color()` is set OR `max_ticks` exceeded.
5. Computes reward and returns `ToolOutput`.

If individual ticks are slow, wrap step 4 in `asyncio.to_thread`. Otherwise leave it sync.

### Dense reward template

```python
def _compute_reward(self) -> float:
    agent_score = score_of(self.game.state, self.agent_color)
    delta_agent = (agent_score - self._agent_score_prev) / WIN_THRESHOLD

    opp_deltas = [(score_of(s, c) - prev) / WIN_THRESHOLD
                  for c, prev in self._opp_score_prev.items()]
    opp_penalty = -0.5 * max(opp_deltas) if opp_deltas else 0.0

    reward = delta_agent + opp_penalty

    if self._terminal() and not self._terminal_reward_emitted:
        self._terminal_reward_emitted = True
        winner = self.game.winning_color()
        if winner == self.agent_color: reward += 1.0
        elif winner is not None:        reward -= 0.5
        # max-tick timeout: no extra bonus

    self._agent_score_prev = agent_score
    self._opp_score_prev = {c: score_of(self.game.state, c) for c in self._opp_score_prev}
    return max(-1.0, min(1.0, reward))
```

Track `_agent_score_prev` and `_opp_score_prev` in `setup()`.

## Phase 3 — Project layout

```
<repo>/
├── pyproject.toml
├── src/<env_name>/
│   ├── __init__.py
│   ├── agent_player.py          # OpenRewardPlayer with __reduce__
│   ├── env.py                   # CatanEnv-style Environment subclass
│   └── server.py                # Server([Env]).run() entry point
└── tests/
    ├── test_scripted_agent.py   # smoke
    └── test_claude_agent.py     # SDK e2e, saves trajectory
```

`pyproject.toml` deps: `openreward`, `<game_lib>`, `claude-agent-sdk`, `pydantic>=2`, plus `pytest pytest-asyncio` in `[project.optional-dependencies] dev`. Add `[tool.pytest.ini_options] asyncio_mode = "auto"`.

`__init__.py` re-exports the env class and player.

`server.py`:
```python
from openreward.environments import Server
from .env import MyEnv
def main(): Server([MyEnv]).run()
if __name__ == "__main__": main()
```

## Phase 4 — Scripted test (always run this first)

Pure Python, no LLM, runs in ~1s. Confirms the env loop terminates and rewards flow.

```python
import pytest
from <env_pkg>.env import MyEnv, PlayActionParams

@pytest.mark.asyncio
async def test_scripted_first_action_terminates():
    env = MyEnv(task_spec={"seed": 42, "max_ticks": 4000})
    env.setup()
    assert env.player.last_playable, "no legal actions at first decision"

    rewards, finished, steps = [], False, 0
    while not finished:
        steps += 1
        out = await env.play_action(PlayActionParams(index=0))
        if out.reward is not None: rewards.append(out.reward)
        finished = out.finished
        assert steps < 2000

    assert finished and rewards
    print(f"steps={steps} sum_reward={sum(rewards):.3f}")

@pytest.mark.asyncio
async def test_invalid_index_returns_error():
    env = MyEnv(task_spec={"seed": 1}); env.setup()
    n = len(env.player.last_playable)
    out = await env.play_action(PlayActionParams(index=n + 50))
    assert not out.finished and out.reward is not None and out.reward < 0
```

Run: `pytest tests/test_scripted_agent.py -v -s`. Fix any errors before going further. Common failures:
- `AttributeError: 'State' object has no attribute 'playable_actions'` — API drift; use `generate_playable_actions(state)` or whatever the current version exposes.
- `decide() called with no pending_action` — the env is calling `play_tick()` when it's the agent's turn; the loop should stop *before* the agent's tick and only advance after the tool sets the pending action.

## Phase 5 — Claude Agent SDK test (uses subscription, no API key)

The user usually does NOT want to use `ANTHROPIC_API_KEY` — they're logged in via Claude Code. Use `claude-agent-sdk` with an in-process MCP server.

```bash
pip install claude-agent-sdk
```

Tool naming convention: `mcp__<server_name>__<tool_name>`.

```python
import json, time
from pathlib import Path
from typing import Annotated
import pytest
from claude_agent_sdk import (
    AssistantMessage, ResultMessage, SystemMessage, TextBlock, ThinkingBlock,
    ToolResultBlock, ToolUseBlock, UserMessage,
    ClaudeAgentOptions, create_sdk_mcp_server, query, tool,
)
from <env_pkg>.env import MyEnv, PlayActionParams

TRAJ_DIR = Path(__file__).parent.parent / "trajectories"

# --- block + message serialization helpers (copy verbatim from catanatron repo) ---
def _block_to_dict(b): ...  # see catan_env reference
def _message_to_dict(m): ...

def _make_tools(env):
    @tool("get_state", "Describe the current state.", {})
    async def get_state(args):
        out = await env.get_state()
        return {"content": [{"type": "text", "text": out.blocks[0].text}]}

    @tool("list_legal_actions", "List indexed legal actions.", {})
    async def list_legal_actions(args):
        out = await env.list_legal_actions()
        return {"content": [{"type": "text", "text": out.blocks[0].text}]}

    @tool("play_action", "Apply a legal action by index.",
          {"index": Annotated[int, "Index from list_legal_actions."]})
    async def play_action(args):
        out = await env.play_action(PlayActionParams(index=int(args["index"])))
        return {"content": [{"type": "text", "text": out.blocks[0].text}]}

    return [get_state, list_legal_actions, play_action]

@pytest.mark.asyncio
async def test_claude_plays():
    env = MyEnv(task_spec={"seed": 7, "max_ticks": 4000})
    env.setup()
    server = create_sdk_mcp_server(name="<env_name>", tools=_make_tools(env))
    allowed = ["mcp__<env_name>__get_state",
               "mcp__<env_name>__list_legal_actions",
               "mcp__<env_name>__play_action"]
    stderr_lines = []
    options = ClaudeAgentOptions(
        system_prompt="You play <game> as ... use the tools to ...",
        mcp_servers={"<env_name>": server},
        allowed_tools=allowed,
        permission_mode="bypassPermissions",
        max_turns=30,                       # raise for full games (cost!)
        model="claude-sonnet-4-5",
        stderr=lambda l: stderr_lines.append(l),
    )

    trajectory = [{"kind": "config", "task_spec": env.task_spec, "allowed_tools": allowed}]
    cli_error, last_result_subtype = None, None
    try:
        async for msg in query(prompt=env.get_prompt()[0].text, options=options):
            trajectory.append({"kind": "message", **_message_to_dict(msg)})
            if isinstance(msg, ResultMessage):
                last_result_subtype = msg.subtype
    except Exception as e:
        cli_error = f"{type(e).__name__}: {e}"
        trajectory.append({"kind": "error", "error": cli_error,
                           "stderr_tail": stderr_lines[-50:]})

    trajectory.append({"kind": "summary", "ticks": env._ticks, ...})
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRAJ_DIR / f"claude_<game>_{int(time.time())}.json"
    out_path.write_text(json.dumps(trajectory, indent=2, default=str))
    print(f"trajectory saved to {out_path}")

    # error_max_turns is benign — model just hit the cap
    benign = {"success", "error_max_turns"}
    if cli_error and last_result_subtype not in benign:
        raise AssertionError(f"CLI failed: {cli_error}")
    assert env._ticks > 0
```

Run: `pytest tests/test_claude_agent.py -v -s`. Expect 1–3 minutes for `max_turns=30`.

## Phase 6 — Self-verify and analyze

After both tests pass, **read the trajectory file** and report to the user:
- Total LLM turns and game ticks reached.
- Number of `play_action` calls and the action sequence summary.
- Sum of step rewards.
- Final score per player and winner (if any).
- Cost from the final ResultMessage usage.
- Anything unusual (e.g. agent stuck in a loop, tool errors).

Use this snippet:

```python
import json, re
t = json.load(open(traj_path))
play_calls = []
rewards = []
for e in t:
    if e.get('kind') != 'message': continue
    if e.get('role') == 'assistant':
        for b in e['content']:
            if b['type'] == 'tool_use' and b['name'].endswith('play_action'):
                play_calls.append(b['input'])
    if e.get('role') == 'user' and isinstance(e.get('content'), list):
        for b in e['content']:
            if b.get('type') == 'tool_result':
                cont = b.get('content')
                if isinstance(cont, list):
                    for x in cont:
                        if isinstance(x, dict):
                            m = re.search(r'Step reward: ([+-]\d+\.\d+)', x.get('text',''))
                            if m: rewards.append(float(m.group(1)))
print(f"play_action calls: {len(play_calls)}, sum reward: {sum(rewards):+.3f}")
```

## Phase 7 (optional) — Visualization via the game's web UI

Only do this if (a) the game ships a web UI and (b) the user explicitly asks.

### Path 1: live UI (preferred when the game has Docker compose)

1. Clone the upstream game repo and `docker compose up -d`. Three containers usually: `db` (postgres), `server` (flask/api), `react-ui`.
2. Check ports — host conflicts are common. Native postgres on 5432 will steal the docker DB; remap with a `docker-compose.override.yml`:
   ```yaml
   services:
     db:
       ports: ["5433:5432"]
   ```
3. Use `127.0.0.1` not `localhost` in DATABASE_URL — IPv6 `::1` may resolve to a different postgres.
4. Add a `visualize: bool` flag to env's `task_spec`. In `play_action` after each tick, if visualize, call the game's persistence helper (catanatron: `from catanatron.web.utils import ensure_link; url = ensure_link(self.game)`) and append `View: {url}` to the tool result text.
5. **Version sync**: the docker stack and your venv must run the same git commit of the game library (pickle compatibility). If `pip install <lib>` resolves an older release than `git clone master`, do `pip install -e /path/to/clone[web] --force-reinstall --no-deps` to align them.
6. The game's React UI usually has multiple routes:
   - `/games/<id>/states/<n>` — frozen single snapshot
   - `/replays/<id>` — full step-through replay (this is what users want; default to emitting this URL)
   Find them in `ui/src/App.tsx` of the game repo.

### Path 2: replay-only (no Docker)

Save the action history (`(seed, [serialized_actions])`) per game. Provide a `replay.py` that re-instantiates `Game(seed=...)`, replays each action, and prints a text-mode board between steps. Lighter weight, no Docker, but no graphical view.

## Known gotchas (debug shortcuts)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: <env_pkg>` from upstream server | Upstream pickles the Game with our custom Player | Add `__reduce__` returning a stock player class |
| `AttributeError: 'State' object has no attribute 'playable_actions'` | Game library version drift | Find the new API (e.g. `generate_playable_actions(state)`); inspect installed source |
| `Exception: Command failed with exit code 1 / Error output: Check stderr output for details` after Claude run | `error_max_turns` from SDK | Treat as benign in the assertion (`last_result_subtype in {"success","error_max_turns"}`) |
| `tools` field requires schema | Anthropic SDK path: ToolSpec.input_schema is None for no-arg tools | Coerce None → `{"type":"object","properties":{}}` |
| Claude's first turn calls `ToolSearch` before our MCP tools | Claude Code defers MCP tool schemas | Cosmetic; one wasted turn |
| `psycopg2.OperationalError: role "<x>" does not exist` | Hit the wrong postgres (host's instead of docker's) | Use `127.0.0.1` and remap docker port to 5433 |
| `discard_counts` / similar attribute errors from upstream API | Pickle version mismatch | Reinstall venv lib from the same commit as docker image |

## Authentication for Claude SDK

`claude-agent-sdk` shells out to the `claude` CLI, which uses the user's existing Claude Code subscription auth. **Do not** require `ANTHROPIC_API_KEY`. If the user has the `claude` CLI working in a terminal, the SDK works.

## Reference implementation

`/Users/julie3399/rl-env-hackathon-complex-worlds/` — full catanatron build with all files mentioned above. When in doubt, copy from there and adapt names.
