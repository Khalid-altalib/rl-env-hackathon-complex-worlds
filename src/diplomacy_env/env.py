"""OpenReward environment wrapping the `diplomacy` package.

The agent controls one Great Power (default FRANCE). The other six powers play
random legal orders. Each `play_action` call submits ONE order for ONE of the
agent's orderable locations; once the agent has chosen orders for all of its
orderable locations in the current phase, the env fills in random orders for
all other powers, calls `Game.process()`, and advances to the next phase.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from diplomacy import Game

from openreward.environments import (
    Environment,
    TextBlock,
    ToolOutput,
    tool,
)


WIN_THRESHOLD = 18  # supply centers required to win standard Diplomacy
ALL_POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]


class PlayActionParams(BaseModel):
    index: int = Field(
        ...,
        description="0-based index into the most recent `list_legal_actions` output for the current orderable location.",
    )


def _phase_year(phase: str) -> int:
    """Extract year from phase abbr like 'S1901M' or 'COMPLETED'."""
    if not phase or not phase[0].isalpha() or len(phase) < 5:
        return 0
    try:
        return int(phase[1:5])
    except ValueError:
        return 0


def _format_state(env: "DiplomacyEnv") -> str:
    g = env.game
    phase = g.get_current_phase()
    sc_lines = []
    for p in ALL_POWERS:
        sc_count = len(g.get_centers(p))
        units = g.get_units(p)
        marker = " <- you" if p == env.agent_power else ""
        sc_lines.append(f"  {p:8s}  SCs={sc_count:2d}  units={len(units):2d}{marker}")

    agent_units = g.get_units(env.agent_power)
    agent_centers = g.get_centers(env.agent_power)

    pending_loc = env._orderable_queue[0] if env._orderable_queue else None
    queue_remaining = len(env._orderable_queue)
    chosen = ", ".join(f"{loc}: {ord_}" for loc, ord_ in env._pending_orders.items()) or "(none)"

    last_phase_summary = ""
    if g.order_history:
        last_phase = list(g.order_history.keys())[-1]
        agent_results = g.result_history.get(last_phase, {})
        if agent_results:
            agent_lines = []
            for unit, results in agent_results.items():
                res_str = ", ".join(str(r) for r in results) if results else "ok"
                agent_lines.append(f"    {unit}: {res_str}")
            last_phase_summary = f"\nLast phase ({last_phase}) results (all powers):\n" + "\n".join(agent_lines[:20])

    parts = [
        f"Phase: {phase}  (you are {env.agent_power})",
        f"Year cap: {env.max_year}  (game truncates after the year {env.max_year} completes)",
        "",
        "Supply centers / units per power:",
        *sc_lines,
        "",
        f"Your units: {agent_units}",
        f"Your supply centers: {agent_centers}",
        "",
        f"Currently deciding order for: {pending_loc}  ({queue_remaining} location(s) left this phase)",
        f"Orders you have already chosen this phase: {chosen}",
    ]
    if last_phase_summary:
        parts.append(last_phase_summary)
    return "\n".join(parts)


class DiplomacyEnv(Environment):
    """Single-agent Diplomacy environment.

    task_spec keys:
      - agent_power: str, one of ALL_POWERS (default 'FRANCE')
      - seed: int, RNG seed for opponent random orders (default 0)
      - max_year: int, year cap; episode truncates after this year (default 1910)
      - render_dir: optional str path; if set, write per-phase SVG snapshots there
        (one SVG before any orders + one after each processed phase) plus index.html
    """

    def __init__(self, task_spec=None, secrets=None):
        super().__init__(task_spec or {}, secrets or {})
        self.agent_power: str = self.task_spec.get("agent_power", "FRANCE")
        if self.agent_power not in ALL_POWERS:
            raise ValueError(f"agent_power must be one of {ALL_POWERS}, got {self.agent_power!r}")
        self.seed: int = int(self.task_spec.get("seed", 0))
        self.max_year: int = int(self.task_spec.get("max_year", 1910))
        rd = self.task_spec.get("render_dir")
        self.render_dir: Optional[Path] = Path(rd) if rd else None

        self.game: Optional[Game] = None
        self._rng: Optional[random.Random] = None
        self._orderable_queue: list[str] = []
        self._pending_orders: dict[str, str] = {}
        self._prev_sc: dict[str, int] = {}
        self._terminal_reward_emitted: bool = False
        self._phases_processed: int = 0
        self._play_action_calls: int = 0
        self._render_files: list[tuple[str, str]] = []  # (filename, phase_label)

    def setup(self):
        self._rng = random.Random(self.seed)
        self.game = Game()
        self._pending_orders = {}
        self._terminal_reward_emitted = False
        self._phases_processed = 0
        self._play_action_calls = 0
        self._prev_sc = {p: len(self.game.get_centers(p)) for p in ALL_POWERS}
        self._render_files = []
        if self.render_dir is not None:
            self.render_dir.mkdir(parents=True, exist_ok=True)
            self._render_snapshot(label="initial")
        self._refresh_queue_skipping_empty()

    def teardown(self):
        self.game = None

    def get_prompt(self):
        text = (
            f"You are playing Diplomacy as {self.agent_power}.\n"
            "Diplomacy is a 7-power strategic board game on a map of pre-WWI Europe. "
            "The 6 other powers in this env play uniformly random legal orders. "
            "Win condition: control 18 supply centers. You also lose if you are eliminated (0 SCs).\n"
            f"Episode truncates after year {self.max_year}.\n\n"
            "Each turn, every unit you control needs an order. This env asks you for ONE order at a time:\n"
            "  - get_state          -> shows board, current orderable location, and orders chosen so far this phase\n"
            "  - list_legal_actions -> numbered list of legal orders for the current location\n"
            "  - play_action(index) -> chooses that order; advances to the next of your locations,\n"
            "                          or processes the phase (random orders for the 6 other powers, then adjudication)\n\n"
            "Order syntax (for reading the action list):\n"
            "  'A PAR H'             army in PAR holds\n"
            "  'A PAR - BUR'         army in PAR moves to BUR\n"
            "  'F LON - NTH'         fleet in LON moves to NTH\n"
            "  'A MAR S A PAR - BUR' army in MAR supports A PAR moving to BUR\n"
            "  'A MAR S A PAR'       army in MAR supports A PAR holding\n"
            "  'F LON C A YOR - NWY' fleet in LON convoys A YOR to NWY (army needs '-VIA' move)\n"
            "  'A PAR R BUR'         retreat (Retreats phase)\n"
            "  'A PAR D'             disband (Retreats or Adjustments phase)\n"
            "  'A PAR B'             build new army at home center (Adjustments phase)\n"
            "  'WAIVE'               skip a build (Adjustments phase)\n\n"
            "Reward is supply-center delta per processed phase, normalized; +1 for winning, -1 for elimination.\n"
            "Initial state:\n"
            + _format_state(self)
        )
        return [TextBlock(text=text)]

    @classmethod
    def list_tasks(cls, split: str):
        if split == "train":
            return [
                {"seed": s, "agent_power": p, "max_year": 1910}
                for p in ["FRANCE", "ENGLAND", "GERMANY"]
                for s in range(5)
            ]
        if split == "test":
            return [
                {"seed": 100, "agent_power": "FRANCE", "max_year": 1910},
                {"seed": 101, "agent_power": "ENGLAND", "max_year": 1910},
            ]
        return []

    @classmethod
    def list_splits(cls):
        return ["train", "test"]

    # ---------- internal helpers ----------

    def _terminal(self) -> bool:
        if self.game is None:
            return False
        if self.game.is_game_done:
            return True
        if len(self.game.get_centers(self.agent_power)) == 0 and self._phases_processed > 0:
            return True
        if _phase_year(self.game.get_current_phase()) > self.max_year:
            return True
        return False

    def _winner(self) -> Optional[str]:
        if self.game is None or not self.game.is_game_done:
            return None
        outcome = self.game.outcome  # [last_phase, winner1, winner2, ...]
        if outcome and len(outcome) > 1:
            return outcome[1]
        return None

    def _advance_phase(self):
        """Submit pending agent orders, sample random orders for opponents, process."""
        # Agent's own orders
        self.game.set_orders(self.agent_power, list(self._pending_orders.values()))
        # Other powers: random legal orders
        po = self.game.get_all_possible_orders()
        for power in ALL_POWERS:
            if power == self.agent_power:
                continue
            locs = self.game.get_orderable_locations(power)
            orders = []
            for loc in locs:
                choices = po.get(loc, [])
                if choices:
                    orders.append(self._rng.choice(choices))
            self.game.set_orders(power, orders)

        # Render the pre-process state with all orders shown so the SVG includes arrows.
        if self.render_dir is not None:
            self._render_snapshot(label="orders", incl_orders=True)

        self.game.process()
        self._phases_processed += 1
        self._pending_orders = {}

        # Render the post-process state (no pending orders, just the resulting board).
        if self.render_dir is not None:
            self._render_snapshot(label="result", incl_orders=False)
            self._write_index_html()

    def _render_snapshot(self, label: str, incl_orders: bool = True):
        """Write one SVG of the current game state to self.render_dir."""
        idx = len(self._render_files)
        phase = self.game.get_current_phase()
        fname = f"phase_{idx:03d}_{phase}_{label}.svg"
        out_path = self.render_dir / fname
        try:
            self.game.render(incl_orders=incl_orders, output_path=str(out_path))
        except Exception as e:
            # Don't let rendering failures break the env.
            log = self.render_dir / "render_errors.log"
            prev = log.read_text() if log.exists() else ""
            log.write_text(prev + f"{fname}: {type(e).__name__}: {e}\n")
            return
        self._render_files.append((fname, f"{phase} ({label})"))

    def _write_index_html(self):
        """Write a tiny index.html that lists all rendered SVGs and embeds the latest."""
        if not self._render_files:
            return
        items = "\n".join(
            f'<li><a href="{f}" target="viewer">{label}</a></li>'
            for f, label in self._render_files
        )
        latest = self._render_files[-1][0]
        html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Diplomacy replay</title>
<style>
  body {{ font: 14px/1.4 system-ui, sans-serif; margin: 0; display: flex; height: 100vh; }}
  nav  {{ width: 260px; overflow: auto; padding: 12px; border-right: 1px solid #ddd; }}
  main {{ flex: 1; }}
  iframe {{ width: 100%; height: 100%; border: 0; }}
  ol {{ padding-left: 18px; }}
  a {{ text-decoration: none; color: #06c; }}
  a:hover {{ text-decoration: underline; }}
  h1 {{ font-size: 14px; margin: 0 0 8px; }}
</style></head>
<body>
  <nav>
    <h1>Phases ({len(self._render_files)}) — agent: {self.agent_power}</h1>
    <ol>{items}</ol>
  </nav>
  <main><iframe name="viewer" src="{latest}"></iframe></main>
</body></html>"""
        (self.render_dir / "index.html").write_text(html)

    def _refresh_queue_skipping_empty(self):
        """Refresh orderable queue, auto-advancing past phases where the agent has nothing to do."""
        # Loop because: if agent has no orderable locations this phase, we need to
        # process the phase (with empty agent orders + random opponent orders) and check the next.
        guard = 0
        while True:
            guard += 1
            if guard > 50:
                # Hard safety; shouldn't happen.
                self._orderable_queue = []
                return
            if self._terminal():
                self._orderable_queue = []
                return
            locs = self.game.get_orderable_locations(self.agent_power)
            if locs:
                # Stable order so action indices are deterministic for the agent.
                self._orderable_queue = sorted(locs)
                return
            # Agent has nothing to order this phase: advance with empty agent orders.
            self._advance_phase()

    def _compute_reward(self) -> float:
        agent_sc = len(self.game.get_centers(self.agent_power))
        delta_agent = (agent_sc - self._prev_sc[self.agent_power]) / WIN_THRESHOLD

        opp_deltas = []
        for p in ALL_POWERS:
            if p == self.agent_power:
                continue
            d = (len(self.game.get_centers(p)) - self._prev_sc[p]) / WIN_THRESHOLD
            opp_deltas.append(d)
        opp_penalty = -0.5 * max(opp_deltas) if opp_deltas else 0.0

        reward = delta_agent + opp_penalty

        if self._terminal() and not self._terminal_reward_emitted:
            self._terminal_reward_emitted = True
            winner = self._winner()
            if winner == self.agent_power:
                reward += 1.0
            elif winner is not None:
                reward -= 1.0
            elif agent_sc == 0:
                reward -= 1.0
            # truncation by year cap with no winner: no extra bonus

        # Update prev SCs for next delta computation.
        for p in ALL_POWERS:
            self._prev_sc[p] = len(self.game.get_centers(p))

        return max(-1.0, min(1.0, reward))

    # ---------- tools ----------

    @tool
    async def get_state(self) -> ToolOutput:
        """Return the current Diplomacy board state from the agent's perspective."""
        return ToolOutput(
            blocks=[TextBlock(text=_format_state(self))],
            finished=self._terminal(),
        )

    @tool
    async def list_legal_actions(self) -> ToolOutput:
        """Return the legal orders for the agent's current orderable location, indexed."""
        if self._terminal():
            return ToolOutput(
                blocks=[TextBlock(text="Episode is over; no actions available.")],
                finished=True,
            )
        if not self._orderable_queue:
            return ToolOutput(
                blocks=[TextBlock(text="No orderable locations (this should not happen mid-episode).")],
                finished=True,
            )
        loc = self._orderable_queue[0]
        po = self.game.get_all_possible_orders()
        choices = po.get(loc, [])
        if not choices:
            return ToolOutput(
                blocks=[TextBlock(text=f"No legal orders for {loc}; this is unexpected.")],
                finished=True,
            )
        body = "\n".join(f"  {i}: {ord_}" for i, ord_ in enumerate(choices))
        text = (
            f"Legal orders for {loc}  (phase {self.game.get_current_phase()}, {len(choices)} options):\n"
            f"{body}"
        )
        return ToolOutput(blocks=[TextBlock(text=text)], finished=False)

    @tool
    async def play_action(self, params: PlayActionParams) -> ToolOutput:
        """Choose order at `index` for the current orderable location and advance the env."""
        self._play_action_calls += 1

        if self._terminal():
            return ToolOutput(
                blocks=[TextBlock(text="Episode already over.")],
                reward=0.0,
                finished=True,
            )
        if not self._orderable_queue:
            return ToolOutput(
                blocks=[TextBlock(text="No orderable locations to play.")],
                reward=0.0,
                finished=True,
            )

        loc = self._orderable_queue[0]
        po = self.game.get_all_possible_orders()
        choices = po.get(loc, [])
        if not (0 <= params.index < len(choices)):
            return ToolOutput(
                blocks=[
                    TextBlock(
                        text=(
                            f"Invalid index {params.index}; {loc} has {len(choices)} legal orders "
                            f"(valid range 0..{len(choices)-1})."
                        )
                    )
                ],
                reward=-0.05,
                finished=False,
            )

        chosen = choices[params.index]
        self._pending_orders[loc] = chosen
        self._orderable_queue.pop(0)

        # If still more agent locations to order this phase, return without processing or reward.
        if self._orderable_queue:
            next_loc = self._orderable_queue[0]
            text = (
                f"Recorded order: {chosen}\n"
                f"Still need orders for {len(self._orderable_queue)} more location(s) this phase. "
                f"Next location: {next_loc}.\n"
                f"--- State ---\n{_format_state(self)}"
            )
            return ToolOutput(
                blocks=[TextBlock(text=text)],
                reward=None,
                finished=False,
                metadata={
                    "loc": loc,
                    "order": chosen,
                    "phase_processed": False,
                    "play_action_call": self._play_action_calls,
                },
            )

        # Last orderable location: process the phase.
        prev_phase = self.game.get_current_phase()
        self._advance_phase()
        self._refresh_queue_skipping_empty()

        reward = self._compute_reward()
        finished = self._terminal()

        agent_sc = len(self.game.get_centers(self.agent_power))
        text_lines = [
            f"Recorded order: {chosen}",
            f"Phase {prev_phase} processed. New phase: {self.game.get_current_phase()}",
            f"Step reward: {reward:+.4f}   Your SCs: {agent_sc}",
        ]
        if finished:
            winner = self._winner()
            if winner == self.agent_power:
                text_lines.append("YOU WON: 18+ supply centers.")
            elif winner is not None:
                text_lines.append(f"GAME OVER: {winner} won with 18+ supply centers.")
            elif agent_sc == 0:
                text_lines.append("YOU WERE ELIMINATED.")
            elif _phase_year(self.game.get_current_phase()) > self.max_year:
                text_lines.append(f"TRUNCATED: year cap {self.max_year} exceeded.")
        text_lines.append("--- State ---")
        text_lines.append(_format_state(self))

        return ToolOutput(
            blocks=[TextBlock(text="\n".join(text_lines))],
            reward=reward,
            finished=finished,
            metadata={
                "loc": loc,
                "order": chosen,
                "phase_processed": True,
                "phase_after": self.game.get_current_phase(),
                "phases_processed": self._phases_processed,
                "agent_sc": agent_sc,
                "play_action_call": self._play_action_calls,
            },
        )
