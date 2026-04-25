from typing import Optional

from pydantic import BaseModel, Field
from catanatron import Game, RandomPlayer, Color
from catanatron.models.actions import generate_playable_actions
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.minimax import AlphaBetaPlayer

from openreward.environments import (
    Environment,
    TextBlock,
    ToolOutput,
    tool,
)

from .agent_player import OpenRewardPlayer


COLORS = {
    "RED": Color.RED,
    "BLUE": Color.BLUE,
    "WHITE": Color.WHITE,
    "ORANGE": Color.ORANGE,
}


# Difficulty presets. Each entry is (opponent_class, num_opponents, vps_to_win, friendly_robber).
# Higher tiers = harder for the LLM agent.
DIFFICULTY = {
    "very_easy": (RandomPlayer, 1, 5, True),
    "easy":      (RandomPlayer, 3, 8, True),
    "medium":    (WeightedRandomPlayer, 3, 10, False),
    "hard":      (ValueFunctionPlayer, 3, 10, False),
    "very_hard": (AlphaBetaPlayer, 3, 10, False),
}


class PlayActionParams(BaseModel):
    index: int = Field(..., description="Index into the list returned by list_legal_actions.")


def _vp(state, color) -> int:
    idx = state.color_to_index[color]
    return int(state.player_state[f"P{idx}_VICTORY_POINTS"])


def _resources(state, color) -> dict[str, int]:
    idx = state.color_to_index[color]
    return {
        r: int(state.player_state[f"P{idx}_{r}_IN_HAND"])
        for r in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
    }


def _format_state(game, agent_color) -> str:
    state = game.state
    vps = {c.value: _vp(state, c) for c in state.colors}
    res = _resources(state, agent_color)
    cur = state.current_color()
    lines = [
        f"Turn: {state.num_turns}  current_player={cur.value}  agent={agent_color.value}",
        "VPs: " + " ".join(f"{c}={v}" for c, v in vps.items()),
        "Your resources: " + " ".join(f"{k}={v}" for k, v in res.items()),
        f"Initial build phase: {state.is_initial_build_phase}",
    ]
    if game.winning_color() is not None:
        lines.append(f"GAME OVER. Winner: {game.winning_color().value}")
    return "\n".join(lines)


class CatanEnv(Environment):
    """Settlers of Catan environment driven by Catanatron.

    The agent controls one seat (default RED). The other three seats are
    catanatron RandomPlayers. Each tool call advances the game until the
    next decision belongs to the agent again, or the game ends.
    """

    def __init__(self, task_spec=None, secrets=None):
        super().__init__(task_spec or {}, secrets or {})
        self.seed: Optional[int] = self.task_spec.get("seed")
        self.agent_color: Color = COLORS[self.task_spec.get("agent_color", "RED")]
        self.max_ticks: int = int(self.task_spec.get("max_ticks", 1000))
        self.visualize: bool = bool(self.task_spec.get("visualize", False))
        self.difficulty: str = str(self.task_spec.get("difficulty", "medium"))
        if self.difficulty not in DIFFICULTY:
            raise ValueError(f"Unknown difficulty {self.difficulty!r}; choose from {list(DIFFICULTY)}")
        opp_cls, num_opp, vps, friendly = DIFFICULTY[self.difficulty]
        self.opp_cls = opp_cls
        self.num_opponents: int = int(num_opp)
        self.vps_to_win: int = int(vps)
        self.friendly_robber: bool = bool(friendly)
        self.player: Optional[OpenRewardPlayer] = None
        self.game: Optional[Game] = None
        self._ticks = 0
        self._agent_vp_prev = 0
        self._opp_vp_prev: dict[Color, int] = {}
        self._terminal_reward_emitted = False
        self._cum_reward: float = 0.0
        self.last_view_url: Optional[str] = None

    def setup(self):
        self.player = OpenRewardPlayer(self.agent_color)
        all_other = [c for c in (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE) if c != self.agent_color]
        opp_colors = all_other[: self.num_opponents]
        players = [self.player] + [self.opp_cls(c) for c in opp_colors]
        self.game = Game(
            players,
            seed=self.seed,
            vps_to_win=self.vps_to_win,
            friendly_robber=self.friendly_robber,
        )
        self._advance_until_agent_or_end()
        self._agent_vp_prev = _vp(self.game.state, self.agent_color)
        self._opp_vp_prev = {c: _vp(self.game.state, c) for c in self.game.state.colors if c != self.agent_color}
        self._publish_view()

    def _publish_view(self):
        """If visualize=True, upsert game state to catanatron's web DB and store the URL."""
        if not self.visualize or self.game is None:
            return
        try:
            from catanatron.web.utils import ensure_link
            self.last_view_url = ensure_link(self.game)
        except Exception as e:
            self.last_view_url = f"<viz error: {type(e).__name__}: {e}>"

    def _next_decider_is_agent(self) -> bool:
        return self.game.state.current_color() == self.agent_color

    def _advance_until_agent_or_end(self):
        # Tick the engine forward through any opponent decisions until either it's the
        # agent's turn or the game ends. If the next decider is already the agent, do
        # nothing — they need to call list_legal_actions / play_action.
        while (
            self.game.winning_color() is None
            and self._ticks < self.max_ticks
            and not self._next_decider_is_agent()
        ):
            self.game.play_tick()
            self._ticks += 1
        # If we're at the agent's seat, calling play_tick would consume pending_action.
        # We instead need to populate `last_playable` so list_legal_actions has data.
        # catanatron exposes the current legal actions on state.playable_actions.
        if self._next_decider_is_agent() and self.game.winning_color() is None:
            self.player.last_playable = list(generate_playable_actions(self.game.state))

    def _terminal(self) -> bool:
        return self.game.winning_color() is not None or self._ticks >= self.max_ticks

    def _compute_reward(self) -> float:
        agent_vp = _vp(self.game.state, self.agent_color)
        delta_agent = (agent_vp - self._agent_vp_prev) / 10.0

        opp_deltas = []
        for c, prev in self._opp_vp_prev.items():
            now = _vp(self.game.state, c)
            opp_deltas.append((now - prev) / 10.0)
        opp_penalty = -0.5 * max(opp_deltas) if opp_deltas else 0.0

        reward = delta_agent + opp_penalty

        if self._terminal() and not self._terminal_reward_emitted:
            self._terminal_reward_emitted = True
            winner = self.game.winning_color()
            if winner == self.agent_color:
                reward += 1.0
            elif winner is not None:
                reward -= 0.5
            # max-tick timeout: no extra bonus

        # update prev
        self._agent_vp_prev = agent_vp
        self._opp_vp_prev = {c: _vp(self.game.state, c) for c in self.game.state.colors if c != self.agent_color}

        return max(-1.0, min(1.0, reward))

    def get_prompt(self):
        opp_name = self.opp_cls.__name__
        robber_note = (
            "Robber is FRIENDLY (it blocks production but doesn't steal cards)."
            if self.friendly_robber
            else "Robber is HOSTILE (it blocks production AND steals one card from a victim)."
        )
        text = (
            f"You are playing Settlers of Catan as the {self.agent_color.value} player "
            f"against {self.num_opponents} {opp_name} opponent(s). "
            f"Win by reaching {self.vps_to_win} victory points (VP). "
            f"{robber_note} "
            "VP come from settlements (+1), cities (+2), longest road (+2), "
            "largest army (+2), and VP development cards (+1).\n\n"
            "On each of your turns:\n"
            "  1. Call `list_legal_actions` to see indexed action options.\n"
            "  2. Call `play_action` with the chosen index.\n"
            "  3. Call `get_state` whenever you want a board summary.\n\n"
            "The tool result for `play_action` advances the engine through opponent moves "
            "and returns the next state plus your step reward.\n\n"
            "Initial state:\n" + _format_state(self.game, self.agent_color)
        )
        return [TextBlock(text=text)]

    @classmethod
    def list_tasks(cls, split: str):
        if split == "train":
            return [{"seed": s, "agent_color": "RED", "max_ticks": 2000} for s in range(5)]
        if split == "test":
            return [{"seed": s, "agent_color": "RED", "max_ticks": 2000} for s in range(100, 103)]
        return []

    @classmethod
    def list_splits(cls):
        return ["train", "test"]

    @tool
    async def get_state(self) -> ToolOutput:
        """Return a textual summary of the current board state from the agent's perspective."""
        return ToolOutput(
            blocks=[TextBlock(text=_format_state(self.game, self.agent_color))],
            finished=self._terminal(),
        )

    @tool
    async def list_legal_actions(self) -> ToolOutput:
        """Return the numbered list of legal actions you may pass to play_action."""
        if self._terminal():
            return ToolOutput(
                blocks=[TextBlock(text="Game is over; no legal actions.")],
                finished=True,
            )
        actions = self.player.last_playable
        body = "\n".join(f"{i}: {a}" for i, a in enumerate(actions))
        return ToolOutput(
            blocks=[TextBlock(text=f"{len(actions)} legal action(s):\n{body}")],
        )

    @tool
    async def play_action(self, params: PlayActionParams) -> ToolOutput:
        """Apply one of the legal actions (by index from list_legal_actions) and advance the game."""
        if self._terminal():
            return ToolOutput(
                blocks=[TextBlock(text="Game already over.")],
                reward=0.0,
                finished=True,
            )
        actions = self.player.last_playable
        if not actions:
            return ToolOutput(
                blocks=[TextBlock(text="No legal actions available; this should not happen on your turn.")],
                reward=0.0,
                finished=self._terminal(),
            )
        if not (0 <= params.index < len(actions)):
            return ToolOutput(
                blocks=[TextBlock(text=f"Invalid index {params.index}; must be in [0, {len(actions)}).")],
                reward=-0.05,
                finished=False,
            )

        chosen = actions[params.index]
        self.player.pending_action = chosen
        # One tick consumes our action.
        self.game.play_tick()
        self._ticks += 1
        # Then run opponents until our next turn (or end / cap).
        self._advance_until_agent_or_end()

        reward = self._compute_reward()
        self._cum_reward += reward
        finished = self._terminal()
        self._publish_view()

        text = f"Applied: {chosen}\nStep reward: {reward:+.3f}\n--- State ---\n{_format_state(self.game, self.agent_color)}"
        if finished:
            winner = self.game.winning_color()
            if winner is None:
                text += f"\n(Reached max_ticks={self.max_ticks}; treated as timeout.)"
        if self.last_view_url:
            text += f"\nView: {self.last_view_url}"
        return ToolOutput(
            blocks=[TextBlock(text=text)],
            reward=reward,
            finished=finished,
            metadata={"chosen_action": str(chosen), "ticks": self._ticks, "view_url": self.last_view_url},
        )
