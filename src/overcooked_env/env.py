import random
from typing import Optional

from pydantic import BaseModel, Field
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as _OvercookedEnv
from overcooked_ai_py.mdp.actions import Action

from openreward.environments import (
    Environment,
    TextBlock,
    ToolOutput,
    tool,
)


LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit",
]

ACTION_NAMES = {
    0: "NORTH (move up)",
    1: "SOUTH (move down)",
    2: "EAST (move right)",
    3: "WEST (move left)",
    4: "STAY",
    5: "INTERACT",
}


class PlayActionParams(BaseModel):
    index: int = Field(..., description="Action index 0-5: 0=NORTH, 1=SOUTH, 2=EAST, 3=WEST, 4=STAY, 5=INTERACT")


def _format_state(env: _OvercookedEnv) -> str:
    grid = env.mdp.state_string(env.state)
    t = env.state.timestep
    lines = [
        f"Timestep: {t}/{env.horizon}",
        "Grid (you are player 0):",
        grid,
    ]
    return "\n".join(lines)


class OvercookedEnv(Environment):
    """Overcooked cooperative cooking environment.

    The agent controls player 0. Player 1 is a RandomAgent with all_actions=True.
    Goal: deliver as many soups as possible within the horizon.
    Each soup delivery gives +20 sparse reward. Shaped rewards are also given
    for useful intermediate actions (dish pickup, soup pickup, ingredient placement).
    """

    def __init__(self, task_spec=None, secrets=None):
        super().__init__(task_spec or {}, secrets or {})
        self.layout: str = self.task_spec.get("layout", "cramped_room")
        self.horizon: int = int(self.task_spec.get("horizon", 400))
        self._env: Optional[_OvercookedEnv] = None
        self._done: bool = False
        self._soups_delivered: int = 0
        self._step_count: int = 0

    def setup(self):
        mdp = OvercookedGridworld.from_layout_name(self.layout)
        self._env = _OvercookedEnv.from_mdp(mdp, horizon=self.horizon, info_level=0)
        self._env.reset()
        self._done = False
        self._soups_delivered = 0
        self._step_count = 0

    def teardown(self):
        self._env = None

    def get_prompt(self):
        text = (
            f"You are playing Overcooked on the '{self.layout}' layout as player 0.\n"
            "Work cooperatively with player 1 (an AI partner) to cook and deliver soups.\n\n"
            f"Each soup delivery scores +20 points. You have {self.horizon} timesteps.\n\n"
            "Actions (always all 6 available):\n"
            + "\n".join(f"  {i}: {name}" for i, name in ACTION_NAMES.items())
            + "\n\nLegend: ↑/↓/←/→ = player facing direction, digit = player index, "
            "O=onion dispenser, T=tomato, D=dish dispenser, P=pot, S=serving station, "
            "X=counter, ' '=floor\n\n"
            "On each timestep:\n"
            "  1. Call `get_state` to see the current grid.\n"
            "  2. Call `list_legal_actions` to see your 6 indexed options.\n"
            "  3. Call `play_action` with your chosen index to advance one step.\n\n"
            "Initial state:\n"
            + _format_state(self._env)
        )
        return [TextBlock(text=text)]

    @classmethod
    def list_tasks(cls, split: str):
        if split == "train":
            return [{"layout": layout, "horizon": 400} for layout in LAYOUTS]
        if split == "test":
            return [{"layout": "cramped_room", "horizon": 400}]
        return []

    @classmethod
    def list_splits(cls):
        return ["train", "test"]

    @tool
    async def get_state(self) -> ToolOutput:
        """Return the current ASCII grid state of the kitchen."""
        return ToolOutput(
            blocks=[TextBlock(text=_format_state(self._env))],
            finished=self._done,
        )

    @tool
    async def list_legal_actions(self) -> ToolOutput:
        """Return the 6 always-available actions with their indices."""
        if self._done:
            return ToolOutput(
                blocks=[TextBlock(text="Episode is over; no actions available.")],
                finished=True,
            )
        body = "\n".join(f"{i}: {name}" for i, name in ACTION_NAMES.items())
        return ToolOutput(
            blocks=[TextBlock(text=f"6 actions available:\n{body}")],
            finished=False,
        )

    @tool
    async def play_action(self, params: PlayActionParams) -> ToolOutput:
        """Apply action by index (0-5) and advance one timestep."""
        if self._done:
            return ToolOutput(
                blocks=[TextBlock(text="Episode already over.")],
                reward=0.0,
                finished=True,
            )
        if not (0 <= params.index <= 5):
            return ToolOutput(
                blocks=[TextBlock(text=f"Invalid index {params.index}; must be in [0, 5].")],
                reward=-0.05,
                finished=False,
            )

        agent_action = Action.INDEX_TO_ACTION[params.index]
        partner_action = random.choice(Action.ALL_ACTIONS)
        joint_action = (agent_action, partner_action)

        _next_state, sparse_r, done, env_info = self._env.step(joint_action)
        self._done = done
        self._step_count += 1

        shaped_r = env_info["shaped_r_by_agent"][0]
        step_reward = max(-1.0, min(1.0, (sparse_r + shaped_r) / 20.0))

        if sparse_r > 0:
            self._soups_delivered += int(round(sparse_r / 20.0))

        action_name = ACTION_NAMES[params.index]
        text = (
            f"Action: {action_name}\n"
            f"Sparse reward: {sparse_r:.1f}  Shaped reward (agent 0): {shaped_r:.3f}\n"
            f"Step reward (normalized): {step_reward:+.4f}\n"
            f"Soups delivered so far: {self._soups_delivered}\n"
            f"Timestep: {self._env.state.timestep}/{self.horizon}\n"
            "--- State ---\n"
            + _format_state(self._env)
        )
        if done:
            text += f"\n\nEpisode complete. Total soups delivered: {self._soups_delivered}"

        return ToolOutput(
            blocks=[TextBlock(text=text)],
            reward=step_reward,
            finished=done,
            metadata={
                "action": action_name,
                "sparse_r": sparse_r,
                "shaped_r": float(shaped_r),
                "step": self._step_count,
                "soups_delivered": self._soups_delivered,
            },
        )
