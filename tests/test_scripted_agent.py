"""Scripted-agent smoke test: always pick the first legal action.

Verifies the env wires together correctly: setup() reaches the agent's first
decision, list_legal_actions returns a non-empty list, and play_action loop
terminates with finished=True before max_ticks.
"""

import pytest

from catan_env.env import CatanEnv, PlayActionParams


@pytest.mark.asyncio
async def test_scripted_first_action_terminates():
    env = CatanEnv(task_spec={"seed": 42, "agent_color": "RED", "max_ticks": 4000})
    env.setup()

    # First decision must be the agent's, with at least one legal action.
    listing = await env.list_legal_actions()
    assert listing.blocks, "list_legal_actions returned no blocks"
    assert env.player.last_playable, "agent has no legal actions at first decision"

    rewards: list[float] = []
    finished = False
    steps = 0
    while not finished:
        steps += 1
        out = await env.play_action(PlayActionParams(index=0))
        if out.reward is not None:
            rewards.append(out.reward)
        finished = out.finished
        assert steps < 2000, "agent took too many steps without terminating"

    assert finished, "loop exited without finished=True"
    assert env.game.winning_color() is not None or env._ticks >= env.max_ticks
    assert rewards, "no rewards collected"
    print(f"steps={steps} ticks={env._ticks} winner={env.game.winning_color()} sum_reward={sum(rewards):.3f}")


@pytest.mark.asyncio
async def test_invalid_index_returns_error():
    env = CatanEnv(task_spec={"seed": 1, "agent_color": "RED"})
    env.setup()
    n = len(env.player.last_playable)
    out = await env.play_action(PlayActionParams(index=n + 50))
    assert not out.finished
    assert out.reward is not None and out.reward < 0
