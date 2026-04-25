"""Scripted-agent smoke test: always pick action index=0 (NORTH).

Verifies the environment wires together correctly: setup() initialises state,
list_legal_actions returns all 6 options, and play_action loop terminates with
finished=True at or before horizon steps.
"""

import pytest

from overcooked_env.env import OvercookedEnv, PlayActionParams


@pytest.mark.asyncio
async def test_scripted_always_north_terminates():
    env = OvercookedEnv(task_spec={"layout": "cramped_room", "horizon": 400})
    env.setup()

    assert env._env is not None
    assert env._env.state is not None

    listing = await env.list_legal_actions()
    assert listing.blocks, "list_legal_actions returned no blocks"
    assert "6 actions" in listing.blocks[0].text

    rewards: list[float] = []
    finished = False
    steps = 0
    while not finished:
        steps += 1
        out = await env.play_action(PlayActionParams(index=0))
        if out.reward is not None:
            rewards.append(out.reward)
        finished = out.finished
        assert steps <= 400, "agent exceeded horizon without terminating"

    assert finished, "loop exited without finished=True"
    assert env._done is True
    assert env._step_count == steps
    print(
        f"steps={steps}  soups_delivered={env._soups_delivered}  "
        f"sum_reward={sum(rewards):.4f}"
    )


@pytest.mark.asyncio
async def test_invalid_index_returns_error():
    env = OvercookedEnv(task_spec={"layout": "cramped_room", "horizon": 400})
    env.setup()
    out = await env.play_action(PlayActionParams(index=7))
    assert not out.finished
    assert out.reward is not None and out.reward < 0


@pytest.mark.asyncio
async def test_done_guard():
    """Calling play_action after episode ends returns finished=True gracefully."""
    env = OvercookedEnv(task_spec={"layout": "cramped_room", "horizon": 5})
    env.setup()
    for _ in range(5):
        await env.play_action(PlayActionParams(index=4))  # STAY
    assert env._done is True
    out = await env.play_action(PlayActionParams(index=0))
    assert out.finished
    assert out.reward == 0.0


@pytest.mark.asyncio
async def test_interact_action_is_valid():
    """Action index 5 (INTERACT) is accepted without error."""
    env = OvercookedEnv(task_spec={"layout": "cramped_room", "horizon": 400})
    env.setup()
    out = await env.play_action(PlayActionParams(index=5))
    assert out.reward is not None
    assert not out.finished or env._done
