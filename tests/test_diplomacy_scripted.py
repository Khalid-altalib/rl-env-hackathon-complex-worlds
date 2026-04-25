"""Scripted-agent smoke test: always pick action index=0 for every order.

Verifies the env loop terminates within the year cap and produces rewards.
"""

import pytest

from diplomacy_env.env import DiplomacyEnv, PlayActionParams


@pytest.mark.asyncio
async def test_scripted_first_action_terminates():
    env = DiplomacyEnv(task_spec={"seed": 42, "agent_power": "FRANCE", "max_year": 1905})
    env.setup()

    assert env.game is not None
    assert env._orderable_queue, "no orderable locations at game start"

    rewards: list[float] = []
    finished = False
    steps = 0
    while not finished:
        steps += 1
        out = await env.play_action(PlayActionParams(index=0))
        if out.reward is not None:
            rewards.append(out.reward)
        finished = out.finished
        assert steps < 5000, "env failed to terminate within step cap"

    assert finished
    assert env._phases_processed > 0
    print(
        f"steps={steps}  phases_processed={env._phases_processed}  "
        f"agent_sc_final={len(env.game.get_centers(env.agent_power))}  "
        f"sum_reward={sum(rewards):+.4f}"
    )


@pytest.mark.asyncio
async def test_invalid_index_returns_negative_reward():
    env = DiplomacyEnv(task_spec={"seed": 1, "agent_power": "FRANCE", "max_year": 1905})
    env.setup()
    out = await env.play_action(PlayActionParams(index=9999))
    assert not out.finished
    assert out.reward is not None and out.reward < 0
    # Queue should be unchanged
    assert env._orderable_queue, "invalid index should not consume the orderable location"


@pytest.mark.asyncio
async def test_no_reward_until_phase_processes():
    """Reward is None for intermediate orders within a phase, only set when phase processes."""
    env = DiplomacyEnv(task_spec={"seed": 7, "agent_power": "FRANCE", "max_year": 1905})
    env.setup()
    # FRANCE starts with 3 orderable locations; first 2 picks should not return a reward.
    initial_queue_len = len(env._orderable_queue)
    assert initial_queue_len >= 2
    for _ in range(initial_queue_len - 1):
        out = await env.play_action(PlayActionParams(index=0))
        assert out.reward is None, "reward should be None until last order of phase"
        assert not out.finished
    out = await env.play_action(PlayActionParams(index=0))
    assert out.reward is not None, "reward should be set after final order of phase"
    assert env._phases_processed >= 1


@pytest.mark.asyncio
async def test_done_guard_after_termination():
    env = DiplomacyEnv(task_spec={"seed": 3, "agent_power": "FRANCE", "max_year": 1902})
    env.setup()
    finished = False
    while not finished:
        out = await env.play_action(PlayActionParams(index=0))
        finished = out.finished
    out = await env.play_action(PlayActionParams(index=0))
    assert out.finished
    assert out.reward == 0.0
