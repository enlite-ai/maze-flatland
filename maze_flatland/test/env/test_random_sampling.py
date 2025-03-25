"""
Tests random sampling of FlatlandEnvironment.
"""

from __future__ import annotations

from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.test.env_instantation import create_env_for_testing


def test_random_action_sampling() -> None:
    """
    Tests whether random action sampling can be run without problems.
    """

    env: FlatlandEnvironment = create_env_for_testing()
    env.reset()
    last_step = 0
    n_steps = 100

    for _ in range(n_steps):
        action = {'train_move': 0}
        while action['train_move'] == 0:
            action = env.action_space.sample()
        env.step(action)
        last_step += 1

    env.close()

    assert last_step == n_steps
