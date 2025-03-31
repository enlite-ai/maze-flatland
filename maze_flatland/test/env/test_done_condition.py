"""File holding the tests to check the different condition on the termination of an episode."""

from __future__ import annotations

from maze_flatland.agents.greedy_policy import GreedyPolicy
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.test.env_instantation import create_env_for_testing


def test_successful():
    """
    Test that a successful episode is terminated with Flatland.Done.successful
    """
    env: FlatlandEnvironment = create_env_for_testing()
    policy = GreedyPolicy()
    env.seed(9999)
    obs = env.reset()
    step: int = 0
    max_n_steps = 305
    done: bool = False
    info = {}
    while not done and step < max_n_steps:
        action = policy.compute_action(obs, maze_state=env.get_maze_state(), actor_id=env.actor_id(), env=None)
        obs, rewards, done, info = env.step(action)
        step += 1
    assert step == 304

    assert 'Flatland.Done.successful' in info and info['Flatland.Done.successful']
    assert (False, True) == env.get_done_info(done, info)


def test_done_unsuccessful():
    """
    Test that a failing episode is terminated without triggering the successful.
    """

    env: FlatlandEnvironment = create_env_for_testing()
    env.seed(9999)
    obs = env.reset()
    step: int = 0
    done: bool = False
    info = {}
    while not done:
        action = {'train_move': 0}
        while action['train_move'] == 0:
            action = env.action_space.sample()
        obs, rewards, done, info = env.step(action)
        step += 1
    assert 'Flatland.Done.successful' not in info
    assert (True, False) == env.get_done_info(done, info)
