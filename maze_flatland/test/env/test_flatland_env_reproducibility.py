"""
Tests checking reproducibility of FlatlandEnvironment.
"""

from __future__ import annotations

import random

from flatland.envs.step_utils.states import TrainState
from maze.test.shared_test_utils.reproducibility import conduct_env_reproducibility_test
from maze_flatland.agents.greedy_policy import GreedyPolicy
from maze_flatland.env.maze_action import FlatlandMazeAction
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.test.env_instantation import create_env_for_testing
from maze_flatland.test.test_utils import check_if_equal


def test_env_reproducibility_with_greedy_policy():
    """
    Checks for equality of hash-keys with resulting state hash after utilization of InverseKinematicsPolicy.
    """

    env: FlatlandEnvironment = create_env_for_testing()
    policy: GreedyPolicy = GreedyPolicy()

    assert conduct_env_reproducibility_test(
        env,
        lambda observation, action_space: policy.compute_action(
            observation, deterministic=True, maze_state=env.get_maze_state(), actor_id=env.actor_id(), env=None
        ),
        n_steps=20,
    ), 'Hash-keys do not match.'
    env.close()


def _init_env(seed):
    """Initialise and returns an instance of the environment."""
    env = create_env_for_testing()
    env.seed(seed)
    env.reset()
    return env


def test_parallel_envs_equivalence():
    """
    Tests equivalence of two parallel environment equally seeded.
    """

    # selected to take an action...
    envs = [_init_env(9999) for _ in range(2)]
    visited_states = {0: [], 1: []}

    for idx, env in enumerate(envs):
        env.seed(1)
        env.reset()
        trains_ids = list(range(env.get_maze_state().n_trains))
        trains_have_departed = [False for _ in trains_ids]
        i = 0
        # Move trains.
        while i < 20:
            for handle in trains_ids:
                train_state = env.get_maze_state().trains[handle].status
                if train_state == TrainState.READY_TO_DEPART:
                    env.step(env.action_conversion.maze_to_space(FlatlandMazeAction.GO_FORWARD))
                    trains_have_departed[handle] = True
                elif train_state in [TrainState.MOVING, TrainState.STOPPED]:
                    # step random action between forward, stop and turn
                    env.step(
                        env.action_conversion.maze_to_space(
                            random.choice(
                                [
                                    FlatlandMazeAction.GO_FORWARD,
                                    FlatlandMazeAction.STOP_MOVING,
                                    FlatlandMazeAction.DEVIATE_LEFT,
                                    FlatlandMazeAction.DEVIATE_RIGHT,
                                ]
                            )
                        )
                    )
                else:
                    env.step(env.action_conversion.maze_to_space(FlatlandMazeAction.DO_NOTHING))
            if all(trains_have_departed):
                i += 1
            visited_states[idx].append(env.get_maze_state())

        assert all((check_if_equal(vs0, vs1) for vs0, vs1 in zip(visited_states[0], visited_states[1])))


def test_different_seeds_different_states():
    """
    Tests that two environment differently seeded are different.
    """

    # selected to take an action...
    envs = [_init_env(_) for _ in [1234, 9999]]
    visited_states = {0: [], 1: []}

    for idx, env in enumerate(envs):
        trains_ids = list(range(env.get_maze_state().n_trains))
        trains_have_departed = [False for _ in trains_ids]
        i = 0
        # Move trains.
        while i < 20:
            for handle in trains_ids:
                train_state = env.get_maze_state().trains[handle].status
                if train_state == TrainState.READY_TO_DEPART:
                    env.step(env.action_conversion.maze_to_space(FlatlandMazeAction.GO_FORWARD))
                    trains_have_departed[handle] = True
                elif train_state in [TrainState.MOVING, TrainState.STOPPED]:
                    # step random action between forward, stop and turn
                    env.step(
                        env.action_conversion.maze_to_space(
                            random.choice(
                                [
                                    FlatlandMazeAction.GO_FORWARD,
                                    FlatlandMazeAction.STOP_MOVING,
                                    FlatlandMazeAction.DEVIATE_LEFT,
                                    FlatlandMazeAction.DEVIATE_RIGHT,
                                ]
                            )
                        )
                    )

                else:
                    env.step(env.action_conversion.maze_to_space(FlatlandMazeAction.DO_NOTHING))
            if all(trains_have_departed):
                i += 1
            visited_states[idx].append(env.get_maze_state())

    assert not all((check_if_equal(vs0, vs1) for vs0, vs1 in zip(visited_states[0], visited_states[1])))
