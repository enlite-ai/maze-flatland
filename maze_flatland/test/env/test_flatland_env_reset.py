"""
Tests reset of the FlatlandEnvironment.
"""

from __future__ import annotations

import random

from flatland.envs.step_utils.states import TrainState
from maze_flatland.env.core_env import ImpossibleEpisodeException
from maze_flatland.env.maze_action import FlatlandMazeAction
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.env.maze_state import FlatlandMazeState
from maze_flatland.test.env_instantation import create_core_env, create_env_for_testing
from maze_flatland.test.test_utils import check_if_equal


def test_flatland_env_reset():
    """
    Tests reset functionality of FlatlandEnvironment.
    """
    # todo this test can and should be improved. no checking last actions, not checkign the agent_id that is currently
    # selected to take an action...
    env: FlatlandEnvironment = create_env_for_testing()
    env.seed(1)
    env.reset()
    trains_ids = list(range(env.get_maze_state().n_trains))
    trains_have_departed = [False for _ in trains_ids]
    i = 0
    # Move trains.
    while i < 20:
        for handle in trains_ids:
            train = env.get_maze_state().trains[handle]
            train_state = train.status
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

    state: FlatlandMazeState = env.get_maze_state()

    # Trains should have left initial positions.
    for train in state.trains:
        assert train.position != train.initial_position
        assert train.is_done() or train.is_on_map()

    # Reset environment and fetch state.
    env.reset()
    state: FlatlandMazeState = env.get_maze_state()

    # Trains should be at initial positions.
    for train in state.trains:
        assert train.position == train.initial_position
        assert train.status == TrainState.WAITING
    env.close()


def test_flatland_impossible_episode():
    """Tests that a faulty episode is correctly identified at the reset time."""
    env = create_core_env(3, 37, 37, 2, 0, {1.0: 1})

    faulty_seed = 1954
    env.seed(faulty_seed)
    try:
        env.reset()
        assert False
    except ImpossibleEpisodeException:
        pass


def test_rail_env_reset():
    """Test that resetting the environment is equal to re-initialising it."""
    seeds = [1234, 9999]
    malf_rate = 1 / 10
    env = create_core_env(3, 37, 37, 2, malf_rate, {1.0: 1})
    for seed in seeds:
        env.seed(seed)
        s1 = env.reset()
        env_2 = create_core_env(3, 37, 37, 2, malf_rate, {1.0: 1})
        env_2.seed(seed)
        s2 = env_2.reset()
        assert check_if_equal(s1, s2)
        done = False
        while not done:
            action = random.choice(
                [FlatlandMazeAction.GO_FORWARD, FlatlandMazeAction.DEVIATE_LEFT, FlatlandMazeAction.DEVIATE_RIGHT]
            )
            s1, _, d1, _ = env.step(action)
            s2, _, d2, _ = env_2.step(action)
            assert d1 == d2
            done = d1
            assert check_if_equal(s1, s2)
