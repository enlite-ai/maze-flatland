"""File holding the tests for the early termination conditions."""
from __future__ import annotations

from maze_flatland.env.termination_condition import (
    DeadlockEarlyTermination,
    IncludeOutOfTimeTrainsInEarlyTermination,
    NoDelayEarlyTermination,
    OutOfTimeEarlyTermination,
)
from maze_flatland.test.env_instantation import create_env_for_testing


def test_no_delay():
    """Check the early termination cause of the strict no delay."""
    env = create_env_for_testing()
    # overwrite the termination conditions
    env.core_env.termination_conditions.append(NoDelayEarlyTermination())
    env.seed(1234)
    done = False
    _ = env.reset()
    while not done:
        _, _, done, _ = env.step({'train_move': 4})
    ms = env.get_maze_state()
    assert env.get_env_time() == 44
    assert ms.trains[-1].target_distance > ms.trains[-1].time_left_to_scheduled_arrival


def test_out_of_time():
    """Check the early termination cause of a train not being able to reach its target."""
    env = create_env_for_testing()
    # overwrite the termination conditions
    env.core_env.termination_conditions.append(OutOfTimeEarlyTermination())
    env.seed(1234)
    done = False
    _ = env.reset()
    while not done:
        _, _, done, _ = env.step({'train_move': 4})
    ms = env.get_maze_state()
    assert env.get_env_time() == 71
    assert ms.trains[-2].target_distance / ms.trains[-2].speed > ms.max_episode_steps - ms.env_time


def test_no_early_termination():
    """Check that early termination is not triggered."""
    env = create_env_for_testing()
    # overwrite the termination conditions
    env.seed(1234)
    done = False
    _ = env.reset()
    while not done:
        _, _, done, _ = env.step({'train_move': 4})
    ms = env.get_maze_state()
    assert env.get_env_time() == ms.max_episode_steps


def test_all_done_or_out_of_time():
    """Check that either are all done or all trains are too late to make it in time."""
    env = create_env_for_testing()
    # overwrite the termination conditions
    env.core_env.termination_conditions = [IncludeOutOfTimeTrainsInEarlyTermination()]
    env.seed(1234)
    done = False
    _ = env.reset()
    while not done:
        _, _, done, _ = env.step({'train_move': 4})
    ms = env.get_maze_state()
    assert env.get_env_time() == 141
    assert all(train.out_of_time for train in ms.trains)


def test_early_termination_deadlock():
    """Check that deadlock triggers the early termination."""
    env = create_env_for_testing()
    # overwrite the termination conditions
    env.core_env.termination_conditions.append(DeadlockEarlyTermination())
    env.seed(1234)
    done = False
    _ = env.reset()
    while not done:
        _, _, done, _ = env.step({'train_move': 2})
    ms = env.get_maze_state()
    assert env.get_env_time() == 44
    assert sum(train.deadlock for train in ms.trains) == 2
