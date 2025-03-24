"""File holding the tests for the delay_based_reward in flatland"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
from flatland.envs.step_utils.states import TrainState
from maze_flatland.reward.delay_based import DelayBasedReward

DUMMY_STATE = namedtuple(
    'FlatlandMazeState',
    ['env_time', 'n_trains', 'max_episode_steps', 'map_size', 'trains'],
)

DUMMY_TRAIN_STATE = namedtuple(
    'MazeTrainState',
    [
        'status',
        'in_transition',
        'deadlock',
        'target_distance',
        'arrival_delay',
        'speed',
        'time_left_to_scheduled_arrival',
        'latest_arrival',
    ],
)


def test_deadlock_transition_and_off_maps():
    """Test trains out of the map, in transition and deadlock."""
    reward_aggregator = DelayBasedReward(0.1)
    env_time = 1
    n_trains = 4
    trains_states = [TrainState.WAITING, TrainState.READY_TO_DEPART, TrainState.MOVING, TrainState.STOPPED]
    arrival_delays = [0, 0, 0, 0]
    target_distances = [10, 20, 5, 2]
    deadlocks = [0, 0, 0, 1]
    latest_arrivals = [21, 51, 11, 1]
    time_to_scheduled_arrival = np.asarray(latest_arrivals) - env_time
    in_transitions = [False, False, True, False]
    ms = DUMMY_STATE(
        env_time=env_time,
        n_trains=n_trains,
        max_episode_steps=100,
        map_size=(30, 30),
        trains=[
            DUMMY_TRAIN_STATE(
                trains_states[idx],
                in_transitions[idx],
                deadlocks[idx],
                target_distances[idx],
                arrival_delays[idx],
                1,
                time_to_scheduled_arrival[idx],
                latest_arrivals[idx],
            )
            for idx in range(n_trains)
        ],
    )
    rewards = reward_aggregator.summarize_reward(ms)
    assert len(rewards) == n_trains
    assert rewards[-1] == -1  # penalty for being in a deadlock
    assert sum(rewards[:-1]) == 0  # out of map and in_transitions


def test_trains_just_arrived():
    """Test for trains that have arrived with different delays."""
    reward_aggregator = DelayBasedReward(0.1)
    n_trains = 3
    trains_states = [TrainState.DONE for _ in range(n_trains)]
    # 1st arrived early, 2nd arrived sharp and 3rd arrived 10 steps later.
    arrival_delays = [-10, 0, 10]
    target_distances = [0 for _ in range(n_trains)]
    time_to_scheduled_arrival = [0 for _ in range(n_trains)]
    latest_arrivals = [10, 20, 20]
    in_transitions = [False, False, False]
    ms = DUMMY_STATE(
        env_time=1,
        n_trains=n_trains,
        max_episode_steps=100,
        map_size=(30, 30),
        trains=[
            DUMMY_TRAIN_STATE(
                trains_states[idx],
                in_transitions[idx],
                0,
                target_distances[idx],
                arrival_delays[idx],
                1,
                time_to_scheduled_arrival[idx],
                latest_arrivals[idx],
            )
            for idx in range(n_trains)
        ],
    )

    rewards = reward_aggregator.summarize_reward(ms)
    assert len(rewards) == n_trains
    assert sum(rewards[:-1]) == 2  # test reward for trains arrived with no delay
    assert rewards[-1] == 1 - arrival_delays[-1] / (1 + 100 - latest_arrivals[-1])  # check reward for the late train


def test_trains_already_arrived():
    """Test for case with a train that has already arrived."""
    reward_aggregator = DelayBasedReward(0.1)
    # override params of the reward aggregator
    reward_aggregator.already_arrived = [0]
    # pylint: disable=protected-access
    reward_aggregator._n_trains = 1

    n_trains = 1
    trains_states = [TrainState.DONE for _ in range(n_trains)]
    arrival_delays = [0]
    target_distances = [0 for _ in range(n_trains)]
    time_to_scheduled_arrival = [0 for _ in range(n_trains)]
    latest_arrivals = [10]
    in_transitions = [False]
    ms = DUMMY_STATE(
        env_time=10,
        n_trains=n_trains,
        max_episode_steps=100,
        map_size=(30, 30),
        trains=[
            DUMMY_TRAIN_STATE(
                trains_states[idx],
                in_transitions[idx],
                0,
                target_distances[idx],
                arrival_delays[idx],
                1,
                time_to_scheduled_arrival[idx],
                latest_arrivals[idx],
            )
            for idx in range(n_trains)
        ],
    )
    rewards = reward_aggregator.summarize_reward(ms)
    assert len(rewards) == n_trains
    assert rewards[0] == 0


def test_trains_on_maps():
    """Test for case with trains on the map that have not yet arrived."""

    reward_aggregator = DelayBasedReward(0.1)
    n_trains = 3
    env_time = 1
    trains_states = [TrainState.STOPPED, TrainState.MOVING, TrainState.MOVING]
    arrival_delays = [0 for _ in range(n_trains)]
    target_distances = [10, 90, 120]
    latest_arrivals = [80, 20, 20]
    time_to_scheduled_arrival = np.asarray(latest_arrivals) - env_time
    in_transitions = [False, False, False]
    ms = DUMMY_STATE(
        env_time=env_time,
        n_trains=n_trains,
        max_episode_steps=100,
        map_size=(30, 30),
        trains=[
            DUMMY_TRAIN_STATE(
                trains_states[idx],
                in_transitions[idx],
                0,
                target_distances[idx],
                arrival_delays[idx],
                1,
                time_to_scheduled_arrival[idx],
                latest_arrivals[idx],
            )
            for idx in range(n_trains)
        ],
    )
    rewards = reward_aggregator.summarize_reward(ms)
    assert len(rewards) == n_trains
    assert rewards[0] == 0  # train can arrive in time.
    assert -0.1 < rewards[1] < 0  # train can arrive but will be late.
    assert rewards[2] == -0.1  # train will never arrive.
