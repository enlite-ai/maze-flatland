"""File holdings tests checking that there are no differences in the backend based
on the last checked version (v 4.0.1)"""
from __future__ import annotations

import os
import pickle

from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.space_interfaces.action_conversion.directional import DirectionalAC
from maze_flatland.space_interfaces.observation_conversion.simple import SimpleObservationConversion
from maze_flatland.test.env_instantation import create_core_env
from maze_flatland.test.test_utils import _compare


def test_rail():
    """Initialises an environment and compares its data against a serialised file."""
    env = create_core_env(3, 35, 35, 3, 1 / 10, {1: 1})
    env = FlatlandEnvironment(
        env,
        action_conversion={'train_move': DirectionalAC()},
        observation_conversion={'train_move': SimpleObservationConversion(False)},
    )
    env.seed(1234)
    _ = env.reset()
    re = env.rail_env

    agents_info = [
        {
            'earliest_departure': agent.earliest_departure,
            'latest_arrival': agent.latest_arrival,
            'direction': agent.direction,
            'initial_position': agent.position,
            'target': agent.target,
        }
        for agent in re.agents
    ]

    malfunction_times = {tid: [] for tid in range(env.n_trains)}
    done = False
    action = {'train_move': 2}
    while not done:
        if env.is_flat_step():
            for train in env.get_maze_state().trains:
                if train.malfunction_time_left > 0:
                    malfunction_times[train.handle].append(train.env_time)
        # get the malfunctions and records the step.
        _, _, done, info = env.step(action)
    env_info = {
        'distance_maps': re.distance_map.distance_map,
        'agents_data': agents_info,
        'malfunction_times': malfunction_times,
    }

    pickled_datapath = os.path.split(os.path.dirname(__file__))[0] + '/serialised_data/env_info_v_4.0.1.pkl'
    with open(pickled_datapath, 'rb') as f:
        expected_agents_info = pickle.load(f)

    for k, v in env_info.items():
        assert k in expected_agents_info
        assert _compare(v, expected_agents_info[k]), f'Expected key {k} not matching.'
