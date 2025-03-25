"""
Tests graph.ObservationConversion for FlatlandEnvironment.
"""
from __future__ import annotations

import random

import numpy as np
from maze_flatland.env.masking.mask_builder import LogicMaskBuilder
from maze_flatland.space_interfaces.observation_conversion.graph_based_directional import (
    GraphDirectionalObservationConversion,
)
from maze_flatland.test.env_instantation import create_env_for_testing
from maze_flatland.wrappers.masking_wrapper import FlatlandMaskingWrapper


def test_flatland_observation_in_space():
    """
    Tests whether all elements in observation are in observation space.
    """

    env = create_env_for_testing(observation_conversion={'train_move': GraphDirectionalObservationConversion(True)})
    env.reset()
    # Check whether observation resulting from random action is in observation space.
    for step in range(5):
        action = {'train_move': 0}
        while action['train_move'] == 0:
            action = env.action_conversion.space().sample()
        obs = env.step(action)[0]
        assert obs in env.observation_space

    env.close()


def test_flatland_space_in_observation():
    """
    Tests whether all elements in observation space are in observation.
    """

    env = create_env_for_testing(observation_conversion={'train_move': GraphDirectionalObservationConversion(True)})
    env.reset()
    obs = env.step(env.action_conversion.space().sample())[0]

    assert all(space_key in obs for space_key in env.observation_space.spaces.keys())


def test_flatland_observation_space_key_match():
    """
    Tests whether observation and observation space keys match.
    """

    env = create_env_for_testing(observation_conversion={'train_move': GraphDirectionalObservationConversion(True)})
    env.reset()
    assert sorted(list(env.observation_conversion.space().spaces.keys())) == sorted(
        list(env.step(env.action_conversion.space().sample())[0].keys())
    )


def test_observation():
    """Test that the observation conversion works correctly."""
    env = create_env_for_testing(observation_conversion={'train_move': GraphDirectionalObservationConversion(True)})
    env = FlatlandMaskingWrapper.wrap(env, mask_builder=LogicMaskBuilder())
    random.seed(1234)
    env.seed(1234)
    s = env.reset()
    done = False
    while not done:
        possible_actions = np.where(s['train_move_mask'] == 1)[0]
        assert len(possible_actions) > 0
        a = random.choice(possible_actions)
        s, _, done, _ = env.step({'train_move': a})
        if done:
            continue
        dict_obs = env.observation_conversion.convert_to_dict(s)
        assert dict_obs['timesteps_left'] == env.get_maze_state().max_episode_steps - env.get_maze_state().env_time
        if sum(s['train_move_mask'][1:4]) > 1:  # train on a switch
            check_edges(dict_obs)


def check_edges(obs: dict):
    """Check that non-existing edges have the dummy obs.
    :param obs: the observation
    """
    all_dirs = ['L', 'F', 'R']
    invalid_distance_length = -1
    global_keys_to_ignore = [
        'timesteps_left',
        'in_transition',
        'train_status',
        'train_speed',
        'time_left_to_latest_arrival',
        'train_move_mask',
    ]
    direction_mask = np.asarray(obs['train_move_mask'][1:4])
    assert sum(direction_mask) <= 2
    for idx, _d in enumerate(all_dirs):
        if direction_mask[idx]:
            if obs[f'{_d}--other_train_dist'] > invalid_distance_length:
                assert obs[f'{_d}--num_agents_same_dir'] + obs[f'{_d}--num_agents_opposite_dir'] > 0
                assert obs[f'{_d}--slowest_speed'] > 0
            else:  # no train on the edge
                assert obs[f'{_d}--num_agents_same_dir'] + obs[f'{_d}--num_agents_opposite_dir'] == 0
                assert not obs[f'{_d}--deadlock']
                assert obs[f'{_d}--slowest_speed'] == 0
            assert obs[f'{_d}--distance'] > invalid_distance_length
            assert obs[f'{_d}--target_dist'] > invalid_distance_length
        else:
            for key in obs:  # not existing edges
                if key in global_keys_to_ignore:
                    continue
                prefix, feature_id = key.split('--')
                if prefix == _d:
                    assert obs[key] == -1


def test_consistent_order():
    """Tests that the keys ordering from the observation conversion never change."""
    env = create_env_for_testing(observation_conversion={'train_move': GraphDirectionalObservationConversion(True)})
    env = FlatlandMaskingWrapper.wrap(env, mask_builder=LogicMaskBuilder())
    random.seed(1234)
    env.seed(1234)
    s = env.reset()
    done = False
    obs_conv = env.observation_conversion

    def get_ordered_graph_oc_keys(obs_conv: GraphDirectionalObservationConversion, train_handle: int) -> list[str]:
        """helper method to extract the keys from the graph oc.
        :param obs_conv: the observation conversion object
        :param maze_state: the current maze state.
        :param train_handle: handle of the current train.
        :return: the list of keys in the dictionary.
        """

        graph_based_obs = obs_conv.current_obs
        return list(obs_conv.generate_observation(graph_based_obs, train_handle).keys())

    ordered_keys = get_ordered_graph_oc_keys(obs_conv, 0)

    while not done:
        possible_actions = np.where(s['train_move_mask'] == 1)[0]
        a = random.choice(possible_actions)
        s, _, done, _ = env.step({'train_move': a})
        if done:
            continue
        assert ordered_keys == get_ordered_graph_oc_keys(obs_conv, env.get_maze_state().current_train_id)
