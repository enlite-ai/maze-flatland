"""
Tests positional.ObservationConversion for FlatlandEnvironment.
"""
from __future__ import annotations

import flatland.envs.line_generators
import flatland.envs.malfunction_generators
import flatland.envs.observations
import flatland.envs.rail_generators
import numpy as np
from maze.core.env.observation_conversion import ObservationType
from maze_flatland.agents.greedy_policy import GreedyPolicy
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.space_interfaces.action_conversion.directional import DirectionalAC
from maze_flatland.space_interfaces.observation_conversion.positional import PositionalObservationConversion
from maze_flatland.test.env_instantation import create_core_env, create_env_for_testing


def run_env() -> tuple[FlatlandEnvironment, list[ObservationType]]:
    """
    Runs small test environment until completion and returns collected observations.
    :return: Small test environment, list of collected observations.
    """

    core_env = create_core_env(2, 30, 30, 2, 0, {1: 1}, False, 4, 4)

    env = FlatlandEnvironment(
        core_env,
        action_conversion={'train_move': DirectionalAC()},
        observation_conversion={'train_move': PositionalObservationConversion(False)},
    )
    # Fix seed to avoid running into a deadlock, which would mean that no agent might be done at any point. This would
    # render the environment unsuitable for our purposes, since we want to check whether observations for done agents
    # are compatible with the specified observation space.
    env.seed(2)
    obs = env.reset()
    policy = GreedyPolicy()
    done = False
    max_n_steps = 65
    step = 0
    observations: list[ObservationType] = []

    while not done and step < max_n_steps:
        action = policy.compute_action(obs, maze_state=env.get_maze_state(), actor_id=env.actor_id(), env=None)
        obs, _, done, info = env.step(action)
        step += 1
        observations.append(obs)

    assert done is True
    assert step == 64

    env.close()

    return env, observations


def test_flatland_observation_in_space():
    """
    Tests whether all elements in observation are in observation space.
    Flatland's observation builders may provide special values (e.g. Nones) for done trains, so we run a test
    environment until completion to ensure our observations are valid in these cases too.
    """

    env, observations = run_env()

    # Check whether observation resulting from random action is in observation space.

    for i, obs in enumerate(observations):
        for key, value in obs.items():
            assert value in env.observation_conversion.space()[key], f'{key} - {value.shape}'
        assert obs in env.observation_space

    env.close()


def test_flatland_space_in_observation():
    """
    Tests whether all elements in observation space are in observation.
    Flatland's observation builders may provide special values (e.g. Nones) for done trains, so we run a test
    environment until completion to ensure our observations are valid in these cases too.
    """

    env: FlatlandEnvironment = create_env_for_testing()
    env.reset()
    obs = env.step(env.action_conversion.space().sample())[0]

    assert all(space_key in obs for space_key in env.observation_space.spaces.keys())


def test_flatland_observation_space_key_match():
    """
    Tests whether observation and observation space keys match.
    Flatland's observation builders may provide special values (e.g. Nones) for done trains, so we run a test
    environment until completion to ensure our observations are valid in these cases too.
    """

    env: FlatlandEnvironment = create_env_for_testing()
    env.reset()
    assert sorted(list(env.observation_conversion.space().spaces.keys())) == sorted(
        list(env.step(env.action_conversion.space().sample())[0].keys())
    )


def test_flatland_transition_map_match():
    """
    Tests whether transition maps are identical for each train (as long as they haven't reached their goal).
    """

    env: FlatlandEnvironment = create_env_for_testing()
    env.reset()
    # Make sure that transition maps are identical, assuming agents are not done yet.
    for _ in range(5):
        action = {'train_move': 0}
        while action['train_move'] == 0:
            action = env.action_conversion.space().sample()
        env.step(action)
        ob_glob_builder = env.observation_conversion.observation_builders[
            flatland.envs.observations.GlobalObsForRailEnv
        ]
        global_obs_repr = ob_glob_builder.get_many(range(env.get_maze_state().n_trains))
        assert all(
            np.asarray(global_obs_repr[i][0] == global_obs_repr[j][0]).all()
            for i in range(len(global_obs_repr))
            for j in range(i + 1, len(global_obs_repr))
        )
