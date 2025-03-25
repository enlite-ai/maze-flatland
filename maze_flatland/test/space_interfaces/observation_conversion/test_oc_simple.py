"""
Tests base.ObservationConversion for FlatlandEnvironment.
"""
from __future__ import annotations

from maze.core.utils.config_utils import make_env_from_hydra


def test_flatland_observation_in_space():
    """
    Tests whether all elements in observation are in observation space.
    """

    env = make_env_from_hydra(
        'maze_flatland.conf', 'conf_rollout', env='flatland', env_configuration='flatland-baseobs'
    )
    env.reset()
    # Check whether observation resulting from random action is in observation space.
    for step in range(5):
        assert env.step(env.action_conversion.space().sample())[0] in env.observation_space

    env.close()


def test_flatland_space_in_observation():
    """
    Tests whether all elements in observation space are in observation.
    """

    env = make_env_from_hydra(
        'maze_flatland.conf', 'conf_rollout', env='flatland', env_configuration='flatland-baseobs'
    )
    env.reset()
    obs = env.step(env.action_conversion.space().sample())[0]

    assert all(space_key in obs for space_key in env.observation_space.spaces.keys())


def test_flatland_observation_space_key_match():
    """
    Tests whether observation and observation space keys match.
    """

    env = make_env_from_hydra(
        'maze_flatland.conf', 'conf_rollout', env='flatland', env_configuration='flatland-baseobs'
    )
    env.reset()
    assert sorted(list(env.observation_conversion.space().spaces.keys())) == sorted(
        list(env.step(env.action_conversion.space().sample())[0].keys())
    )
