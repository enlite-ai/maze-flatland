"""
Tests for instantiation of Flatland environment.
"""

from __future__ import annotations

import os

import maze_flatland.conf as conf_module
from maze.core.env.maze_env import MazeEnv
from maze.core.utils.config_utils import make_env_from_hydra
from maze.core.utils.factory import Factory
from maze_flatland.env.core_env import FlatlandCoreEnvironment
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.env.termination_condition import BaseEarlyTermination
from maze_flatland.space_interfaces.observation_conversion.simple import SimpleObservationConversion
from maze_flatland.test.env_instantation import create_env_for_testing
from omegaconf import OmegaConf


def env_from_dict_configs_example() -> FlatlandEnvironment:
    """Instantiate an env using dict configs.
        The returned environment consists of a 30x30 map with 4 trains.
        Malfunctions are enabled and trains have a 30% chance of having a fractional speed of 0.5.

    :return: Flatland Environment instance.
    """

    n_trains = 4
    speed_ratio_map = {1.0: 0.7, 1.0 / 2.0: 0.3}

    core_env: FlatlandCoreEnvironment = FlatlandCoreEnvironment(
        map_width=30,
        map_height=30,
        n_trains=n_trains,
        reward_aggregator={
            '_target_': 'maze_flatland.reward.default_flatland_v2.RewardAggregator',
            'alpha': 1,
            'beta': 1,
            'reward_for_goal_reached': 10,
            'penalty_for_start': 0,
            'penalty_for_stop': 0,
            'use_train_speed': True,
            'penalty_for_block': 5,
            'penalty_for_deadlock': 500,
            'distance_penalty_weight': 1 / 100,
        },
        malfunction_generator={
            '_target_': 'flatland.envs.malfunction_generators.ParamMalfunctionGen',
            '_recursive_': True,
            'parameters': {
                '_target_': 'flatland.envs.malfunction_generators.MalfunctionParameters',
                'malfunction_rate': 0.1,
                'min_duration': 1,
                'max_duration': 2,
            },
        },
        line_generator={'_target_': 'flatland.envs.line_generators.SparseLineGen', 'speed_ratio_map': speed_ratio_map},
        rail_generator={
            '_target_': 'flatland.envs.rail_generators.SparseRailGen',
            'max_num_cities': 3,
            'grid_mode': False,
            'max_rails_between_cities': 3,
            'max_rail_pairs_in_city': 3,
        },
        termination_conditions=BaseEarlyTermination(),
        renderer={
            '_target_': 'maze_flatland.env.renderer.FlatlandRendererBase',
            'img_width': 1500,
            'agent_render_variant': 'flatland.utils.rendertools.AgentRenderVariant.AgentRenderVariant.BOX_ONLY',
            'render_out_of_map_trains': True,
            'highlight_current_train': False,
        },
    )
    return FlatlandEnvironment(
        core_env,
        action_conversion={
            'train_move': {
                '_target_': 'maze_flatland.space_interfaces.action_conversion.directional.DirectionalAC',
            }
        },
        observation_conversion={
            'train_move': {
                '_target_': 'maze_flatland.space_interfaces.observation_conversion.positional'
                + '.PositionalObservationConversion',
                'serialize_representation': True,
            }
        },
    )


def env_from_config_example() -> FlatlandEnvironment:
    """Instantiate a Flatland environment using a config file.
        The returned environment matches the one detailed at conf/env/flatland.yaml

    :return: Flatland Environment instance.
    """

    module_path = list(conf_module.__path__)[0]
    default_config_path = os.path.join(module_path, 'env/flatland.yaml')

    # Load the default config.
    # Use OmegaConf (Hydra dependency / predecessor) to support interpolations (when using plain YAML files,
    # config can be loaded through `yaml.safe_load(...)` directly)
    env_config = OmegaConf.load(default_config_path)['env']
    # Instantiate env.
    return Factory(base_type=FlatlandEnvironment).instantiate(env_config)


def env_from_hydra_example() -> MazeEnv:
    """Instantiating an environment through hydra.

    The returned environment matches the one detailed at conf/env/flatland.yaml
    :return: FlatlandEnvironment instance."""
    env = make_env_from_hydra('maze_flatland.conf', 'conf_rollout', env='flatland')
    return env


def test_flatland_env_registration():
    env = env_from_dict_configs_example()
    assert isinstance(env, FlatlandEnvironment)

    env = env_from_config_example()
    assert isinstance(env, FlatlandEnvironment)

    env = env_from_hydra_example()
    assert isinstance(env, FlatlandEnvironment)

    env = create_env_for_testing()
    assert isinstance(env, FlatlandEnvironment)

    env = create_env_for_testing(observation_conversion={'train_move': SimpleObservationConversion(False)})
    assert isinstance(env, FlatlandEnvironment)
