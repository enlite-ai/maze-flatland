"""
File holding instantiation of Flatland Environment.
"""
from __future__ import annotations

import flatland.core.env_prediction_builder
import flatland.envs.line_generators
import flatland.envs.malfunction_generators
import flatland.envs.observations
import flatland.envs.predictions
import flatland.envs.rail_generators
from flatland.utils.rendertools import AgentRenderVariant
from maze_flatland.env.core_env import FlatlandCoreEnvironment
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.env.renderer import FlatlandRendererBase
from maze_flatland.env.termination_condition import BaseEarlyTermination
from maze_flatland.reward.constant_reward import ConstantReward
from maze_flatland.reward.flatland_reward import FlatlandReward
from maze_flatland.space_interfaces.action_conversion.directional import DirectionalAC
from maze_flatland.space_interfaces.observation_conversion.base import BaseObservationConversion
from maze_flatland.space_interfaces.observation_conversion.positional import PositionalObservationConversion


def create_core_env(
    n_trains: int,
    map_width: int,
    map_height: int,
    n_cities: int,
    malfunction_rate: float,
    speed_ratio_map: dict[float, float],
    include_maze_state_in_serialization: bool = False,
    max_rails_between_cities: int = 3,
    max_rail_pairs_in_city: int = 3,
    reward_aggregator: FlatlandReward = ConstantReward(value=-1),
) -> FlatlandCoreEnvironment:
    """
    Generates CoreEnv by passing in attributes as classes.

    :param n_trains: Number of trains.
    :param map_width: Map width.
    :param map_height: Map height.
    :param n_cities: Number of cities.
    :param malfunction_rate: Malfunction rate.
    :param speed_ratio_map: Ratios per speed values for trains.
    :param include_maze_state_in_serialization: Whether to include maze state in serialization. Default: False
    :param max_rails_between_cities: Maximum number of rails between cities.
    :param max_rail_pairs_in_city: Maximum number of rails between pairs of cities.
    :param reward_aggregator: Reward aggregator to be used. Default: ConstantReward.
    :return: FlatlandCoreEnvironment instance.
    """

    return FlatlandCoreEnvironment(
        map_width=map_width,
        map_height=map_height,
        n_trains=n_trains,
        reward_aggregator=reward_aggregator,
        malfunction_generator=flatland.envs.malfunction_generators.ParamMalfunctionGen(
            flatland.envs.malfunction_generators.MalfunctionParameters(
                malfunction_rate=malfunction_rate, min_duration=1, max_duration=2
            )
        ),
        line_generator=flatland.envs.line_generators.sparse_line_generator(speed_ratio_map=speed_ratio_map),
        rail_generator=flatland.envs.rail_generators.SparseRailGen(
            max_num_cities=n_cities,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rail_pairs_in_city,
        ),
        include_maze_state_in_serialization=include_maze_state_in_serialization,
        termination_conditions=BaseEarlyTermination(),
        renderer=FlatlandRendererBase(1000, AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX, False, False),
    )


def create_env_for_testing(
    action_conversion: dict[str, DirectionalAC] | None = None,
    observation_conversion: dict[str, BaseObservationConversion] | None = None,
) -> FlatlandEnvironment:
    """Instantiate an env with the optionally given action and observation conversions.
        The returned environment consists of a 30x30 map with 4 trains.
        Malfunctions are disabled and trains have a 30% chance of having a fractional speed of 0.5.

    :param action_conversion: [Optional] Action conversion. Default: None
    :param observation_conversion: [Optional] Observation conversion. Default: None
    :return: FlatlandEnvironment instance.
    """
    if action_conversion is None:
        action_conversion = {'train_move': DirectionalAC()}
    if observation_conversion is None:
        observation_conversion = {'train_move': PositionalObservationConversion(False)}

    core_env: FlatlandCoreEnvironment = create_core_env(4, 30, 30, 3, 0, {1.0: 0.7, 1.0 / 2.0: 0.3})

    return FlatlandEnvironment(
        core_env,
        action_conversion=action_conversion,
        observation_conversion=observation_conversion,
    )
