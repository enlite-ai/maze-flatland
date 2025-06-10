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
import numpy as np
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.malfunction_generators import NoMalfunctionGen
from flatland.envs.rail_generators import RailFromGridGen
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.timetable_utils import Line, Timetable
from flatland.utils.rendertools import AgentRenderVariant
from maze_flatland.env.core_env import FlatlandCoreEnvironment
from maze_flatland.env.custom_env_builder import (
    line_generator_from_line,
    place_element,
    timetable_generator_from_timetable,
)
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.env.renderer import FlatlandRendererBase
from maze_flatland.env.termination_condition import BaseEarlyTermination, NoEarlyTermination
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
        timetable_generator=None,
        include_maze_state_in_serialization=include_maze_state_in_serialization,
        termination_conditions=BaseEarlyTermination(),
        renderer=FlatlandRendererBase(1000, AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX, False, False),
    )


def create_env_for_testing(
    action_conversion: dict[str, DirectionalAC] | None = None,
    observation_conversion: dict[str, BaseObservationConversion] | None = None,
    speed_ratio_map=None,
) -> FlatlandEnvironment:
    """Instantiate an env with the optionally given action and observation conversions.
        The returned environment consists of a 30x30 map with 4 trains.
        Malfunctions are disabled and trains have a 30% chance of having a fractional speed of 0.5.

    :param action_conversion: [Optional] Action conversion. Default: None
    :param observation_conversion: [Optional] Observation conversion. Default: None
    :param speed_ratio_map: [Optional] Speed ratio map. Default: None
    :return: FlatlandEnvironment instance.
    """
    if speed_ratio_map is None:
        speed_ratio_map = {1.0: 0.7, 0.5: 0.3}
    if action_conversion is None:
        action_conversion = {'train_move': DirectionalAC()}
    if observation_conversion is None:
        observation_conversion = {'train_move': PositionalObservationConversion(False)}

    core_env: FlatlandCoreEnvironment = create_core_env(4, 30, 30, 3, 0, speed_ratio_map)

    return FlatlandEnvironment(
        core_env,
        action_conversion=action_conversion,
        observation_conversion=observation_conversion,
    )


def create_dummy_env_dead_end(n_dead_ends: int) -> FlatlandEnvironment:
    """Create a dummy environment with a simple switch going towards at a dead end.

    :param n_dead_ends: Number of dead ends.
    :return: A minimal 3x3 Flatland Environment instance.
    """
    if n_dead_ends == 0:
        grid_map = create_dummy_rail_no_dead_end()
    elif n_dead_ends == 1:
        grid_map = create_dummy_rail_single_dead_end()
    elif n_dead_ends == 2:
        grid_map = create_rail_double_dead_ends()
    else:
        assert False
    line = Line(
        agent_positions=[[(2, 1)]],
        agent_directions=[[Grid4TransitionsEnum.NORTH]],
        agent_targets=[(0, 1)],
        agent_speeds=[1],
    )
    timetable = Timetable(earliest_departures=[[1]], latest_arrivals=[[30]], max_episode_steps=999)
    height, width = grid_map.height, grid_map.width
    core_env = FlatlandCoreEnvironment(
        width,
        height,
        1,
        ConstantReward(0),
        NoMalfunctionGen(),
        line_generator_from_line(line),
        RailFromGridGen(grid_map, {'agents_hints': {'city_positions': [(1, 0)]}}),
        timetable_generator_from_timetable(timetable),
        NoEarlyTermination(),
        FlatlandRendererBase(1000, AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX, False, False),
        False,
    )
    return FlatlandEnvironment(
        core_env=core_env,
        observation_conversion={'train_move': PositionalObservationConversion(False)},
        action_conversion={'train_move': DirectionalAC()},
    )


def create_dummy_rail_no_dead_end() -> RailGridTransitionMap:
    """Create a dummy 3x3 rail map with no reversible dead-end.

    :return: The map for testing.
    """
    # create grid
    height = 3
    width = 3
    grid = np.array([[0] * width] * height)

    # place vertical straight line at 1,2
    grid, _, _, _ = place_element(grid, 1, 2, 'north', 'n')
    # place vertical DEAD_END line at 1,0 from north
    grid, _, _, _ = place_element(grid, 1, 0, 'north', 'n')

    # place horizontal DEAD_END line at 0, 1
    grid, _, _, _ = place_element(grid, 0, 1, 'west', 'w')

    # place single switch north-left
    grid, _, _, _ = place_element(grid, 1, 1, 'north', 'sssw')

    grid_map = RailGridTransitionMap(width, height, transitions=RailEnvTransitions())
    grid_map.grid = np.array(grid)
    return grid_map


def create_dummy_rail_single_dead_end() -> RailGridTransitionMap:
    """Create a dummy 3x3 rail map.

    :return: The map for testing.
    """
    # create grid
    height = 3
    width = 3
    grid = np.array([[0] * width] * height)

    # place vertical straight line at 1,2
    grid, _, _, _ = place_element(grid, 1, 2, 'north', 'n')
    # place vertical DEAD_END line at 1,0 from north
    grid, _, _, _ = place_element(grid, 1, 0, 'north', 'n')

    # place horizontal DEAD_END line at 0, 1
    grid, _, _, _ = place_element(grid, 0, 1, 'west', 'de')

    # place single switch north-left
    grid, _, _, _ = place_element(grid, 1, 1, 'north', 'sssw')

    grid_map = RailGridTransitionMap(width, height, transitions=RailEnvTransitions())
    grid_map.grid = np.array(grid)
    return grid_map


def create_rail_double_dead_ends() -> RailGridTransitionMap:
    """Create a simple rail_map on a 4x3 grid map with 2 dead ends.
        The agent could reach its target by 'bouncing' against the 2 dead-ends.
        The topology is as follows:
             top-center: target
             center-center: switch left, top (coming from bottom);
             bottom-center: dead-end coming from north;
             left-center: dead-end coming from east;
             An additional connection between bottom-center and center-center to connect these 2 cells.

    :return: RailGridTransitionMap for the rail map..
    """
    # origin is in the left-bottom corner for the array.
    # the image swap the y-axis by bringing the origin at top-left.
    grid = np.array([[0] * 3] * 4)
    # 0 2 dead end from east
    grid, _, _, _ = place_element(grid, 0, 2, 'west', 'de')
    # 1 0 dead end from north
    grid, _, _, _ = place_element(grid, 1, 0, 'south', 'de')
    # 1 1 straight
    grid, _, _, _ = place_element(grid, 1, 1, 'north', 'n')
    # 1 2 switch
    grid, _, _, _ = place_element(grid, 1, 2, 'north', 'sssw')
    # 1 3 straight
    grid, _, _, _ = place_element(grid, 1, 3, 'north', 'n')
    grid_map = RailGridTransitionMap(3, 4, transitions=RailEnvTransitions())
    grid_map.grid = np.array(grid)
    return grid_map
