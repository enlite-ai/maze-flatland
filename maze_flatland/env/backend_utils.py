"""File holding backend utility methods."""

from __future__ import annotations

import numpy as np
from flatland.core.grid.grid4 import Grid4Transitions
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.decorators import enable_infrastructure_lru_cache
from maze.core.env.maze_state import MazeStateType


def create_all_transitions_list() -> list[int]:
    """Create a list of all possible rail env transition in the right order."""
    transitions_all = []
    for index, trans in enumerate(RailEnvTransitions.transition_list):
        transitions_all.append(trans)
        for _ in range(3):
            trans = RailEnvTransitions().rotate_transition(trans, rotation=90)
            transitions_all.append(trans)
    return transitions_all


TRANSITIONS_ALL = create_all_transitions_list()
TRANSITIONS_NAMES = [
    'Case 0 - empty cell                 ',
    'Case 1 - straight                    ',
    'Case 2 - simple switch               ',
    'Case 3 - diamond crossing            ',
    'Case 4 - single slip                 ',
    'Case 5 - double slip                 ',
    'Case 6 - symmetrical                 ',
    'Case 7 - dead end                    ',
    'Case 1b (8)  - simple turn right     ',
    'Case 1c (9)  - simple turn left      ',
    'Case 2b (10) - simple switch mirrored',
]


def identify_cell_type(cell_int: int) -> tuple[str, int, int]:
    """Identify the type of cell and the rotation of the cell type.

    :param cell_int: The cell int identified.
    :return: The name of the cell type, the id of the cell type and the id of the rotation.
    """

    idx = TRANSITIONS_ALL.index(cell_int)
    idx_base = idx // 4
    rotation = idx % 4

    return TRANSITIONS_NAMES[idx_base], idx_base, rotation


CELL_TYPE_SWITCH = [2, 3, 4, 5, 6, 10]


def get_cell_int_for_pos(x_pos: int, y_pos: int, transition_map: np.ndarray) -> int:
    """Get the cell int identifier for the given positon and the transition map.

    :param x_pos: The x position.
    :param y_pos: The y position.
    :param transition_map: The global transition map.

    :return: The cell int identifier.
    """
    binary = transition_map[x_pos, y_pos]
    binary_str = ''.join([str(x) for x in binary.astype(int)])
    cell_int = int(binary_str, 2)
    return cell_int


def get_next_position(x_pos: int, y_pos: int, direction: int) -> tuple[int, int]:
    """Get the next position of the given direction."""
    next_position = get_new_position((x_pos, y_pos), direction)
    return next_position


def get_global_map(maze_state: MazeStateType, train_id: int) -> np.ndarray:
    """Get the global transition map for the current train.

    :param maze_state: The current maze state.
    :param train_id: The id of the train.
    """
    assert GlobalObsForRailEnv in maze_state.observation_representations
    return maze_state.observation_representations[GlobalObsForRailEnv][train_id][0]


def get_cell_int_and_type(transition_map: np.ndarray, train_pos: tuple[int, int]) -> tuple[int, tuple[str, int, int]]:
    """Get the cell type for the position from a transition map and a position.
    :param transition_map: The global transition map for a train.
    :param train_pos: The position of the train to get the cell type for.
    :return: tuple with cell_int and the name of the cell type, the id of the cell type and the id of the rotation.
    """
    cell_int = get_cell_int_for_pos(*train_pos, transition_map)
    return cell_int, identify_cell_type(cell_int)


def get_possible_transitions(transition_map: np.ndarray, train_pos: tuple[int, int], train_dir: int) -> np.ndarray:
    """Get the possible transitions for the specified input.
    :param transition_map: The global transition map for a train.
    :param train_pos: The position of the train.
    :param train_dir: The direction of the train.
    :return: The possible transitions for the specified input.
    """
    cell_int = get_cell_int_and_type(transition_map, train_pos)[0]
    transitions = Grid4Transitions([]).get_transitions(cell_int, train_dir)
    return np.where(transitions)[0]


# Keep a version of the directions to position map.s
direction_to_pos_change_map = Grid4Transitions([]).gDir2dRC


def direction_to_pos_change(direction: int) -> tuple[int, int]:
    """Return the position change tuple for the given direction."""
    return direction_to_pos_change_map[direction]


# 150x150 grid should take ~ 350_000
@enable_infrastructure_lru_cache(maxsize=500_000)
def get_transitions_map(rail_env: RailEnv, use_cached: bool = True) -> np.ndarray:
    """Extracts the transition of each cell from a given rail environment.

    :param rail_env: the rail environment.
    :param use_cached: whether to use cached version or not.

    :return: a matrix of the grid where each cell is encoded.
            the returned object has a shape of [width, height, 16]
    """

    # when needed re-compute it.
    if not use_cached:
        get_transitions_map.cache_clear()
    rail_obs = np.zeros((rail_env.height, rail_env.width, 16))
    for i in range(rail_obs.shape[0]):
        for j in range(rail_obs.shape[1]):
            bitlist = [int(digit) for digit in bin(rail_env.rail.get_full_transitions(i, j))[2:]]
            bitlist = [0] * (16 - len(bitlist)) + bitlist
            rail_obs[i, j] = np.array(bitlist, dtype=bool)  # cast to bool as more efficient.
    return rail_obs


def agent_to_str(agent: EnvAgent) -> str:
    """Get a string of the given agent to print for debugging.

    :param agent: The agent to get the string for
    :return: A string representation of the agent.
    """
    txt = f'Handle: {agent.handle}'
    txt += (
        f'\n\tInitial Position: {agent.initial_position}, initial direction: {agent.initial_direction}, '
        f'target: {agent.target}'
    )
    txt += f'\n\tCurrent Position: {agent.position}, current direction: {agent.direction}'
    txt += '\n\tMalfunction'
    txt += f'\n\t\tin_malfunction              : {agent.malfunction_handler.in_malfunction}'
    txt += f'\n\t\tmalfunction_counter_complete: {agent.malfunction_handler.malfunction_counter_complete}'
    txt += f'\n\t\tmalfunction_down_counter    : {agent.malfunction_handler.malfunction_down_counter}'
    txt += f'\n\t\tnum_malfunctions            : {agent.malfunction_handler.num_malfunctions}'
    txt += (
        f'\n\tState Machine: {agent.state_machine.state.name}, '
        f'previous state: {agent.state_machine.previous_state.name}, '
        f'next state: {agent.state_machine.next_state.name}'
    )
    txt += '\n\tState Machine signals:'
    txt += f'\n\t\tin_malfunction              : {agent.state_machine.st_signals.in_malfunction}'
    txt += f'\n\t\tmalfunction_counter_complete: {agent.state_machine.st_signals.malfunction_counter_complete}'
    txt += f'\n\t\tearliest_departure_reached  : {agent.state_machine.st_signals.earliest_departure_reached}'
    txt += f'\n\t\tstop_action_given           : {agent.state_machine.st_signals.stop_action_given}'
    txt += f'\n\t\tvalid_movement_action_given : {agent.state_machine.st_signals.valid_movement_action_given}'
    txt += f'\n\t\tmovement_conflict           : {agent.state_machine.st_signals.movement_conflict}'
    txt += f'\n\t\ttarget_reached              : {agent.state_machine.st_signals.target_reached}'
    txt += f'\n\tActions Saved: {agent.action_saver}'
    return txt
