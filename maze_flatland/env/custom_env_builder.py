"""File holding utils to create customized environment from scratch."""
from __future__ import annotations

from typing import Callable

import numpy as np
from flatland.core.grid.rail_env_grid import RailEnvTransitionsEnum
from flatland.envs.timetable_utils import Line, Timetable

elements = {
    'north': {
        'empty': {'element': RailEnvTransitionsEnum.empty, 'orientation': 'north'},
        'n': {'element': RailEnvTransitionsEnum.vertical_straight, 'orientation': 'north'},
        'w': {'element': RailEnvTransitionsEnum.right_turn_from_west, 'orientation': 'west'},
        'e': {'element': RailEnvTransitionsEnum.right_turn_from_south, 'orientation': 'east'},
        'ssnw': {'element': RailEnvTransitionsEnum.simple_switch_south_right, 'orientation': 'north'},
        'ssws': {'element': RailEnvTransitionsEnum.simple_switch_east_right, 'orientation': 'west'},
        'ssse': {'element': RailEnvTransitionsEnum.simple_switch_north_right, 'orientation': 'north'},
        'ssse_e': {'element': RailEnvTransitionsEnum.simple_switch_north_right, 'orientation': 'east'},
        'sses': {'element': RailEnvTransitionsEnum.simple_switch_west_left, 'orientation': 'east'},
        'ssne': {'element': RailEnvTransitionsEnum.simple_switch_south_left, 'orientation': 'north'},
        'sssw': {'element': RailEnvTransitionsEnum.simple_switch_north_left, 'orientation': 'north'},
        'sssw_w': {'element': RailEnvTransitionsEnum.simple_switch_north_left, 'orientation': 'west'},
        'c': {'element': RailEnvTransitionsEnum.diamond_crossing, 'orientation': 'north'},
        'sslnw': {'element': RailEnvTransitionsEnum.single_slip_NW, 'orientation': 'north'},
        'sslsw': {'element': RailEnvTransitionsEnum.single_slip_SW, 'orientation': 'north'},
        'sslsw_w': {'element': RailEnvTransitionsEnum.single_slip_SW, 'orientation': 'west'},
        'ssles': {'element': RailEnvTransitionsEnum.single_slip_SE, 'orientation': 'north'},
        'ssles_e': {'element': RailEnvTransitionsEnum.single_slip_SE, 'orientation': 'east'},
        'sslne': {'element': RailEnvTransitionsEnum.single_slip_NE, 'orientation': 'north'},
        'dslne': {'element': RailEnvTransitionsEnum.double_slip_NE_SW, 'orientation': 'north'},
        'dslne_w': {'element': RailEnvTransitionsEnum.double_slip_NE_SW, 'orientation': 'west'},
        'dslnw': {'element': RailEnvTransitionsEnum.double_slip_NW_SE, 'orientation': 'north'},
        'dslnw_e': {'element': RailEnvTransitionsEnum.double_slip_NW_SE, 'orientation': 'east'},
        'sysw': {'element': RailEnvTransitionsEnum.symmetric_switch_from_west, 'orientation': 'west'},
        'syss_e': {'element': RailEnvTransitionsEnum.symmetric_switch_from_south, 'orientation': 'east'},
        'syss_w': {'element': RailEnvTransitionsEnum.symmetric_switch_from_south, 'orientation': 'west'},
        'syse': {'element': RailEnvTransitionsEnum.symmetric_switch_from_east, 'orientation': 'east'},
        'de': {'element': RailEnvTransitionsEnum.dead_end_from_south, 'orientation': 'north'},
    },
    'east': {
        'empty': {'element': RailEnvTransitionsEnum.empty, 'orientation': 'east'},
        'e': {'element': RailEnvTransitionsEnum.horizontal_straight, 'orientation': 'east'},
        'n': {'element': RailEnvTransitionsEnum.right_turn_from_north, 'orientation': 'north'},
        's': {'element': RailEnvTransitionsEnum.right_turn_from_west, 'orientation': 'south'},
        'ssnw': {'element': RailEnvTransitionsEnum.simple_switch_south_right, 'orientation': 'north'},
        'ssws': {'element': RailEnvTransitionsEnum.simple_switch_east_right, 'orientation': 'east'},
        'ssws_s': {'element': RailEnvTransitionsEnum.simple_switch_east_right, 'orientation': 'south'},
        'ssen': {'element': RailEnvTransitionsEnum.simple_switch_west_right, 'orientation': 'east'},
        'sswn': {'element': RailEnvTransitionsEnum.simple_switch_east_left, 'orientation': 'east'},
        'sswn_n': {'element': RailEnvTransitionsEnum.simple_switch_east_left, 'orientation': 'north'},
        'sssw': {'element': RailEnvTransitionsEnum.simple_switch_north_left, 'orientation': 'south'},
        'sses': {'element': RailEnvTransitionsEnum.simple_switch_west_left, 'orientation': 'east'},
        'c': {'element': RailEnvTransitionsEnum.diamond_crossing, 'orientation': 'east'},
        'sslnw': {'element': RailEnvTransitionsEnum.single_slip_NW, 'orientation': 'east'},
        'sslnw_n': {'element': RailEnvTransitionsEnum.single_slip_NW, 'orientation': 'north'},
        'sslsw': {'element': RailEnvTransitionsEnum.single_slip_SW, 'orientation': 'east'},
        'sslsw_s': {'element': RailEnvTransitionsEnum.single_slip_SW, 'orientation': 'south'},
        'ssles': {'element': RailEnvTransitionsEnum.single_slip_SE, 'orientation': 'east'},
        'sslne': {'element': RailEnvTransitionsEnum.single_slip_NE, 'orientation': 'east'},
        'dslne': {'element': RailEnvTransitionsEnum.double_slip_NE_SW, 'orientation': 'east'},
        'dslne_s': {'element': RailEnvTransitionsEnum.double_slip_NE_SW, 'orientation': 'south'},
        'dslnw': {'element': RailEnvTransitionsEnum.double_slip_NW_SE, 'orientation': 'east'},
        'dslnw_n': {'element': RailEnvTransitionsEnum.double_slip_NW_SE, 'orientation': 'north'},
        'sysn': {'element': RailEnvTransitionsEnum.symmetric_switch_from_north, 'orientation': 'north'},
        'sysw_n': {'element': RailEnvTransitionsEnum.symmetric_switch_from_west, 'orientation': 'north'},
        'sysw_s': {'element': RailEnvTransitionsEnum.symmetric_switch_from_west, 'orientation': 'south'},
        'syss': {'element': RailEnvTransitionsEnum.symmetric_switch_from_south, 'orientation': 'south'},
        'de': {'element': RailEnvTransitionsEnum.dead_end_from_west, 'orientation': 'east'},
    },
    'south': {
        'empty': {'element': RailEnvTransitionsEnum.empty, 'orientation': 'south'},
        's': {'element': RailEnvTransitionsEnum.vertical_straight, 'orientation': 'south'},
        'w': {'element': RailEnvTransitionsEnum.right_turn_from_north, 'orientation': 'west'},
        'e': {'element': RailEnvTransitionsEnum.right_turn_from_east, 'orientation': 'east'},
        'ssnw': {'element': RailEnvTransitionsEnum.simple_switch_south_right, 'orientation': 'south'},
        'ssnw_w': {'element': RailEnvTransitionsEnum.simple_switch_south_right, 'orientation': 'west'},
        'ssse': {'element': RailEnvTransitionsEnum.simple_switch_north_right, 'orientation': 'south'},
        'ssen': {'element': RailEnvTransitionsEnum.simple_switch_west_right, 'orientation': 'east'},
        'ssne': {'element': RailEnvTransitionsEnum.simple_switch_south_left, 'orientation': 'south'},
        'ssne_e': {'element': RailEnvTransitionsEnum.simple_switch_south_left, 'orientation': 'east'},
        'sswn': {'element': RailEnvTransitionsEnum.simple_switch_east_left, 'orientation': 'west'},
        'sssw': {'element': RailEnvTransitionsEnum.simple_switch_north_left, 'orientation': 'south'},
        'c': {'element': RailEnvTransitionsEnum.diamond_crossing, 'orientation': 'south'},
        'sslnw': {'element': RailEnvTransitionsEnum.single_slip_NW, 'orientation': 'south'},
        'sslnw_w': {'element': RailEnvTransitionsEnum.single_slip_NW, 'orientation': 'west'},
        'sslsw': {'element': RailEnvTransitionsEnum.single_slip_SW, 'orientation': 'south'},
        'ssles': {'element': RailEnvTransitionsEnum.single_slip_SE, 'orientation': 'south'},
        'sslne': {'element': RailEnvTransitionsEnum.single_slip_NE, 'orientation': 'south'},
        'sslne_e': {'element': RailEnvTransitionsEnum.single_slip_NE, 'orientation': 'east'},
        'dslne': {'element': RailEnvTransitionsEnum.double_slip_NE_SW, 'orientation': 'south'},
        'dslne_e': {'element': RailEnvTransitionsEnum.double_slip_NE_SW, 'orientation': 'east'},
        'dslnw': {'element': RailEnvTransitionsEnum.double_slip_NW_SE, 'orientation': 'south'},
        'dslnw_w': {'element': RailEnvTransitionsEnum.double_slip_NW_SE, 'orientation': 'west'},
        'sysw': {'element': RailEnvTransitionsEnum.symmetric_switch_from_west, 'orientation': 'west'},
        'sysn_e': {'element': RailEnvTransitionsEnum.symmetric_switch_from_north, 'orientation': 'east'},
        'sysn_w': {'element': RailEnvTransitionsEnum.symmetric_switch_from_north, 'orientation': 'west'},
        'syse': {'element': RailEnvTransitionsEnum.symmetric_switch_from_east, 'orientation': 'east'},
        'de': {'element': RailEnvTransitionsEnum.dead_end_from_north, 'orientation': 'south'},
    },
    'west': {
        'empty': {'element': RailEnvTransitionsEnum.empty, 'orientation': 'west'},
        'w': {'element': RailEnvTransitionsEnum.horizontal_straight, 'orientation': 'west'},
        'n': {'element': RailEnvTransitionsEnum.right_turn_from_east, 'orientation': 'north'},
        's': {'element': RailEnvTransitionsEnum.right_turn_from_south, 'orientation': 'south'},
        'ssws': {'element': RailEnvTransitionsEnum.simple_switch_east_right, 'orientation': 'west'},
        'ssse': {'element': RailEnvTransitionsEnum.simple_switch_north_right, 'orientation': 'south'},
        'ssen': {'element': RailEnvTransitionsEnum.simple_switch_west_right, 'orientation': 'west'},
        'ssen_n': {'element': RailEnvTransitionsEnum.simple_switch_west_right, 'orientation': 'north'},
        'ssne': {'element': RailEnvTransitionsEnum.simple_switch_south_left, 'orientation': 'north'},
        'sswn': {'element': RailEnvTransitionsEnum.simple_switch_east_left, 'orientation': 'west'},
        'sses': {'element': RailEnvTransitionsEnum.simple_switch_west_left, 'orientation': 'west'},
        'sses_s': {'element': RailEnvTransitionsEnum.simple_switch_west_left, 'orientation': 'south'},
        'c': {'element': RailEnvTransitionsEnum.diamond_crossing, 'orientation': 'west'},
        'sslnw': {'element': RailEnvTransitionsEnum.single_slip_NW, 'orientation': 'west'},
        'sslsw': {'element': RailEnvTransitionsEnum.single_slip_SW, 'orientation': 'west'},
        'ssles': {'element': RailEnvTransitionsEnum.single_slip_SE, 'orientation': 'west'},
        'ssles_s': {'element': RailEnvTransitionsEnum.single_slip_SE, 'orientation': 'south'},
        'sslne': {'element': RailEnvTransitionsEnum.single_slip_NE, 'orientation': 'west'},
        'sslne_n': {'element': RailEnvTransitionsEnum.single_slip_NE, 'orientation': 'north'},
        'dslne': {'element': RailEnvTransitionsEnum.double_slip_NE_SW, 'orientation': 'west'},
        'dslne_n': {'element': RailEnvTransitionsEnum.double_slip_NE_SW, 'orientation': 'north'},
        'dslnw': {'element': RailEnvTransitionsEnum.double_slip_NW_SE, 'orientation': 'west'},
        'dslnw_n': {'element': RailEnvTransitionsEnum.double_slip_NW_SE, 'orientation': 'south'},
        'sysn': {'element': RailEnvTransitionsEnum.symmetric_switch_from_north, 'orientation': 'north'},
        'syse_n': {'element': RailEnvTransitionsEnum.symmetric_switch_from_east, 'orientation': 'north'},
        'syse_s': {'element': RailEnvTransitionsEnum.symmetric_switch_from_east, 'orientation': 'south'},
        'syss': {'element': RailEnvTransitionsEnum.symmetric_switch_from_south, 'orientation': 'south'},
        'de': {'element': RailEnvTransitionsEnum.dead_end_from_east, 'orientation': 'west'},
    },
}


def go_to_next_cell(x: int, y: int, orientation: str) -> tuple[int, int]:
    """Method used to travel to next cell based on the x,y and orientation.

    :param x: The x coordinate of the cell
    :param y: The y coordinate of the cell
    :param orientation: The orientation of the cell
    :return: The next cell coordinate.
    """

    if orientation == 'north':
        return x, y + 1
    if orientation == 'east':
        return x + 1, y
    if orientation == 'south':
        return x, y - 1
    assert orientation == 'west', f'Invalid orientation: {orientation}'
    return x - 1, y


def convert_coordinates(grid: np.array, x: int, y: int, x_origin: int = 0, y_origin: int = 0) -> tuple[int, int]:
    """Maps Cartesian coordinates (x, y) to grid indices (i, j) in a 2D NumPy array for flatland.

    :param grid: 2D NumPy array with Cartesian coordinates.
    :param x: x-coordinate
    :param y: y-coordinate
    :param x_origin: x-origin of Cartesian coordinate
    :param y_origin: y-origin of Cartesian coordinate
    :return: tuple with grid indices (i, j), where i is the row index (from top),
           and j is the column index.
    """

    n, e = grid.shape

    i = n - y - 1 - y_origin
    j = x + x_origin

    return i, j


def place_element(
    grid: np.array, x: int, y: int, orientation: str, element: str, x_origin: int = 0, y_origin: int = 0
) -> tuple[np.array, int, int, str]:
    """Place an element on the grid at the given positions.

    :param grid: Numpy array containing the int_transitions.
    :param x: X-coordinate
    :param y: Y-coordinate
    :param orientation: Either 'north', 'east', 'south' or 'west'
    :param element: Name of the element to place.
    :param x_origin: Origin of the x-coordinate.
    :param y_origin: Origin of the y-coordinate.
    :return: 3-element tuple with the updated grid, coordinates of next cell and orientation of the element.
    """
    # place element and adjust orientation
    element, orientation = elements[orientation][element].values()
    i, j = convert_coordinates(grid, x, y, x_origin, y_origin)
    grid[i][j] = element.value

    # go to next cell
    x, y = go_to_next_cell(x, y, orientation)

    return grid, x, y, orientation


def line_generator_from_line(line: Line) -> Callable:
    """Dummy line generator that returns a given schedule (line).

    :param line: Line to return.
    :return: Callable generation function.
    """

    def line_generator(*args, **kwargs) -> Line:
        """Dummy generator that returns the given line

        :return: Line given.
        """
        _ = (args, kwargs)
        return line

    return line_generator


def timetable_generator_from_timetable(timetable: Timetable) -> Callable:
    """Dummy timetable generator that returns a given timetable.

    :param timetable: Timetable to return
    :return: Callable generation function.
    """

    def timetable_generator(*args, **kwargs) -> Timetable:
        """Returns the given timetable.

        :return: Timetable given.
        """
        _ = (args, kwargs)
        return timetable

    return timetable_generator
