from __future__ import annotations

import numpy as np
from flatland.envs.step_utils.states import TrainState
from maze_flatland.env.maze_action import FlatlandMazeAction


def _compare(value: any, other_value: any, verbose: bool = False):
    """Compare if two flatland's objects are equivalent."""
    equivalent = False
    if isinstance(value, FlatlandMazeAction):
        equivalent = value == other_value
    elif hasattr(value, '__dict__') or isinstance(value, np.random.RandomState):
        equivalent = check_if_equal(value, other_value, verbose)
    elif isinstance(value, np.ndarray) and value.dtype.kind == 'f':
        if np.all(np.isclose(value, other_value)):
            equivalent = True
    elif isinstance(value, np.ndarray):
        if np.all(value == other_value):
            equivalent = True
    elif isinstance(value, dict) or hasattr(value, '__parameters__'):
        equivalent = check_if_equal(value, other_value, verbose)
    elif isinstance(value, tuple):
        equivalent = check_if_equal(value, other_value, verbose)
    elif isinstance(value, list):
        equivalent = check_if_equal(tuple(value), tuple(other_value), verbose)
    else:
        if value == other_value:
            equivalent = True
    return equivalent


# pylint: disable=too-many-branches
def check_if_equal(obj1: any, obj2: any, verbose: bool = False) -> bool:
    """Test if the two flatlands' objects are equivalent.

    :param obj1: The first object instance.
    :param obj2: The second object instance.
    """
    equivalent = True
    if isinstance(obj1, TrainState):
        equivalent = obj1 == obj2
    elif isinstance(obj1, np.random.RandomState):
        equivalent = _compare(obj1.get_state(), obj2.get_state(), verbose)
    elif isinstance(obj1, tuple) and isinstance(obj2, tuple):
        for _obj1, _obj2 in zip(obj1, obj2):
            equivalent = equivalent and _compare(_obj1, _obj2, verbose)
            if not equivalent:
                if verbose:
                    print(f'{_obj1} and {_obj2} are not equivalent')
                break
    elif isinstance(obj1, dict) and isinstance(obj2, dict):
        for key in obj1:
            if key not in obj2:
                if verbose:
                    print(f'{key} not in object2.')
                return False
            value = obj1[key]
            other_value = obj2[key]
            equivalent = equivalent and _compare(value, other_value, verbose)
            if not equivalent:
                if verbose:
                    print(f'{key} not equivalent')
                break
    else:
        for key, value in obj1.__dict__.items():
            if not hasattr(obj2, key):
                if verbose:
                    print(f'{key} not exist in obj2')
                return False
            other_value = getattr(obj2, key)
            equivalent = equivalent and _compare(value, other_value, verbose)
            if not equivalent:
                if verbose:
                    print(f'{key} not equivalent')
                break
    return equivalent
