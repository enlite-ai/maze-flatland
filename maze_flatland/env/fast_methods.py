"""File holding fast methods for the flatland environment."""
from __future__ import annotations

import numpy as np


def fast_where(binary_iterable: np.ndarray | list) -> np.ndarray:
    """Returns first element of `binary_iterable` which is not zero.

    :param binary_iterable: Binary iterable to get the first non-zero element from.
    :return: First non-zero element of `binary_iterable`.
    """
    return np.nonzero(binary_iterable)[0]


def fast_delete(lis: list, index: int) -> list:
    """Removes index from given list.

    :param lis: List with an element to remove.
    :param index: Index of element to remove.
    :return: List with element removed.
    """
    return lis[:index] + lis[index + 1 :]
