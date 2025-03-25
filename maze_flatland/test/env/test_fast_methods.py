"""File holding tests for fast methods."""
from __future__ import annotations

import numpy as np
from flatland.envs.fast_methods import fast_delete as old_fast_delete
from flatland.envs.fast_methods import fast_where as old_fast_where
from maze_flatland.env.fast_methods import fast_delete as new_fast_delete
from maze_flatland.env.fast_methods import fast_where as new_fast_where


def test_fast_delete():
    """Tests fast delete method."""
    test_list = [1, 2, 3, 4]
    test_delete_idx = 2
    assert np.all(new_fast_delete(test_list, test_delete_idx) == [1, 2, 4])
    assert np.all(old_fast_delete(test_list, test_delete_idx) == new_fast_delete(test_list, test_delete_idx))

    test_list = [10, 20, 30, 40]
    test_delete_idx = 0
    assert np.all(new_fast_delete(test_list, test_delete_idx) == [20, 30, 40])
    assert np.all(old_fast_delete(test_list, test_delete_idx) == new_fast_delete(test_list, test_delete_idx))

    test_list_last_element = [1, 2, 3]
    test_delete_idx = len(test_list_last_element) - 1
    assert np.all(new_fast_delete(test_list_last_element, test_delete_idx) == [1, 2])
    assert np.all(
        old_fast_delete(test_list_last_element, test_delete_idx)
        == new_fast_delete(test_list_last_element, test_delete_idx)
    )

    # Careful, new_fast_delete can handle more than the old version
    test_edge_case_list_a = []
    test_delete_idx = 0
    assert not new_fast_delete(test_edge_case_list_a, test_delete_idx)
    with np.testing.assert_raises(IndexError):
        old_fast_delete(test_edge_case_list_a, test_delete_idx)

    test_edge_case_list_b = [1, 2, 3]
    test_delete_idx = 10
    assert np.all(new_fast_delete(test_edge_case_list_b, test_delete_idx) == [1, 2, 3])
    with np.testing.assert_raises(IndexError):
        old_fast_delete(test_edge_case_list_b, test_delete_idx)


def test_fast_where():
    """Tests fast where method.
    This also tests different input types. That includes types that are not specified in the old methods to ensure
    backwards-compatibility.
    """
    test_list = [0, 1, 0, 2, 0, 3]
    assert np.all(new_fast_where(test_list) == [1, 3, 5])
    assert np.all(old_fast_where(test_list) == new_fast_where(test_list))

    test_tuple = (0, 1, 0, 2, 0, 3)
    assert np.all(new_fast_where(test_tuple) == [1, 3, 5])
    assert np.all(old_fast_where(test_tuple) == new_fast_where(test_tuple))

    test_float_list = [0.0, 1.1, 0.0, 2.2, 0.0, 3.3]
    assert np.all(new_fast_where(test_float_list) == [1, 3, 5])
    assert np.all(old_fast_where(test_float_list) == new_fast_where(test_float_list))

    test_boolean_list = [False, True, False]
    assert np.all(new_fast_where(test_boolean_list) == [1])
    assert np.all(old_fast_where(test_boolean_list) == new_fast_where(test_boolean_list))

    test_edge_case_list_a = []
    assert not new_fast_where(test_edge_case_list_a)
    assert np.all(old_fast_where(test_edge_case_list_a) == new_fast_where(test_edge_case_list_a))

    test_edge_case_list_b = [0, 0, 0]
    assert not new_fast_where(test_edge_case_list_b)
    assert np.all(old_fast_where(test_edge_case_list_b) == new_fast_where(test_edge_case_list_b))
