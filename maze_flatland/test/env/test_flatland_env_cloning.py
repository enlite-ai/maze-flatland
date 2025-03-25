"""Tests cloning functionality of FlatlandEnvironment."""
from __future__ import annotations

import pickle
import random
from copy import deepcopy
from typing import Optional

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.rail_env import RailEnv
from maze_flatland.env.core_env import FlatlandCoreEnvironment
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.env.maze_state import FlatlandMazeState
from maze_flatland.space_interfaces.action_conversion.directional import DirectionalAC
from maze_flatland.space_interfaces.observation_conversion.positional import PositionalObservationConversion
from maze_flatland.test.env_instantation import create_core_env
from maze_flatland.test.test_utils import _compare, check_if_equal


# pylint: disable=protected-access
def compare_rail_env(env1: FlatlandCoreEnvironment, env2: FlatlandCoreEnvironment) -> None:
    """Compares the backend of two flatland environments.
    :param env1: first instance of environment to compare
    :param env2: second instance of environment to compare
    :return: True if the rail_env underneath is equivalent.
    """

    assert all(env1.rail_env.agent_positions.flatten() == env2.rail_env.agent_positions.flatten())
    assert (
        env1.rail_env.agents
        == env2.rail_env.agents
        == env1.rail_env.distance_map.agents
        == env2.rail_env.distance_map.agents
    )
    assert all(env1.rail_env.rail.grid.flatten() == env2.rail_env.rail.grid.flatten())
    assert env1.rail_env.rail.transitions.transitions_all == env2.rail_env.rail.transitions.transitions_all
    assert all(env1.rail_env.distance_map.distance_map.flatten() == env2.rail_env.distance_map.distance_map.flatten())
    assert env1.context.step_id == env2.context.step_id and env1.context._episode_id == env2.context._episode_id


def compare_backend(rail_env1: RailEnv, rail_env2: RailEnv):
    """Compares the equivalence of two instances of RailEnv.
    :param rail_env1: first instance to be compared.
    :param rail_env2: second instance to be compared.
    :return: list of params that do not match.
    """
    keys_to_exclude = ['seed_history', 'rail_generator', 'line_generator', 'malfunction_process_data', 'obs_dict']
    dict_re1 = rail_env1.__dict__
    dict_re2 = rail_env2.__dict__
    diff_keys = []
    for k in dict_re1:
        if k in keys_to_exclude:
            continue
        if (
            k == 'obs_builder'
            and type(dict_re1[k]) is type(dict_re2[k])
            and isinstance(dict_re1[k], DummyObservationBuilder)
        ):
            continue  # dummy observation builder mismatch otherwise.
        try:
            are_equal = _compare(dict_re1[k], dict_re2[k])
        except RecursionError:
            if dict_re1[k] != dict_re2[k]:
                diff_keys.append(k)
        if not are_equal:
            diff_keys.append(k)
    return diff_keys


def init_env_with_malfunctions(
    malf_rate: float, seed: int = 9999, n_trains: int = 2, include_maze_state_in_serialization: Optional[bool] = False
) -> FlatlandEnvironment:
    """Initialize an environment with a certain probability for a train to be subjected to a malfunction."""
    core_env: FlatlandCoreEnvironment = create_core_env(
        n_trains,
        30,
        30,
        2,
        malf_rate,
        {1.0: 0.7, 1.0 / 2.0: 0.3},
        include_maze_state_in_serialization=include_maze_state_in_serialization,
    )
    env = FlatlandEnvironment(
        core_env,
        action_conversion={'train_move': DirectionalAC()},
        observation_conversion={'train_move': PositionalObservationConversion(False)},
    )
    env.seed(seed)
    env.reset()
    return env


def _do_rollout(_env: FlatlandEnvironment, actions: list[int]) -> list[FlatlandMazeState]:
    """Step an environment with a certain sequence of actions.
    :param _env: environment to step
    :param actions: list of actions to take
    """
    visited_states = []
    for idx, a in enumerate(actions):
        visited_states.append(deepcopy(_env.get_maze_state()))
        _, _, done, _ = _env.step({'train_move': a})
        if done:
            visited_states.append(deepcopy(_env.get_maze_state()))
            return visited_states
    return visited_states


def test_equivalent_rollout_after_cloning():
    """check the cloning functionality by comparing the states visited while doing twice the same rollout"""
    custom_env = init_env_with_malfunctions(0.5)

    serialised_initial_state = custom_env.serialize_state()
    action_sequence = [random.randint(1, 4) for _ in range(300)]

    first_sequence_of_states = _do_rollout(custom_env, action_sequence[:50])
    serialised_mid_state = custom_env.serialize_state()
    custom_env.deserialize_state(serialised_initial_state)  # restore initial state
    second_sequence_of_states = _do_rollout(custom_env, action_sequence[:50])
    third_sequence_of_states = _do_rollout(custom_env, action_sequence[50:])
    custom_env.deserialize_state(serialised_mid_state)  # restore mid state
    fourth_sequence_of_states = _do_rollout(custom_env, action_sequence[50:])

    assert all((check_if_equal(s1, s2) for s1, s2 in zip(first_sequence_of_states, second_sequence_of_states)))
    assert all((check_if_equal(s1, s2) for s1, s2 in zip(third_sequence_of_states, fourth_sequence_of_states)))


def test_clone_from_environment():
    """Check the state equivalence of two environments after stepping one and cloning it into another."""
    e1 = init_env_with_malfunctions(0.7)
    e2 = init_env_with_malfunctions(0.7)

    _do_rollout(e1, [random.randint(1, 4) for _ in range(100)])
    e2.clone_from(e1)
    compare_rail_env(e1, e2)
    env_ob1 = list(e1.observation_conversion.observation_builders.values())[0]
    env_ob2 = list(e1.observation_conversion.observation_builders.values())[0]
    assert env_ob2 == env_ob1
    assert check_if_equal(e1.get_maze_state(), e2.get_maze_state())


def test_parallel_env_after_cloning():
    """Check the equivalence of visited states after cloning an already stepped environment into a fresh one."""
    e1 = init_env_with_malfunctions(0.5)
    e2 = init_env_with_malfunctions(0.5)

    _do_rollout(e1, [random.randint(1, 4) for _ in range(100)])
    e2.clone_from(e1)
    actions = [random.randint(1, 4) for _ in range(200)]
    states_visited_in_e1 = _do_rollout(e1, actions)
    states_visited_in_e2 = _do_rollout(e2, actions)
    assert all((check_if_equal(s1, s2) for s1, s2 in zip(states_visited_in_e1, states_visited_in_e2)))


def test_clone_different_seed():
    """Check the state equivalence of two environments after stepping one and cloning it into another initialised with
    a different seed."""
    e1 = init_env_with_malfunctions(0.2)

    _do_rollout(e1, [random.randint(1, 4) for _ in range(50)])
    e2 = init_env_with_malfunctions(0.7, seed=42)

    e2.clone_from(e1)
    actions = [random.randint(1, 4) for _ in range(100)]
    states_visited_in_e1 = _do_rollout(e1, actions)
    states_visited_in_e2 = _do_rollout(e2, actions)
    diff_keys = compare_backend(e1.rail_env, e2.rail_env)
    assert len(diff_keys) == 0, f'Keys mismatch in the backend.\n{diff_keys}'
    compare_rail_env(e1, e2)
    assert len(states_visited_in_e1) == len(states_visited_in_e2)
    assert all((check_if_equal(s1, s2) for s1, s2 in zip(states_visited_in_e1, states_visited_in_e2)))


def test_clone_different_seed_single_train():
    """Check the state equivalence of two environments after stepping one and cloning it into another initialised with
    a different seed."""
    e1 = init_env_with_malfunctions(0.2, n_trains=1)

    _do_rollout(e1, [random.randint(1, 4) for _ in range(50)])
    e2 = init_env_with_malfunctions(0.7, seed=42, n_trains=1)
    _do_rollout(e2, [random.randint(1, 4)])
    e2.clone_from(e1)
    actions = [random.randint(1, 4) for _ in range(100)]
    states_visited_in_e1 = _do_rollout(e1, actions)
    states_visited_in_e2 = _do_rollout(e2, actions)
    diff_keys = compare_backend(e1.rail_env, e2.rail_env)
    assert len(diff_keys) == 0, f'Keys mismatch in the backend.\n{diff_keys}'
    compare_rail_env(e1, e2)
    assert len(states_visited_in_e1) == len(states_visited_in_e2)
    assert all((check_if_equal(s1, s2) for s1, s2 in zip(states_visited_in_e1, states_visited_in_e2)))


def test_deserialization_of_serialised_state():
    """Tests the equivalence between two environments when deserializing a serialised state."""
    env = init_env_with_malfunctions(0.2, n_trains=1)
    cloned_env = init_env_with_malfunctions(0, seed=42, n_trains=1)
    actions = [random.randint(1, 4) for _ in range(20)]
    serialised_states = []
    for a in actions:
        serialised_states.append(env.serialize_state())
        _ = env.step({'train_move': a})

    cloned_env.deserialize_state(serialised_states[10])
    for a in actions[10:]:
        _ = cloned_env.step({'train_move': a})
    compare_rail_env(env, cloned_env)
    compare_backend(env.rail_env, cloned_env.rail_env)


def test_serialization_and_deserialization():
    """Check the state equivalence of two environments after stepping one and cloning it into another."""
    e1 = init_env_with_malfunctions(malf_rate=1 / 100, include_maze_state_in_serialization=True)
    e2 = init_env_with_malfunctions(malf_rate=1 / 100, include_maze_state_in_serialization=False)

    # Check for same maze states
    assert check_if_equal(e1.get_maze_state(), e2.get_maze_state())

    # Serialize
    serialized_with_maze_state = e1.serialize_state()
    serialized_without_maze_state = e2.serialize_state()

    # Deserialize
    e1.deserialize_state(serialized_with_maze_state)
    e2.deserialize_state(serialized_without_maze_state)

    # Check for same maze states
    assert check_if_equal(e1.get_maze_state(), e2.get_maze_state())


def test_nested_cloning():
    """Test that serialize(env) is equal to serialize(clone_from(env))."""
    e1 = init_env_with_malfunctions(malf_rate=1 / 10, include_maze_state_in_serialization=True)
    e2 = init_env_with_malfunctions(malf_rate=1 / 100, include_maze_state_in_serialization=True, seed=6789)
    # step e1 with n actions.
    for _ in range(30):
        action = {'train_move': random.randint(1, 4)}
        _ = e1.step(action)
    e2.clone_from(e1)
    serialized_state_s1 = e1.core_env.serialize_state()
    serialized_clone = e2.core_env.serialize_state()
    assert check_if_equal(pickle.loads(serialized_clone), pickle.loads(serialized_state_s1))
    assert check_if_equal(e1._rail_env_rnd_state_for_malfunctions, e2._rail_env_rnd_state_for_malfunctions)
