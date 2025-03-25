"""File holdings the test for Decision Point action masking."""

from __future__ import annotations

import numpy as np
from flatland.envs.step_utils.states import TrainState
from maze.core.utils.seeding import MazeSeeding
from maze_flatland.env.backend_utils import get_transitions_map
from maze_flatland.env.masking.mask_builder import LogicMaskBuilder
from maze_flatland.env.maze_action import FlatlandMazeAction
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.env.maze_state import FlatlandMazeState
from maze_flatland.space_interfaces.action_conversion.directional import DirectionalAC
from maze_flatland.space_interfaces.observation_conversion.minimal import MinimalObservationConversion
from maze_flatland.test.env_instantation import create_core_env
from maze_flatland.wrappers.masking_wrapper import FlatlandMaskingWrapper


def _create_example_env(
    map_width: int,
    map_height: int,
    n_trains: int,
    malfunction_rate: float,
    n_cities: int,
    speed_table: dict[float, float],
    max_rails_between_cities: int,
    max_rail_pairs_in_city: int,
):
    """Create a small example env."""

    core_env = create_core_env(
        n_trains,
        map_width,
        map_height,
        n_cities,
        malfunction_rate,
        speed_table,
        False,
        max_rails_between_cities,
        max_rail_pairs_in_city,
    )
    env = FlatlandEnvironment(
        core_env,
        {'train_move': DirectionalAC()},
        {'train_move': MinimalObservationConversion(True)},
    )
    return FlatlandMaskingWrapper.wrap(env, mask_builder=LogicMaskBuilder())


def test_decision_point_action_masking_rollout():
    """Test the masking by running random rollouts"""
    env = _create_example_env(
        20,
        20,
        n_trains=4,
        malfunction_rate=0.9,
        n_cities=2,
        speed_table={1: 1},
        max_rails_between_cities=1,
        max_rail_pairs_in_city=1,
    )
    rng = np.random.RandomState(1235)
    for i in range(7):
        seed = MazeSeeding.generate_seed_from_random_state(rng)
        env.seed(seed)
        assert_env_transitions_and_mask(env, rng)
        print(f'rollout {i}/10 done')


def assert_env_transitions_and_mask(env: FlatlandEnvironment, rng: np.random.RandomState) -> None:
    """Assert the mask is created correctly."""
    obs = env.reset()
    done = False
    while not done:
        state: FlatlandMazeState = env.get_maze_state()
        current_train = state.current_train_id
        action = rng.choice(np.where(obs['train_move_mask'])[0])
        train_status = state.trains[current_train].status

        if obs['train_move_mask'][FlatlandMazeAction.DO_NOTHING]:
            cond_1 = state.trains[current_train].status in [
                TrainState.WAITING,
                TrainState.MALFUNCTION_OFF_MAP,
                TrainState.DONE,
            ]
            cond_2 = (
                train_status.is_malfunction_state() and state.trains[state.current_train_id].malfunction_time_left > 0
            )
            cond_3 = state.trains[current_train].deadlock
            assert cond_1 or cond_2 or cond_3

        if sum(obs['train_move_mask']) > 1:
            assert not env.logic_mask_builder.create_train_mask(
                state.trains[state.current_train_id], get_transitions_map(env.rail_env)
            ).only_single_option(), 'Expected multiple decisions, found only 1.'
        else:
            assert env.logic_mask_builder.create_train_mask(
                state.trains[state.current_train_id], get_transitions_map(env.rail_env)
            ).only_single_option(), 'Expected no decisions, found > 1.'

        obs, rew, done, info = env.step({'train_move': action})


def test_dead_end_case():
    """Test the case where the go forward is a dead end action"""
    env = _create_example_env(
        30,
        30,
        n_trains=1,
        malfunction_rate=0,
        n_cities=2,
        speed_table={1: 1},
        max_rails_between_cities=3,
        max_rail_pairs_in_city=3,
    )
    env.seed(1567173351)
    rng = np.random.RandomState(1235)
    obs = env.reset()

    done = False
    while not done:
        action = rng.choice(np.where(obs['train_move_mask'])[0])
        obs, rew, done, info = env.step({'train_move': action})
    assert env.get_maze_state().trains[0].is_done()


def test_mask_when_deadlock():
    """Test that no actions are allowed for deadlocked trains."""
    env = _create_example_env(
        30,
        30,
        n_trains=3,
        malfunction_rate=0,
        n_cities=3,
        speed_table={1: 1},
        max_rails_between_cities=3,
        max_rail_pairs_in_city=3,
    )
    env.seed(197251382)
    _ = env.reset()
    action_agent_0 = {16: 4, 17: 3, 18: 1, 20: 1}

    state: FlatlandMazeState = env.get_maze_state()
    done = False
    while not done:
        action = 2
        if state.current_train_id == 0 and state.env_time in action_agent_0:
            action = action_agent_0[state.env_time]

        obs, rew, done, info = env.step({'train_move': action})
        state: FlatlandMazeState = env.get_maze_state()
        if state.env_time == 21:
            flat_step_masks = np.asarray(env.get_mask_for_flat_step())
            # pylint: disable=protected-access
            assert (np.sum(flat_step_masks, axis=1) == 1).all()
            # check that no_op is set to 1 only for trains in a deadlock
            assert sum(flat_step_masks.T[4]) == 2 and flat_step_masks.T[3][0] == 0
            assert state.trains[0].deadlock and state.trains[1].deadlock
            break


def test_decision_point_masking_with_cloning():
    env1 = _create_example_env(
        30,
        30,
        n_trains=1,
        malfunction_rate=0,
        n_cities=2,
        speed_table={0.5: 1},
        max_rails_between_cities=3,
        max_rail_pairs_in_city=3,
    )
    env1.seed(1967693548)
    rng = np.random.RandomState(1235)
    obs = env1.reset()

    env2 = _create_example_env(
        30,
        30,
        n_trains=1,
        malfunction_rate=0,
        n_cities=2,
        speed_table={0.5: 1},
        max_rails_between_cities=3,
        max_rail_pairs_in_city=3,
    )
    for _ in range(25):
        action = rng.choice(np.where(obs['train_move_mask'])[0])
        obs, rew, done, info = env1.step({'train_move': action})
    env2.clone_from(env1)
    masks_e1 = env1.get_mask_for_flat_step()
    masks_e2 = env2.get_mask_for_flat_step()

    for m1, m2 in zip(masks_e1, masks_e2):
        assert (m1 == m2).all()
