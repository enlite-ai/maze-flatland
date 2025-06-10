"""File holdings the test for Decision Point action masking."""

from __future__ import annotations

import unittest
from collections import namedtuple

import numpy as np
from flatland.envs.step_utils.states import TrainState
from maze.core.utils.seeding import MazeSeeding
from maze_flatland.env.masking.mask_builder import LogicMaskBuilder, build_mask_out_of_map
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
            cond_1 = train_status in [
                TrainState.WAITING,
                TrainState.MALFUNCTION_OFF_MAP,
                TrainState.DONE,
            ]
            cond_2 = (
                train_status.is_malfunction_state() and state.trains[state.current_train_id].malfunction_time_left > 0
            )
            cond_3 = state.current_train().deadlock
            time_to_target = 1 + state.env_time + state.current_train().best_travel_time_to_target
            cond_4 = time_to_target > state.max_episode_steps  # will not arrive in time.
            assert cond_1 or cond_2 or cond_3 or cond_4

        if sum(obs['train_move_mask']) > 1:
            assert not env.logic_mask_builder.create_train_mask(
                state.trains[state.current_train_id], env.rail_env
            ).only_single_option(), 'Expected multiple decisions, found only 1.'
        else:
            assert env.logic_mask_builder.create_train_mask(
                state.trains[state.current_train_id], env.rail_env
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
    env.seed(2812)
    rng = np.random.RandomState(1235)
    obs = env.reset()
    dead_end_action = FlatlandMazeAction.GO_FORWARD
    train_state = env.get_maze_state().trains[0]
    while train_state.env_time <= 14:
        # check that the dead end is found.
        if train_state.env_time == 13:
            assert (
                train_state.actions_state[dead_end_action].target_cell is not None
            ), 'Expected a valid cell instead of None.'
            assert np.isinf(train_state.actions_state[dead_end_action].goal_distance), 'Expected a dead end path.'
            assert not obs['train_move_mask'][dead_end_action], ' Action is not masked out.'
        action = rng.choice(np.where(obs['train_move_mask'])[0])
        obs, _, _, info = env.step({'train_move': action})
        train_state = env.get_maze_state().trains[0]


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

    done = False
    while not done:
        action = 2
        obs, rew, done, info = env.step({'train_move': action})
        state: FlatlandMazeState = env.get_maze_state()
        if state.env_time == 33:
            assert state.trains[0].deadlock
            assert state.trains[0].actions_state[FlatlandMazeAction.GO_FORWARD].obstructed_by == 1
            assert state.trains[1].actions_state[FlatlandMazeAction.GO_FORWARD].obstructed_by == 0
            assert state.trains[1].deadlock
            assert not state.trains[2].deadlock
            flat_step_masks = np.asarray(env.get_mask_for_flat_step())
            for tid in [0, 1]:
                assert sum(flat_step_masks[tid]) == 1, f'More than 1 option possible for train {tid}'
                assert flat_step_masks[tid][-1], f'"Stop action not enabled for train {tid}'
            break


def test_decision_point_masking_with_cloning():
    """Check two masks are equivalent after cloning the environment."""
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


class TestOutOfMapMasking(unittest.TestCase):
    dummy_train_state = namedtuple(
        'MazeTrainState',
        [
            'handle',
            'env_time',
            'max_episode_steps',
            'status',
            'position',
            'malfunction_time_left',
            'best_travel_time_to_target',
            'unsolvable',
            'actions_state',
        ],
    )
    dummy_action_state = namedtuple('MazeActionState', ['target_cell'])

    def test_unsolvable(self):
        """Check that when a train is unsolvable, the decision is skipped."""
        train_state = self.dummy_train_state(1, 0, 10, TrainState.READY_TO_DEPART, (10, 10), 0, 2, True, None)
        logic_mask = build_mask_out_of_map(train_state)
        assert logic_mask.skip_decision, logic_mask.explain()

    def test_out_of_time(self):
        """Check that when the travel time is greater than the time left, the decision is skipped.

        travel_time -> 10
        time_to_dispatch -> 1
        time_left -> 10
        """
        train_state = self.dummy_train_state(1, 0, 10, TrainState.READY_TO_DEPART, (10, 10), 0, 10, False, None)
        logic_mask = build_mask_out_of_map(train_state)
        assert logic_mask.skip_decision, logic_mask.explain()

    def test_can_depart_no_malfunction(self):
        """Check that a train can successfully depart."""
        action_state = {FlatlandMazeAction.GO_FORWARD: self.dummy_action_state((10, 11))}
        train_state = self.dummy_train_state(1, 0, 10, TrainState.READY_TO_DEPART, (10, 10), 0, 9, False, action_state)
        logic_mask = build_mask_out_of_map(train_state)

        assert not logic_mask.skip_decision, logic_mask.explain()
        assert len(logic_mask.possible_next_positions) == 2, logic_mask.explain()

    def test_can_depart_with_malf(self):
        """Check that a train can successfully depart when malfunction_time_left=1"""
        action_state = {FlatlandMazeAction.GO_FORWARD: self.dummy_action_state((10, 11))}
        train_state = self.dummy_train_state(
            1, 0, 10, TrainState.MALFUNCTION_OFF_MAP, (10, 10), 1, 9, False, action_state
        )
        logic_mask = build_mask_out_of_map(train_state)

        assert not logic_mask.skip_decision, logic_mask.explain()
        assert len(logic_mask.possible_next_positions) == 2, logic_mask.explain()

    def test_out_of_map(self):
        """Check that decision is skipped for a waiting train."""
        train_state = self.dummy_train_state(1, 0, 10, TrainState.WAITING, (10, 10), 0, 9, False, None)
        logic_mask = build_mask_out_of_map(train_state)

        assert logic_mask.skip_decision, logic_mask.explain()

    def test_malfunction_off_map(self):
        """Check that decision is skipped for a train malfunctioning off map
        and with malfunction_time_left>1.
        """
        train_state = self.dummy_train_state(1, 0, 10, TrainState.MALFUNCTION_OFF_MAP, (10, 10), 2, 9, False, None)
        logic_mask = build_mask_out_of_map(train_state)

        assert logic_mask.skip_decision, logic_mask.explain()
