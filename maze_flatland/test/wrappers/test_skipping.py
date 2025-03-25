"""Holds the tests for the skipping wrapper"""

from __future__ import annotations

import numpy as np
from maze.core.rollout.rollout_generator import RolloutGenerator
from maze.core.trajectory_recording.records.trajectory_record import SpacesTrajectoryRecord
from maze_flatland.agents.greedy_policy import GreedyPolicy
from maze_flatland.env.masking.mask_builder import LogicMaskBuilder
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.env.maze_state import FlatlandMazeState
from maze_flatland.space_interfaces.action_conversion.directional import DirectionalAC
from maze_flatland.space_interfaces.observation_conversion.simple import SimpleObservationConversion
from maze_flatland.test.env_instantation import create_core_env
from maze_flatland.test.test_utils import check_if_equal
from maze_flatland.wrappers.masking_wrapper import FlatlandMaskingWrapper
from maze_flatland.wrappers.skipping_wrapper import FlatStepSkippingWrapper, SubStepSkippingWrapper


def assertions_on_maze_state(state_1: FlatlandMazeState, state_2: FlatlandMazeState) -> None:
    """Compare that two maze state are equivalent on some fields:
        - trains_positions
        - trains_status
        - blocks
        - trains_directions
    :param state_1: the 1st instance of the maze_state
    :param state_2: the 2nd instance of the maze_state"""
    for train_1, train_2 in zip(state_1.trains, state_2.trains):
        assert check_if_equal(train_1, train_2)


def create_example_env(
    n_trains: int,
    n_cities: int,
    max_rails_between_cities: int,
    max_rail_pairs_in_city: int,
) -> FlatlandEnvironment:
    """Create a small example env wrapped with masking wrapper."""
    core_env = create_core_env(
        n_trains, 30, 30, n_cities, 0, {1: 1}, False, max_rails_between_cities, max_rail_pairs_in_city
    )

    env = FlatlandEnvironment(
        core_env,
        {'train_move': DirectionalAC()},
        {'train_move': SimpleObservationConversion(False)},
    )
    return FlatlandMaskingWrapper.wrap(env, mask_builder=LogicMaskBuilder())


# pylint: disable=too-many-branches
def run_equivalency_test_for_flat_skipping(n_trains: int, seed: int, skip_on_reset: bool = False):
    """Steps two environment, with and without flat_step skipping, and checks for their equivalency.
    :param n_trains: the number of trains to initialize an environment with.
    :param seed: the seed to initialise the env.
    :param skip_on_reset: whether to enable skipping on the reset of the environment
    """
    rng = np.random.RandomState(seed)

    # create environment with skipping.
    skip_env = create_example_env(n_trains, 3, 2, 2)
    skip_env = FlatStepSkippingWrapper(skip_env, do_skipping_in_reset=skip_on_reset)
    skip_env.seed(seed)
    obs = skip_env.reset()

    decision_time = []
    actions_sequence_skipping = []
    rewards_skipping = []
    done = False
    # step the skip environment and collect action and rewards trajectories.
    while not done:
        state: FlatlandMazeState = skip_env.get_maze_state()
        mask = obs['train_move_mask']
        if mask.sum() == 1:
            action = np.where(mask)[0][0]
        else:
            action = rng.choice(np.where(mask)[0])
            actions_sequence_skipping.append(action)
        if state.current_train_id == 0:
            decision_time.append(state.env_time)
        obs, rew, done, info = skip_env.step({'train_move': action})
        if 'skipping.internal_rewards' in info:
            rewards_skipping.extend(info['skipping.internal_rewards'])
        else:
            rewards_skipping.append(rew)

    # create environment without skipping wrapper
    noskip_env = create_example_env(n_trains, 3, 2, 2)
    noskip_env.seed(seed)
    obs = noskip_env.reset()
    timestep_first_decision = decision_time[0]
    # remove the first n-1 substeps reward before stepping the environment if the skip on reset was enabled.
    rewards_skipping = rewards_skipping[skip_env.n_trains - 1 :] if skip_on_reset else rewards_skipping
    done = False
    # step the environment without skipping and check the equivalence
    while not done:
        state: FlatlandMazeState = noskip_env.get_maze_state()
        mask = obs['train_move_mask']
        if mask.sum() == 1:
            action = np.where(mask)[0][0]
        else:
            action = actions_sequence_skipping.pop(0)  # take action from skipping.
        if state.current_train_id == 0:
            masks = noskip_env.get_mask_for_flat_step()
            if max(np.sum(masks, axis=1)) > 1 or (state.env_time == 0 and not skip_on_reset):
                assert state.env_time == decision_time.pop(0), 'Mismatch in decision time.'
        obs, rew, done, info = noskip_env.step({'train_move': action})
        if noskip_env.get_maze_state().env_time > timestep_first_decision or not skip_on_reset:
            assert rew == rewards_skipping.pop(0), 'Mismatch in reward.'
    assert len(rewards_skipping) == len(actions_sequence_skipping) == len(decision_time) == 0
    assertions_on_maze_state(skip_env.get_maze_state(), noskip_env.get_maze_state())


# pylint: disable=too-many-branches
def run_equivalency_test_for_substep_skipping(n_trains: int, seed: int, skip_on_reset: bool = False):
    """Steps two environment, with and without sub_step skipping, and checks for their equivalency.
    :param n_trains: the number of trains to initialize an environment with.
    :param seed: the seed to initialise the env.
    :param skip_on_reset: whether to enable skipping on the reset of the environment
    """
    rng = np.random.RandomState(seed)

    # create environment with skipping.
    skip_env = create_example_env(n_trains, 3, 2, 2)
    skip_env = SubStepSkippingWrapper(skip_env, do_skipping_in_reset=skip_on_reset)
    skip_env.seed(seed)
    obs = skip_env.reset()
    timestep_first_decision = 0 if not skip_on_reset else skip_env.get_maze_state().env_time
    actions_sequence_skipping = []
    rewards_skipping = []
    done = False
    # step the skip environment and collect action and rewards trajectories.
    while not done:
        state: FlatlandMazeState = skip_env.get_maze_state()
        mask = obs['train_move_mask']
        if mask.sum() == 1:
            action = np.where(mask)[0][0]
        else:
            action = rng.choice(np.where(mask)[0])
            actions_sequence_skipping.append(action)
        obs, rew, done, info = skip_env.step({'train_move': action})
        if 'skipping.internal_rewards' in info:
            rewards_skipping.extend(info['skipping.internal_rewards'])
        else:
            rewards_skipping.append(rew)

    # create environment without skipping wrapper
    noskip_env = create_example_env(n_trains, 3, 2, 2)
    noskip_env.seed(seed)
    obs = noskip_env.reset()
    done = False
    # step the environment without skipping and check the equivalence
    while not done:
        state: FlatlandMazeState = noskip_env.get_maze_state()
        mask = obs['train_move_mask']
        if mask.sum() == 1:
            action = np.where(mask)[0][0]
        else:
            action = actions_sequence_skipping.pop(0)  # take action from skipping.
        obs, rew, done, info = noskip_env.step({'train_move': action})
        if state.env_time >= timestep_first_decision:
            assert rew == rewards_skipping.pop(0), 'Mismatch in reward.'
    assert len(rewards_skipping) == len(actions_sequence_skipping)
    assertions_on_maze_state(skip_env.get_maze_state(), noskip_env.get_maze_state())


def test_flat_step_skipping_single_train():
    """Tests equivalency with and without flat_step skipping in a single train scenario."""
    run_equivalency_test_for_flat_skipping(1, 9999)


def test_flat_step_skipping_multi_trains():
    """Tests equivalency with and without flat_step skipping in a 3-train scenario."""
    run_equivalency_test_for_flat_skipping(3, 9999)


def test_flat_step_skipping_at_reset_single_train():
    """Tests equivalency with and without flat_step skipping in a single train scenario
    while enabling skipping at reset level."""
    run_equivalency_test_for_flat_skipping(1, 9999, skip_on_reset=True)


def test_flat_step_skipping_at_reset_multi_trains():
    """Tests equivalency with and without flat_step skipping in a 3-train scenario
    while enabling skipping at reset level."""
    run_equivalency_test_for_flat_skipping(3, 9999, skip_on_reset=True)


def test_substep_skipping_single_train():
    """Tests equivalency with and without substep skipping in a single train scenario."""
    run_equivalency_test_for_substep_skipping(1, 9999)


def test_substep_skipping_multi_trains():
    """Tests equivalency with and without substep skipping in a 3-train scenario."""
    run_equivalency_test_for_substep_skipping(3, 9999)


def test_substep_skipping_at_reset_single_train():
    """Tests equivalency with and without substep skipping in a single train scenario
    while enabling skipping at reset level."""
    run_equivalency_test_for_substep_skipping(1, 9999, skip_on_reset=True)


def test_substep_skipping_at_reset_multi_trains():
    """Tests equivalency with and without substep skipping in a 3-train scenario
    while enabling skipping at reset level."""
    run_equivalency_test_for_substep_skipping(3, 9999, skip_on_reset=True)


def _create_multistep_skipping_trajectory(wrapper: any) -> SpacesTrajectoryRecord:
    """Creates a trajectory record for flatland with the wrapper given in input.
    :param wrapper: The wrapper to wrap the environment.
    :return: A SpacesTrajectoryRecord with the greedy policy.
    """
    env = create_example_env(3, 3, 2, 2)
    env = wrapper(env, do_skipping_in_reset=True) if wrapper is not None else env
    env.seed(1234)
    env.reset()
    rollout_generator = RolloutGenerator(
        env,
        record_logits=False,
        record_step_stats=True,
        record_episode_stats=True,
        record_next_observations=False,
        terminate_on_done=True,
    )
    return rollout_generator.rollout(GreedyPolicy(), 1000)
