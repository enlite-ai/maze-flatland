"""File holding specific tests for the behavior of agents in maze-flatland and flatland-rl when facing dead-ends."""
from __future__ import annotations

from flatland.envs.step_utils.states import TrainState
from maze_flatland.env.masking.mask_builder import LogicMaskBuilder
from maze_flatland.env.maze_action import FlatlandMazeAction
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.env.maze_state import FlatlandMazeState
from maze_flatland.test.env_instantation import create_dummy_env_dead_end
from maze_flatland.wrappers.masking_wrapper import FlatlandMaskingWrapper


def go_to_dead_end(env: FlatlandEnvironment) -> None:
    """Perform sequence of actions such as the agent is facing a dead-end.

    :param env: The environment to step.
    """
    env.seed(213)  # seed does not matter. Configuration is fixed.
    _ = env.reset()
    while env.get_maze_state().trains[0].status == TrainState.WAITING:
        _ = env.step({'train_move': 4})
    _ = env.step({'train_move': 2})  # dispatch
    _ = env.step({'train_move': 2})  # move
    train_state = env.get_maze_state().current_train()
    assert train_state.actions_state[FlatlandMazeAction.DEVIATE_LEFT].target_cell == (1, 0), 'Agent not on switch'
    # agent goes left on the dead end.
    _ = env.step({'train_move': 1})
    train_state = env.get_maze_state().current_train()
    assert train_state.position == (1, 0)  # agent is on the border...
    assert train_state.direction == 3  # agent going out of map...
    assert train_state.status == TrainState.MOVING


def test_do_nothing_facing_dead_end_moving_reverse_dir():
    """Test that doing do-nothing while moving
    results into reversing the direction while staying on the same cell."""

    env = create_dummy_env_dead_end(1)
    go_to_dead_end(env)
    # bypass the maze_env for the do_nothing as this is not allowed by the action_conversion.
    # the outcome is equivalent to go_forward.
    rail_env = env.rail_env
    _ = rail_env.step({0: 0})
    # trigger the maze_state creation
    train_state = FlatlandMazeState(
        0,
        env._last_modifying_actions,  # pylint: disable=protected-access
        rail_env,
    ).current_train()
    assert train_state.direction == 1  # agent going east...
    assert train_state.position == (1, 1)  # agent is on the switch cell.
    assert train_state.status == TrainState.MOVING


def test_do_nothing_facing_dead_end_from_stop_does_nothing():
    """Test that do-nothing, while being stopped,
    results into being in the same cell, in the same state."""

    env = create_dummy_env_dead_end(1)
    go_to_dead_end(env)

    _ = env.step({'train_move': 4})  # stop
    train_state = env.get_maze_state().current_train()
    assert train_state.direction == 3  # agent going west...
    assert train_state.position == (1, 0)  # agent is on the border
    assert train_state.status == TrainState.STOPPED
    # bypass the maze_env for the do_nothing as this is not allowed by the action_conversion.
    rail_env = env.rail_env
    _ = rail_env.step({0: 0})
    # trigger the maze_state creation
    train_state = FlatlandMazeState(
        0,
        env._last_modifying_actions,  # pylint: disable=protected-access
        rail_env,
    ).current_train()
    assert train_state.direction == 3  # agent going east...
    assert train_state.position == (1, 0)  # agent is on the border
    assert train_state.status == TrainState.STOPPED


def test_move_forward_facing_dead_end_moving_reverse_dir():
    """Test that doing MOVE_FORWARD while moving and being on a dead-end results into
    reversing the direction of movement while staying on the same cell."""

    env = create_dummy_env_dead_end(1)
    go_to_dead_end(env)

    _ = env.step({'train_move': 2})
    train_state = env.get_maze_state().current_train()
    assert train_state.direction == 1  # agent going east...
    assert train_state.position == (1, 1)  # agent is on the switch.
    assert train_state.status == TrainState.MOVING


def test_move_forward_facing_dead_end_from_stop_reverse_dir():
    """Test that doing MOVE_FORWARD from a STOPPED status and being on a dead-end results into
    reversing the direction of movement while staying on the same cell."""

    env = create_dummy_env_dead_end(1)

    go_to_dead_end(env)
    _ = env.step({'train_move': 4})  # do the stop action
    train_state = env.get_maze_state().current_train()
    assert train_state.direction == 3  # agent going west...
    assert train_state.position == (1, 0)  # agent is on the border
    assert train_state.status == TrainState.STOPPED

    _ = env.step({'train_move': 2})
    train_state = env.get_maze_state().current_train()
    assert train_state.direction == 1  # agent going east...
    assert train_state.position == (1, 1)  # agent is on the switch.
    assert train_state.status == TrainState.MOVING


def test_move_forward_not_masked_when_facing_dead_end():
    """Test that action MOVE_FORWARD is not masked out while facing a dead-end."""
    env = FlatlandMaskingWrapper.wrap(env=create_dummy_env_dead_end(1), mask_builder=LogicMaskBuilder())

    go_to_dead_end(env)
    obs, _, done, _ = env.step({'train_move': 4})  # do the stop action
    mask = obs.get('train_move_mask', None)
    assert mask is not None and not done
    assert mask[2], 'Expected GO_FORWARD (2) to be active.'
    assert mask[4], 'Expected STOP_MOVING (4) 2) to be active.'
    assert sum(mask) == 2

    train_state = env.get_maze_state().current_train()
    assert train_state.direction == 3  # agent going west...
    assert train_state.position == (1, 0)  # agent is on the border
    assert train_state.should_reverse_dir()
    assert train_state.status == TrainState.STOPPED

    obs, _, done, _ = env.step({'train_move': 2})
    train_state = env.get_maze_state().current_train()
    assert not train_state.should_reverse_dir()
    assert train_state.direction == 1  # agent going east...
    assert train_state.position == (1, 1)  # agent is on the switch.
    assert train_state.status == TrainState.MOVING

    mask = obs.get('train_move_mask', None)
    assert mask is not None
    assert mask[4], 'Expected STOP_MOVING (4) to be active as ' 'no connection to target is found.'
