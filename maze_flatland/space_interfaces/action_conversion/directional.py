"""
ActionConversion for multi-agent Flatland environment.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from flatland.envs.step_utils.states import TrainState
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionConversionInterface
from maze_flatland.env.backend_utils import direction_to_pos_change_map
from maze_flatland.env.masking.mask_builder import TrainLogicMask
from maze_flatland.env.maze_action import FlatlandMazeAction
from maze_flatland.env.maze_state import FlatlandMazeState, MazeTrainState
from maze_flatland.space_interfaces.action_conversion.base import FlatlandActionConversionInterface


class DirectionalAC(FlatlandActionConversionInterface):
    """Specifies conversion between space actions and MazeActions.
        Related actions refer to a single train.

    :param step_key: Identifier for the action space.
    :param action_space: Action space.
    """

    step_key = 'train_move'
    action_space = gym.spaces.Discrete(len(FlatlandMazeAction))

    @override(ActionConversionInterface)
    def space_to_maze(self, action: dict[str, int], maze_state: FlatlandMazeState) -> FlatlandMazeAction:
        """
        See :py:meth:`~maze.core.env.action_conversion.ActionConversionInterface.space_to_maze`.
        """
        if isinstance(action, FlatlandMazeAction):
            return action

        action = FlatlandMazeAction(action[self.step_key])

        train = maze_state.trains[maze_state.current_train_id]
        if action == FlatlandMazeAction.DO_NOTHING:
            assert (
                train.is_done()
                or train.status in [TrainState.WAITING, TrainState.MALFUNCTION_OFF_MAP]
                or (train.status == TrainState.MALFUNCTION and train.malfunction_time_left > 0)
                or train.unsolvable
            )
        return action

    def noop_action(self):
        """
        Return no_op action.
        :return: no_op action for flatland environment.
        """
        return {self.step_key: FlatlandMazeAction.DO_NOTHING}

    @classmethod
    def list_actions(cls) -> list[str]:
        """Returns all the actions available in the action space.

        :return: List of actions available in the action space as str."""
        action_str = [repr(a) for a in FlatlandMazeAction]
        # Format string to trim the action.
        return [a[-a[::-1].index(' ') :] for a in action_str]

    @staticmethod
    def to_boolean_mask(train_mask: TrainLogicMask, train_state: MazeTrainState) -> np.ndarray[bool]:
        """Parse a TrainMask instance into a mask fit for the action space.

        :param train_mask: The train mask to parse.
        :param train_state: The current state for the train.
        :return: A boolean mask for the action space.
        """
        mask = np.zeros(DirectionalAC.action_space.n, dtype=bool)
        if train_mask.in_transition:
            # enable sticky actions
            mask[train_state.last_action] = True
        elif train_mask.skip_decision:
            # enable do nothing
            mask[FlatlandMazeAction.DO_NOTHING] = True
        # assess positions.
        for possible_pos in train_mask.possible_next_positions:
            if np.all(possible_pos == train_state.position):
                mask[FlatlandMazeAction.STOP_MOVING] = True
                train_state.can_stop = True
                continue
            glob_diff = np.array(possible_pos) - np.array(train_state.position)
            action_direction = direction_to_pos_change_map.tolist().index(glob_diff.tolist())
            mask[train_state.get_action_for_direction(action_direction)] = True
        return mask
