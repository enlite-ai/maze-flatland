"""File holding the basic flatland action conversion.."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionConversionInterface
from maze_flatland.env.masking.mask_builder import TrainLogicMask
from maze_flatland.env.maze_action import FlatlandMazeAction
from maze_flatland.env.maze_state import FlatlandMazeState, MazeTrainState
from maze_flatland.space_interfaces.action_conversion.base import FlatlandActionConversionInterface


class BasicAC(FlatlandActionConversionInterface):
    """Basic flatland action conversion."""

    step_key = 'train_move'
    action_space = gym.spaces.Discrete(len(FlatlandMazeAction))

    # pylint: disable=unused-argument
    @override(ActionConversionInterface)
    def space_to_maze(self, action: dict[str, int], maze_state: FlatlandMazeState) -> FlatlandMazeAction:
        """
        See :py:meth:`~maze.core.env.action_conversion.ActionConversionInterface.space_to_maze`.
        """
        if isinstance(action, FlatlandMazeAction):
            return action

        action = FlatlandMazeAction(action[self.step_key])
        return action

    @classmethod
    def action_from_masking(cls, mask: list[int] | np.ndarray[int]) -> dict[str, int]:
        """Return the only possible action given a mask.

        :param mask: The mask to extract the action from.
        :return: A dictionary holding the only possible action given a mask.
        """
        assert np.count_nonzero(mask) == 1
        return {cls.step_key: np.argmax(mask)}

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
        raise NotImplementedError()
