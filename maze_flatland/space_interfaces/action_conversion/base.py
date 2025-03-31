"""File holding the Flatland interface for the action conversion."""
from __future__ import annotations

from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionConversionInterface
from maze_flatland.env.masking.mask_builder import TrainLogicMask
from maze_flatland.env.maze_action import FlatlandMazeAction
from maze_flatland.env.maze_state import MazeTrainState


class FlatlandActionConversionInterface(ActionConversionInterface, ABC):
    """Specific interface class for the action conversion in flatland."""

    @property
    @abstractmethod
    def step_key(self) -> str:
        """Parameter to be specified in the subclass

        :return: String identifier for the action space.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def action_space(self):
        """Parameter to be specified in the subclass.

        :return: Action space.
        """
        raise NotImplementedError

    @classmethod
    def action_from_masking(cls, mask: list[int] | np.ndarray[int]) -> dict[str, int]:
        """Return the only possible action given a mask.

        :param mask: The mask to extract the action from.
        :return: A dictionary holding the only possible action given a mask.
        """
        assert np.count_nonzero(mask) == 1
        return {cls.step_key: np.argmax(mask)}

    @classmethod
    @override(ActionConversionInterface)
    def maze_to_space(cls, maze_action: FlatlandMazeAction) -> dict[str, int]:
        """
        See :py:meth:`~maze.core.env.action_conversion.ActionConversionInterface.maze_to_space`.
        """
        return {cls.step_key: maze_action.value}

    def space(self) -> gym.spaces.Space:
        """
        See :py:meth:`~maze.core.env.action_conversion.ActionConversionInterface.space`.
        """
        return gym.spaces.Dict({self.step_key: self.action_space})

    @classmethod
    @abstractmethod
    def list_actions(cls) -> list[str]:
        """Returns all the actions available in the action space.

        :return: List of actions available in the action space as str.
        """

    @staticmethod
    @abstractmethod
    def to_boolean_mask(train_mask: TrainLogicMask, train_state: MazeTrainState) -> np.ndarray[bool]:
        """Parse a TrainMask instance into a mask fit for the action space.

        :param train_mask: The train mask to parse.
        :param train_state: The current state for the train.
        :return: A boolean mask for the action space.
        """
        raise NotImplementedError()
