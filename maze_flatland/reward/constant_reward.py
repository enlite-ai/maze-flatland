"""Constant reward for the Flatland env.."""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from flatland.envs.step_utils.states import TrainState
from maze_flatland.env.maze_state import FlatlandMazeState
from maze_flatland.reward.flatland_reward import FlatlandReward


def needs_reward(state: TrainState) -> bool:
    """Decides whether a reward is needed based on the current state.
    :param state: state of the train to get the reward.
    :return: False if the train is malfunctioning, arrived to its destination or waiting to spawn.
             True otherwise.
    """
    if state.is_malfunction_state() or state in [TrainState.DONE, TrainState.WAITING]:
        return False
    return True


class ConstantReward(FlatlandReward):
    """Return a constant reward for each timestep.

    :param value: The constant reward value to use for each timestamp.
    """

    def __init__(self, value: float):
        super().__init__()
        self._value = value

    def summarize_reward(self, maze_state: Optional[FlatlandMazeState] = None) -> Union[float, np.ndarray]:
        """Always return the constant reward when reward is called.

        :param maze_state: Current state of the environment.
        :return: Reward for the last structured step. In most cases, either a scalar or an array with an item
                 for each actor active during the last step.
        """

        return np.array([self._value * needs_reward(train.status) for train in maze_state.trains])

    def to_scalar_reward(self, rewards: np.ndarray) -> float:
        """
        Sum up rewards for individual trains.
        :param rewards: Rewards per train.
        :return: Total reward over all trains.
        """

        return rewards.sum()


class ConstantMinusOneReward(ConstantReward):
    """Dummy reward to use as a default"""

    def __init__(self):
        super().__init__(-1)
