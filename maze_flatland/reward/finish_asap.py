"""A simple constant reward for encourage the agent to finish as soon as possible."""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from flatland.envs.step_utils.states import TrainState
from maze_flatland.env.maze_state import FlatlandMazeState
from maze_flatland.reward.flatland_reward import FlatlandReward


class FinishASAPReward(FlatlandReward):
    """A simple reward that encourages the agent to finish an episode as soon as possible, and bootstraps the return
    computation by providing a larger final reward.

    For each step a constant negative rewards is returned unless the agent is off_map or waiting or done then it is 0.
    Also, when the episode is done, the reward computed is such that the return stays constant at
    -unsuccessful_trains/n_trains.

    :param gamma: The discounting factor used during training.
    :param max_value: The maximum value an episode should have at the flat steps.
    """

    def __init__(self, gamma: float, max_value: float):
        super().__init__()
        self._gamma = gamma
        self._max_value = max_value
        assert max_value < 0, 'Max value must be < 0 for this reward to work.'

    @staticmethod
    def step_reward(max_value: float, gamma: float, substeps: int) -> float:
        """Calculate the step reward such that the discounted return at the flat step is the same as the discounted
        return of the previous flat step if no agent arrived.

        :param max_value: The maximum value an episode should have at the flat steps.
        :param gamma: The discounting factor.
        :param substeps: The number of substeps to consider.
        """
        return (max_value * (1 - gamma**substeps) / gamma ** (substeps - 1)) / substeps

    @staticmethod
    def bootstrap_value(max_value: float, gamma: float, substeps: int) -> float:
        """Calculate the bootstrapping reward for the final env step such that the discounted return takes max_value
        at the previous flat step.

        :param max_value: The maximum value an episode should have at the flat steps.
        :param gamma: The discounting factor.
        :param substeps: The number of substeps to consider.
        """
        return max_value / gamma ** (substeps - 1) / substeps

    def summarize_reward(self, maze_state: Optional[FlatlandMazeState] = None) -> Union[float, np.ndarray]:
        """Always return the constant reward when reward is called.

        max value should be specified as value then

        rew = value * (1 - self._gamma**substeps)/self._gamma**(substeps - 1)
        and
        bootstrap = value / (self._gamma**(substeps - 1))

        :param maze_state: Current state of the environment.
        :return: Reward for the last structured step. In most cases, either a scalar or an array with an item
                 for each actor active during the last step.
        """
        rewards = []
        for train_id, train_state in enumerate(maze_state.trains):
            if (
                train_state.status in [TrainState.MALFUNCTION_OFF_MAP, TrainState.WAITING, TrainState.DONE]
                or maze_state.trains[train_id].unsolvable
            ):
                rewards.append(0)
            elif maze_state.terminate_episode:
                rewards.append(self.bootstrap_value(self._max_value, self._gamma, maze_state.n_trains))
            else:
                rewards.append(self.step_reward(self._max_value, self._gamma, maze_state.n_trains))

        return np.array(rewards)

    @classmethod
    def print_values(cls, max_value: float, gamma: float, substeps: int, num_steps: int) -> None:
        """Print the discounting values for debugging.

        :param max_value: The maximum value an episode should have at the flat steps.
        :param gamma: The discounting factor.
        :param substeps: The number of substeps to consider.
        :param num_steps: The total number of steps.
        """
        vv = 0
        for idx in range(num_steps)[::-1]:
            for idx_2 in range(substeps)[::-1]:
                if idx == num_steps - 1 and idx_2 == substeps - 1:
                    rr = cls.bootstrap_value(max_value, gamma, substeps) * substeps
                elif idx_2 == substeps - 1:
                    rr = cls.step_reward(max_value, gamma, substeps) * substeps
                else:
                    rr = 0
                vv = rr + gamma * vv
                print(idx, idx_2, f'{vv:.3f}', rr)

    def to_scalar_reward(self, rewards: np.ndarray) -> float:
        """
        Sum up rewards for individual trains.
        :param rewards: Rewards per train.
        :return: Total reward over all trains.
        """

        return rewards.sum()
