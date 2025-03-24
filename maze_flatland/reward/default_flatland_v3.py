"""
Base class for Flatland version 3 reward aggregator.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from maze_flatland.env.maze_state import FlatlandMazeState
from maze_flatland.reward.flatland_reward import FlatlandReward


class RewardAggregator(FlatlandReward):
    """
    Sparse reward formulation for flatland version 3.
    See https://flatland.aicrowd.com/getting-started/env.html#rewards.
    :param cancellation_factor: Factor for trains being canceled (never departed)
    :param cancellation_time_buffer: extra penalty time added for trains that did not leave the origin point.
    """

    def __init__(
        self,
        cancellation_factor: int,
        cancellation_time_buffer: int,
    ):
        super().__init__()
        self._cancellation_factor = cancellation_factor
        self._cancellation_time_buffer = cancellation_time_buffer

    def summarize_reward(self, maze_state: Optional[FlatlandMazeState] = None) -> Union[float, np.ndarray]:
        """
        Summarize reward based on train positions.
        :return: Reward per train as one-dimensional numpy array.
        """

        rewards = np.zeros(maze_state.n_trains)
        if maze_state.terminate_episode:
            for train in maze_state.trains:
                if train.is_done():
                    rewards[train.handle] -= train.arrival_delay
                else:  # agent is not done
                    theoretical_optimal_time_to_target = int(
                        np.ceil(min(train.target_distance, 1 + np.prod(maze_state.map_size)) / train.speed)
                    )
                    if train.has_not_yet_departed():  # never departed
                        rewards[train.handle] += (
                            -1
                            * self._cancellation_factor
                            * (theoretical_optimal_time_to_target + self._cancellation_time_buffer)
                        )
                    else:  # train has departed but never reached
                        # get time left to latest arrival
                        time_left_to_latest_arrival = train.time_left_to_scheduled_arrival
                        if train.deadlock:
                            # override the time_left
                            time_left_to_latest_arrival = train.latest_arrival - train.max_episode_steps

                        rewards[train.handle] += time_left_to_latest_arrival
                        # time needed to travel to the target
                        rewards[train.handle] -= theoretical_optimal_time_to_target
        return rewards

    def to_scalar_reward(self, rewards: np.ndarray) -> float:
        """
        Sum up rewards for individual trains.
        :param rewards: Rewards per train.
        :return: Total reward over all trains.
        """

        return rewards.sum()

    # assumed that the state of an environment is cloned within an instance with the same reward definitions,
    # i.e., same parameters.


class ChallengeScore(RewardAggregator):
    """
    Normalised sparse reward formulation for flatland version 3.
    See https://flatland.aicrowd.com/challenges/flatland3/eval.html
    """

    def __init__(self):
        super().__init__(cancellation_factor=1, cancellation_time_buffer=0)

    def summarize_reward(self, maze_state: Optional[FlatlandMazeState] = None) -> Union[float, np.ndarray]:
        """Intercepts summarize_reward and normalises it based on the current env. configuration.
        :param maze_state: Current flatland maze state
        :return: Rewards normalised.
        """
        if not maze_state.terminate_episode:
            return np.zeros(maze_state.n_trains)
        rewards = super().summarize_reward(maze_state)
        return (rewards / (maze_state.max_episode_steps * maze_state.n_trains)) + 1 / maze_state.n_trains
