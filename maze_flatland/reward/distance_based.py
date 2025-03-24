"""Defines a distance based reward function suited for the single agent case."""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from maze.core.annotations import override
from maze.core.env.maze_state import MazeStateType
from maze.core.env.reward import RewardAggregatorInterface
from maze_flatland.reward.flatland_reward import FlatlandReward


class DeltaDistanceReward(FlatlandReward):
    """Defines the reward model for the flatland environment.
    :param detour_penalty_factor: The penalty factor to use for.
    """

    def __init__(
        self,
        detour_penalty_factor: float,
    ):
        super().__init__()
        self.distance_to_target = None
        self._n_trains = None
        self._detour_penalty_factor = detour_penalty_factor
        self._last_tstep = None
        self._last_reward = None
        assert detour_penalty_factor >= 0

    def reset_distances(self):
        """Resets the reward aggregator by resetting the distances."""
        self.distance_to_target = [-1 for _ in range(self._n_trains)]
        self._last_tstep = None
        self._last_reward = None

    @override(RewardAggregatorInterface)
    def summarize_reward(self, maze_state: Optional[MazeStateType] = None) -> Union[float, np.ndarray]:
        """Summarize the reward as change in distance between previous maze_state.env_time and current.
        :return: Scalar reward as sum of rewards across agents.
        """
        self._n_trains = maze_state.n_trains if self._n_trains is None else self._n_trains

        if maze_state.env_time == 1:
            self.reset_distances()
        rewards = []
        if maze_state.env_time == self._last_tstep:
            assert self._last_reward is not None
            return self._last_reward
        for idx in range(self._n_trains):
            if maze_state.trains[idx].has_not_yet_departed():
                rewards.append(0)
            else:
                old_distance = self.distance_to_target[idx]
                current_distance_to_target = min(
                    maze_state.trains[idx].target_distance, np.prod(maze_state.map_size) + 1
                )
                if old_distance == -1:
                    rewards.append(1)
                else:
                    rew = old_distance - current_distance_to_target
                    rewards.append(rew if rew > 0 else rew * self._detour_penalty_factor)
                self.distance_to_target[idx] = current_distance_to_target
            self._last_reward = rewards
            self._last_tstep = maze_state.env_time
        return np.array(self._last_reward)

    def to_scalar_reward(self, rewards: np.ndarray) -> float:
        """
        Sum up rewards for individual trains.
        :param rewards: Rewards per train.
        :return: Total reward over all trains.
        """

        return rewards.sum()

    def clone_from(self, reward_aggregator: DeltaDistanceReward) -> None:
        """Clone the given aggregator parameters into the current one.
        :param reward_aggregator: the instance to be cloned
        """
        self.deserialize_state(reward_aggregator.serialize_state())

    def serialize_state(self) -> list[any]:
        """serialize the current state of the reward aggregator."""
        return [
            self.distance_to_target,
            self._n_trains,
            self._detour_penalty_factor,
            self._last_tstep,
            self._last_reward,
        ]

    def deserialize_state(self, serialized_state: list[any]):
        """Deserialize the state given into the current reward aggregator."""
        self.distance_to_target = serialized_state[0]
        self._n_trains = serialized_state[1]
        self._detour_penalty_factor = serialized_state[2]
        self._last_tstep = serialized_state[3]
        self._last_reward = serialized_state[4]
