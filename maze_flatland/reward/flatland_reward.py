"""Interface class for flatland rewards."""
from __future__ import annotations

from abc import abstractmethod

import numpy as np
from maze.core.env.reward import RewardAggregatorInterface


class FlatlandReward(RewardAggregatorInterface):
    """Interface class for flatland rewards"""

    @abstractmethod
    def to_scalar_reward(self, rewards: np.ndarray) -> float:
        """
        Get an overall scalar values of the individual rewards
        :param rewards: Rewards per train.
        :return: Total reward over all trains.
        """

    def serialize_state(self) -> list:
        """Serialize the state of the reward aggregator."""
        return []

    def deserialize_state(self, serialized_state: list):
        """Deserialize the state given into the current reward aggregator."""
        _ = serialized_state
