"""Defines a delay based reward function for the multi-agent case."""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from flatland.envs.step_utils.states import TrainState
from maze.core.annotations import override
from maze.core.env.maze_state import MazeStateType
from maze.core.env.reward import RewardAggregatorInterface
from maze_flatland.reward.flatland_reward import FlatlandReward


class DelayBasedReward(FlatlandReward):
    """Defines the reward model for the flatland environment based on the expected and true delay at arrival.
        semi-sparse formulation. A penalty is given iff train is late upon its schedule, cannot reach its target or is
         in a deadlock.
         A positive reward is given whenever a train arrives to its target.
    :param epsilon_reward: Small reward used to penalise non-optimal behaviours.
    """

    def __init__(
        self,
        epsilon_reward: float,
    ):
        super().__init__()
        self._n_trains = None
        self._last_tstep = None
        self._last_reward = None
        self.already_arrived = []
        self.epsilon_reward = epsilon_reward

    def reset_distances(self):
        """Resets the reward aggregator by resetting the distances."""
        self._last_tstep = None
        self._last_reward = None
        self.already_arrived = []

    @override(RewardAggregatorInterface)
    def summarize_reward(self, maze_state: Optional[MazeStateType] = None) -> Union[float, np.ndarray]:
        """Summarize the reward based on the expected delay at arrival for each train.
        :param maze_state: Current environment state.
        :return: trains reward as a numpy array.
        """
        if maze_state.env_time == 1:
            self._n_trains = maze_state.n_trains
            self.reset_distances()
        if maze_state.env_time == self._last_tstep:
            assert self._last_reward is not None
            return self._last_reward

        rewards = []
        for idx in range(self._n_trains):
            max_delay = maze_state.max_episode_steps - maze_state.trains[idx].latest_arrival
            episode_remaining_time = maze_state.max_episode_steps - maze_state.env_time
            assert max_delay >= 0
            assert episode_remaining_time >= 0
            # if not departed, arrived already or in transition -> 0
            # if dead -> penalty -1
            if maze_state.trains[idx].deadlock:
                rewards.append(-1)

            elif (
                maze_state.trains[idx].status == TrainState.WAITING
                or idx in self.already_arrived
                or maze_state.trains[idx].in_transition
            ):
                rewards.append(0)

            # if arrived -> 1 - normalised_delay | normalised delay in (0, 1]
            elif maze_state.trains[idx].status == TrainState.DONE:
                true_delay = max(0, maze_state.trains[idx].arrival_delay)
                rew = 1 - (true_delay / (max_delay + 1))
                rewards.append(rew)
                self.already_arrived.append(idx)
            else:
                current_distance_to_target = min(
                    maze_state.trains[idx].target_distance, np.prod(maze_state.map_size) + 1
                )
                estimated_travel_time = current_distance_to_target * int(np.reciprocal(maze_state.trains[idx].speed))
                # train still can make in time -> 0
                if maze_state.trains[idx].time_left_to_scheduled_arrival >= estimated_travel_time:
                    rewards.append(0)

                # still can arrive but late -> small penalty
                elif estimated_travel_time <= episode_remaining_time:
                    # if arrives, it will be late
                    estimated_delay = (
                        maze_state.env_time + estimated_travel_time - maze_state.trains[idx].latest_arrival
                    )
                    assert 0 < estimated_delay <= max_delay
                    late_penalty = -0.1 * self.epsilon_reward * (estimated_delay / (max_delay + 1))
                    rewards.append(late_penalty)

                # train is lost already -> negative reward
                else:
                    # can't arrive.
                    rewards.append(-self.epsilon_reward)
        assert len(rewards) == self._n_trains
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

    def clone_from(self, reward_aggregator: DelayBasedReward) -> None:
        """Clone the given aggregator parameters into the current one.
        :param reward_aggregator: the instance to be cloned
        """
        self.deserialize_state(reward_aggregator.serialize_state())

    def serialize_state(self) -> list[any]:
        """serialize the current state of the reward aggregator.
        :return: List of parameters to be serialized.
        """
        return [self._n_trains, self._last_tstep, self._last_reward, self.already_arrived, self.epsilon_reward]

    def deserialize_state(self, serialized_state: list[any]) -> None:
        """Deserialize the state given into the current reward aggregator.
        :param serialized_state: the list of deserialized parameters."""
        self._n_trains = serialized_state[0]
        self._last_tstep = serialized_state[1]
        self._last_reward = serialized_state[2]
        self.already_arrived = serialized_state[3]
        self.epsilon_reward = serialized_state[4]
