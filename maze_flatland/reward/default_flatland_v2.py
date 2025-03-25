"""
Base class for Flatland version 2 reward aggregator.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from maze.core.annotations import override
from maze.core.env.reward import RewardAggregatorInterface
from maze_flatland.env.events import TrainBlockEvents, TrainMovementEvents
from maze_flatland.env.maze_action import FlatlandMazeAction
from maze_flatland.env.maze_state import FlatlandMazeState
from maze_flatland.reward.flatland_reward import FlatlandReward


class RewardAggregator(FlatlandReward):
    """
    Event aggregation object dealing with Flatland rewards.
    See https://flatland.aicrowd.com/getting-started/env.html#rewards.
    :param alpha: Factor for local penalty.
    :param beta: Factor for global reward.
    :param reward_for_goal_reached: Reward for train reaching its goal. alpha only sets penalty for train to zero in
    this case; reward_for_goal_reached assigns an additional reward for this case.
    :param penalty_for_start: Penalty for starting a train. Flatland uses 0 internally.
    :param penalty_for_stop: Penalty for starting a train. Flatland uses 0 internally.
    :param use_train_speed: Whether to multiply local reward per train with their train speed.
    :param penalty_for_block: Penalty for a blocked train.
    :param penalty_for_deadlock: Penalty for a deadlocked train.
    :param distance_penalty_weight: Weight to compute penalty for distance to target. Distance is computed as geodesic
    distance, i.e. the min. number of cells to traverse to reach goal.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        reward_for_goal_reached: float,
        penalty_for_start: float,
        penalty_for_stop: float,
        use_train_speed: bool,
        penalty_for_block: float,
        penalty_for_deadlock: float,
        distance_penalty_weight: float,
    ):
        super().__init__()

        self._alpha = alpha
        self._beta = beta
        self._reward_for_goal_reached = reward_for_goal_reached
        self._penalty_for_start = penalty_for_start
        self._penalty_for_stop = penalty_for_stop
        self._use_train_speed = use_train_speed
        self._penalty_for_block = penalty_for_block
        self._penalty_for_deadlock = penalty_for_deadlock
        self._distance_penalty_weight = distance_penalty_weight

    def summarize_reward(self, maze_state: Optional[FlatlandMazeState] = None) -> Union[float, np.ndarray]:
        """
        Summarize reward based on train positions.
        :return: Reward per train as one-dimensional numpy array.
        """

        # Fetch all emitted events.
        move_events = {evt.train_id: evt for evt in self.query_events([TrainMovementEvents.train_moved])}
        block_events = {evt.train_id: evt for evt in self.query_events([TrainBlockEvents.train_blocked])}
        deadlock_events = {evt.train_id: evt for evt in self.query_events([TrainBlockEvents.train_deadlocked])}

        # Make sure all trains are in queried events exactly once.
        assert set(move_events.keys()) == set(
            range(maze_state.n_trains)
        ), f'{set(move_events.keys())} vs {set(range(maze_state.n_trains))}'

        # Check whether all trains have reached their goal.
        all_goals_reached: bool = all(move_events[tid].goal_reached for tid in move_events)

        return np.asarray(
            [
                # Global reward.
                self._beta * all_goals_reached
                - (
                    (
                        # Local penalty.
                        self._alpha * (not move_events[tid].goal_reached)
                        # Start and stop penalties.
                        + self._penalty_for_stop * (move_events[tid] == FlatlandMazeAction.STOP_MOVING)
                        + self._penalty_for_start
                        * (
                            maze_state.trains[tid].last_action == FlatlandMazeAction.STOP_MOVING
                            and move_events[tid] not in (FlatlandMazeAction.STOP_MOVING, FlatlandMazeAction.DO_NOTHING)
                        )
                    )
                    * move_events[tid].train_speed
                    if self._use_train_speed
                    else 1
                )
                # Local reward.
                + self._reward_for_goal_reached * move_events[tid].goal_reached
                # Block penalties.
                - self._penalty_for_block * (tid in block_events)
                # Deadlock penalties.
                - self._penalty_for_deadlock * (tid in deadlock_events)
                # Distance penalty.
                - self._distance_penalty_weight
                * np.nan_to_num(move_events[tid].target_distance, posinf=np.prod(maze_state.map_size) + 1)
                for tid in move_events
            ]
        )

    def to_scalar_reward(self, rewards: np.ndarray) -> float:
        """
        Sum up rewards for individual trains.
        :param rewards: Rewards per train.
        :return: Total reward over all trains.
        """

        return rewards.sum()

    @override(RewardAggregatorInterface)
    def get_interfaces(self):
        """
        Implementation of :py:meth:`~maze.core.events.pubsub.Subscriber.get_interfaces`.
        """

        return [TrainMovementEvents, TrainBlockEvents]

    # assumed that the state of an environment is cloned within an instance with the same reward definitions,
    # i.e., same parameters.
