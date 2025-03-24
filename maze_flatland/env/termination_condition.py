"""File holding the termination conditions for flatland."""
from __future__ import annotations

from abc import abstractmethod

import numpy as np
from maze.core.annotations import override
from maze_flatland.env.maze_state import FlatlandMazeState


class EarlyTerminationCondition:
    """Defines the interface for early termination cases supported within flatland."""

    @abstractmethod
    def check_for_termination(self, maze_state: FlatlandMazeState) -> bool:
        """Abstract method to check for termination.
        :param maze_state: Maze state to check for termination.
        :return: True if the episode should be terminated early, False otherwise.
        """


class DeadlockEarlyTermination(EarlyTerminationCondition):
    """Terminate if there is at least a train in a deadlock."""

    def check_for_termination(self, maze_state: FlatlandMazeState) -> bool:
        """Implementation of abstract method that terminates iff a deadlock is found."""
        for train in maze_state.trains:
            if train.deadlock:
                return True
        return False


class NoDelayEarlyTermination(EarlyTerminationCondition):
    """Terminate if, from the current state, there is a train that will not be able to get to
    their destination in time.

    Note: this is particularly helpful to collect perfect trajectories.
    """

    def check_for_termination(self, maze_state: FlatlandMazeState) -> bool:
        """Implementation of abstract method that terminates iff a train is late."""
        for train in maze_state.trains:
            if train.is_done():
                continue
            # time need to reach the target is greater than the time left.
            if train.target_distance / train.speed > train.time_left_to_scheduled_arrival:
                return True
        return False


class OutOfTimeEarlyTermination(EarlyTerminationCondition):
    """Terminate if, from the current state, there is a train that will not be able to reach its destination
    within the episode time.
    """

    def check_for_termination(self, maze_state: FlatlandMazeState) -> bool:
        """Implementation of abstract method that terminates iff a train is late."""
        for train in maze_state.trains:
            if train.is_done():
                continue
            if train.out_of_time:
                return True
        return False


class BaseEarlyTermination(EarlyTerminationCondition):
    """Base class for early termination that terminates when all trains have arrived or max_time has elapsed."""

    @classmethod
    def _check_no_train_can_move(cls, maze_state: FlatlandMazeState) -> list[bool]:
        """Helper method to check if a train can move in the current state.
        :param maze_state: Maze state to check for termination.
        :return: True if the episode should be terminated, False otherwise."""
        if maze_state.env_time >= maze_state.max_episode_steps:
            return np.ones(maze_state.n_trains, dtype=bool)
        trains_done = np.asarray([train.is_done() for train in maze_state.trains])
        dead_trains = np.asarray([train.deadlock for train in maze_state.trains])
        trains_path_not_connected = np.asarray([3 == len(train.dead_ends) for train in maze_state.trains])
        return (trains_done + dead_trains + trains_path_not_connected) > 0

    def check_for_termination(self, maze_state: FlatlandMazeState) -> bool:
        """Base implementation of termination condition when all trains have either arrived or cannot arrive
        cause are in a deadlock or on a not connected path.
        """
        train_cannot_move = self._check_no_train_can_move(maze_state)
        return np.all(train_cannot_move)


class IncludeOutOfTimeTrainsInEarlyTermination(BaseEarlyTermination):
    """Overrides the BaseEarlyTermination class to include the trains that can move in the current state but cannot
    reach their destination.
    """

    @classmethod
    @override(BaseEarlyTermination)
    def _check_no_train_can_move(cls, maze_state: FlatlandMazeState) -> list[bool]:
        base_termination = super()._check_no_train_can_move(maze_state)
        trains_out_of_time = np.asarray([train.out_of_time for train in maze_state.trains])
        return np.logical_or(base_termination, trains_out_of_time)
