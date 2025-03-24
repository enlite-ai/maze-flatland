"""
Contains interfaces for events relevant in the Flatland environment.
"""

from __future__ import annotations

import abc

import numpy as np
from maze.core.log_stats.event_decorators import define_episode_stats, define_epoch_stats, define_step_stats


class TrainMovementEvents(abc.ABC):
    """
    Events to be triggered after each step.
    """

    @define_episode_stats(sum)
    @define_step_stats(len)
    def train_moved(self, train_id: int, goal_reached: bool, train_speed: float, target_distance: int):
        """
        Indicates that train moved. Is triggered in each step for each train.
        :param train_id: Train ID.
        :param goal_reached: Whether train reached its goal.
        :param train_speed: Train speed.
        :param target_distance: Geodesic distance to train target.
        """

    # [0,1], 1 -> all trains have arrived.
    @define_epoch_stats(np.mean, output_name='mean_success_rate')
    @define_episode_stats(sum)
    @define_step_stats(sum)
    def trains_arrived(self, success_rate: float):
        """Register the ratio of trains that have arrived at their destination during an episode.
        Triggered at done condition.
        :param success_rate: number of trains arrived to their destination divided by total number of trains.
        """

    @define_epoch_stats(np.mean, output_name='success_rate_over_possible')
    @define_episode_stats(sum)
    @define_step_stats(sum)
    def trains_arrived_possible(self, success_rate: float):
        """Register the ratio of trains that have arrived at their destination during an episode
            and that could have arrived.
        Triggered at done condition.
        :param success_rate: number of trains arrived to their destination divided by total number of trains.
        """

    @define_epoch_stats(np.mean, output_name='mean_delay_arrived_trains')
    @define_episode_stats(sum)
    @define_step_stats(sum)
    def train_delay(self, delay: int):
        """Records the delay of a train that has arrived to its destination.
        :param delay: necessary timesteps for trains to arrive at their destination after their latest tolerated
            time.
        """

    # trains that have not departed
    @define_epoch_stats(np.mean, output_name='mean_rate_cancelled_trains')
    @define_episode_stats(sum)
    @define_step_stats(sum)
    def trains_cancelled(self, cancelled_rate: float):
        """Records the number of trains cancelled.
        In flatland, a train is considered cancelled if it has not departed by the end of simulation.
        :param cancelled_rate: ratio of trains that never left their origin station divided by total number of
        trains.
        """

    @define_epoch_stats(np.mean, output_name='mean_n_trains')
    @define_episode_stats(max)
    @define_step_stats(max)
    def n_trains(self, n_trains: int):
        """
        Record the number of trains in the episode.
            :param n_trains: Count of trains in the episode.
        """


class TrainBlockEvents(abc.ABC):
    """
    Events to be triggered when trains are blocked.
    """

    @define_episode_stats(sum)
    @define_step_stats(len)
    def train_blocked(self, train_id: int):
        """
        Indicates that train is currently blocked from moving in any direction.
        :param train_id: Train ID.
        """

    @define_episode_stats(np.max)
    @define_step_stats(len)
    def train_deadlocked(self, train_id: int):
        """
        Indicates that train is deadlocked, i.e. blocked permanently. A deadlock can't be resolved and persists for the
        remainder of the episode.
        :param train_id: Train ID.
        """

    # average number of deadlocked trains (at last step) over the episodes
    @define_epoch_stats(np.mean, output_name='mean_episode_trains_deadlock')
    @define_episode_stats(max)
    @define_step_stats(max)
    def count_deadlocks(self, dead_rate: float):
        """
        Indicated the ids of trains that are in a deadlock.
        :param dead_rate: Ratio of trains in deadlock / all trains.
        """


class TrainDetouringEvents(abc.ABC):
    """
    Events to be triggered after each step.
    """

    @define_epoch_stats(np.mean, input_name='ep_sum_detour', output_name='mean_detour_delay')
    @define_episode_stats(np.mean, input_name='step_delay_detour', output_name='ep_sum_detour')
    @define_step_stats(np.mean, output_name='step_delay_detour')
    # count detouring
    @define_epoch_stats(np.mean, input_name='ep_num_detours', output_name='mean_num_detours')
    @define_episode_stats(sum, input_name='step_num_detours', output_name='ep_num_detours')
    @define_step_stats(len, output_name='step_num_detours')
    def train_detouring(self, expected_delay: int):
        """Indicates the amount of delay expected due to the detouring."""


class ScheduleEvents(abc.ABC):
    """
    Events to be triggered at the beginning of simulation.
    """

    @define_epoch_stats(np.mean, output_name='mean_ratio')
    @define_episode_stats(max)
    @define_step_stats(max)
    def impossible_dest(self, trains_ratio: float):
        """Indicated the amount of traint with an origin point not connected to the destination."""

    @define_epoch_stats(np.mean, output_name='mean_invalid_episodes')
    @define_episode_stats(sum)
    @define_step_stats(sum)
    def invalid_episode(self, episode_is_invalid: bool):
        """Tracks whether the episode is invalid."""


class SeedingEvent(abc.ABC):
    """
    Events that record the seeding information
    """

    def seed_used(self, value: int):
        """Record the seed used after each reset."""


class FlatlandExecutionEvents(abc.ABC):
    """Events recording the executions of the core env."""

    def action_taken(self, step_key: str, agent_id: int, action: int):
        """Record the execution taken."""


class FlatlandDepartingEvents(abc.ABC):
    """Records the events related to the departure of trains."""

    @define_epoch_stats(np.mean, output_name='mean_ratio')
    @define_episode_stats(max)
    @define_step_stats(max)
    def departure_asap(self, trains_ratio: float):
        """Indicate whether a train has departed as soon as possible."""

    @define_epoch_stats(np.mean, output_name='mean_ratio')
    @define_episode_stats(max)
    @define_step_stats(max)
    def departure_in_time(self, trains_ratio: float):
        """Indicate whether a train has departed in a time frame that allows it to arrive at the scheduled time."""

    @define_epoch_stats(np.mean, output_name='mean_ratio')
    @define_episode_stats(max)
    @define_step_stats(max)
    def departure_severe_delay(self, trains_ratio: float):
        """Indicate whether a train left the station at a time such as that the train will never arrive in time."""

    @define_epoch_stats(np.mean, output_name='mean_ratio')
    @define_episode_stats(np.mean)
    @define_step_stats(np.mean)
    def departure_delay(self, norm_delay: float):
        """Record the normalised delay."""
