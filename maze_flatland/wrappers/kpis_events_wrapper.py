"""File holding a simple wrapper to record events and KPIs. To be used with TrajectoryRecordingWrapper."""
from __future__ import annotations

from typing import Any

import numpy as np
from flatland.envs.step_utils.states import TrainState
from maze.core.env.maze_env import MazeEnv
from maze.core.wrappers.wrapper import Wrapper


class KpisEventsWrapper(Wrapper[MazeEnv]):
    """Wrapper used to catch events and KPIs and include them into the info dict.

    :param env: MazeEnv to wrap.
    """

    def __init__(self, env: MazeEnv):
        super().__init__(env)
        self.onmap_time = np.zeros(self.n_trains)

    def reset(self):
        """Intercepts the base reset method."""
        self.onmap_time = np.zeros(self.n_trains)
        return self.env.reset()

    # pylint: disable=protected-access
    def step(self, action) -> tuple[Any, Any, bool, dict[Any, Any]]:
        """Intercept ``BaseEnv.step`` and tracks the metrics."""
        maze_state = self.env.get_maze_state()
        ready_to_depart_handles = (train.status == TrainState.READY_TO_DEPART for train in maze_state.trains)
        shortest_path_pre_step = [train.best_travel_time_to_target for train in maze_state.trains]

        observation, reward, done, info = self.env.step(action)
        if not self.is_flat_step():
            return observation, reward, done, info

        maze_state = self.env.get_maze_state()
        # extend info with KPIs
        expected_delay = {
            train.handle: (
                max(0, (train.best_travel_time_to_target - train.time_left_to_scheduled_arrival))
                if not train.is_done()
                else train.arrival_delay
            )
            for train in maze_state.trains
        }
        dead_ratio = np.mean([train.deadlock for train in maze_state.trains])
        info['kpis'] = {}

        if not done:
            trains_in_time = sum(1 for delay in expected_delay.values() if delay <= 0)
            trains_in_delay = {train_id: delay for train_id, delay in expected_delay.items() if delay > 0}
            for train in maze_state.trains:
                # increment counter if train is on map.
                self.onmap_time[train.handle] += train.is_on_map()
            for train_id, delay in trains_in_delay.items():
                info['kpis'][f'Train {train_id} expected delay'] = delay
                info['kpis']['on schedule'] = 100 * trains_in_time / maze_state.n_trains

        else:
            info['kpis']['Arrival ratio'] = np.mean([train.is_done() for train in maze_state.trains])
            avg_delay = np.mean([train.arrival_delay for train in maze_state.trains])
            info['kpis']['Avg. delay'] = avg_delay
            cancelled = np.mean([train.out_of_time and train.has_not_yet_departed() for train in maze_state.trains])
            info['kpis']['Cancelled'] = cancelled
            info['kpis']['Avg. travel time'] = np.mean(self.onmap_time)
        if dead_ratio > 0:
            info['kpis']['Deadlock ratio'] = dead_ratio

        # fill in the events...
        events = []
        for train in maze_state.trains:
            if train.is_done() or train.has_not_yet_departed():
                continue
            if train.deadlock:
                events.append(f'[Deadlock] Train {train.handle} at position {train.position}')
            if train.is_on_map() and train.handle in ready_to_depart_handles:
                delta_time_dispatch = maze_state.env_time - maze_state.trains[train.handle].earliest_departure
                events.append(
                    f'[Dispatch] Train {train.handle} after {delta_time_dispatch} steps from earliest departure'
                )
            travel_time_delta = train.best_travel_time_to_target - shortest_path_pre_step[train.handle]
            if travel_time_delta > 0:
                events.append(
                    f'[Detour] Train {train.handle} detour from planned path. Expected delay of {travel_time_delta}'
                )
        info['events'] = events
        return observation, reward, done, info
