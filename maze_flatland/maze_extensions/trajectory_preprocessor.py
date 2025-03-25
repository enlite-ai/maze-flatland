"""File holding maze extensions for trajectory processing"""
from __future__ import annotations

import numpy as np
from flatland.envs.step_utils.states import TrainState
from maze.core.annotations import override
from maze.core.trajectory_recording.datasets.trajectory_processor import TrajectoryProcessor, retrieve_done_info
from maze.core.trajectory_recording.records.trajectory_record import StateTrajectoryRecord, TrajectoryRecord


class RemoveAllLostTrains(TrajectoryProcessor):
    """Trajectory preprocessor that removes all trajectories where no train arrived at the destination."""

    @override(TrajectoryProcessor)
    def pre_process(self, trajectory: TrajectoryRecord) -> TrajectoryRecord:
        """Implementation of
        :class:`~maze.core.trajectory_recording.datasets.trajectory_processor.TrajectoryProcessor` interface."""

        assert not isinstance(trajectory, StateTrajectoryRecord), 'StateTrajectoryRecord not supported.'
        last_info = retrieve_done_info(trajectory)[-1]
        any_done = np.any(np.array(list(last_info['state'].values())) == TrainState.DONE)
        if any_done:
            assert any(np.array(last_info['mcts_node_info--dist_to_target']) == 0)
        else:
            assert all(np.array(last_info['mcts_node_info--dist_to_target']) > 0)

        if not any_done:
            trajectory.step_records = []

        return trajectory


class FilterOnlyArrivedTrains(TrajectoryProcessor):
    """Trajectory preprocessor that cherry-picks from the trajectories the
    trains that have arrived."""

    @override(TrajectoryProcessor)
    def pre_process(self, trajectory: TrajectoryRecord) -> TrajectoryRecord:
        """Implementation of
        :class:`~maze.core.trajectory_recording.datasets.trajectory_processor.TrajectoryProcessor` interface."""

        assert not isinstance(trajectory, StateTrajectoryRecord), 'StateTrajectoryRecord not supported.'
        last_info = retrieve_done_info(trajectory)[-1]
        train_states = np.asarray(list(last_info['state'].values()))
        trains_arrived = np.where(train_states == TrainState.DONE)[0]
        sr_to_be_removed = []
        for idx_sr, sr in enumerate(trajectory.step_records):
            ssr_to_be_removed = []
            for idx_ssr, ssr in enumerate(sr.substep_records):
                if ssr.actor_id.agent_id not in trains_arrived:
                    ssr_to_be_removed.append(idx_ssr)
            for idx in ssr_to_be_removed[::-1]:
                del sr.substep_records[idx]
            if len(sr.substep_records) == 0:
                sr_to_be_removed.append(idx_sr)
        for jdx in sr_to_be_removed[::-1]:
            del trajectory.step_records[jdx]
        return trajectory


class RemoveImperfectArrivals(TrajectoryProcessor):
    """Trajectory preprocessor that removes all trajectories where
    at least 1 train has not arrived at the destination."""

    @override(TrajectoryProcessor)
    def pre_process(self, trajectory: TrajectoryRecord) -> TrajectoryRecord:
        """Implementation of
        :class:`~maze.core.trajectory_recording.datasets.trajectory_processor.TrajectoryProcessor` interface."""

        assert not isinstance(trajectory, StateTrajectoryRecord), 'StateTrajectoryRecord not supported.'
        last_info = retrieve_done_info(trajectory)[-1]
        train_states = np.asarray(list(last_info['state'].values()))
        all_arrived = np.all(train_states == TrainState.DONE)
        if all_arrived:
            assert all(np.array(last_info['mcts_node_info--dist_to_target']) == 0)
        else:
            assert any(np.array(last_info['mcts_node_info--dist_to_target']) > 0)

        if not all_arrived:
            trajectory.step_records = []
        return trajectory
