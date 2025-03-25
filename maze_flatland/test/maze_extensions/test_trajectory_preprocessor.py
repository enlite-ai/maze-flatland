"""File holdings tests for the trajectory pre-processor."""
from __future__ import annotations

import os
import pickle
import re
import subprocess

import numpy as np
from flatland.envs.step_utils.states import TrainState
from maze.core.trajectory_recording.records.trajectory_record import SpacesTrajectoryRecord, TrajectoryRecord
from maze_flatland.maze_extensions.trajectory_preprocessor import (
    FilterOnlyArrivedTrains,
    RemoveAllLostTrains,
    RemoveImperfectArrivals,
)


def run_rollout_and_collect_trajectories() -> str:
    """Run a greedy rollout and collect trajectories. It returns the output_path where there are the trajectories.

    :return: Path of the trajectories generated.
    """
    # run rollout and collect the trajectory.
    override = ['wrappers=[spaces_recording, sub_step_skipping_monitored]']
    result = subprocess.run(
        ['maze-run', '+experiment=multi_train/rollout/heuristic/simple_greedy'] + override,
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r'Output directory: (.+)', result.stdout)
    assert match, f'Could not find the output directory in the stdout of the rollout runner: {result.stdout}'
    # Extract the output directory path
    rollout_output_dir = match.group(1)
    print(f'Rollout directory: {rollout_output_dir}')

    spaces_record = os.path.join(rollout_output_dir, 'space_records/')
    pickle_names = os.listdir(spaces_record)
    output_path = str(spaces_record) + '/' + pickle_names[0]
    return output_path


def extract_actor_ids(traj: SpacesTrajectoryRecord | TrajectoryRecord) -> list[int]:
    """Extract actor ids from a trajectory.

    :param traj: Trajectory as SpacesTrajectoryRecord or TrajectoryRecord.
    :return: List of actor ids.
    """

    agent_ids = []
    for sr in traj.step_records:
        for ssr in sr.substep_records:
            agent_ids.append(ssr.agent_id)
    return agent_ids


def overwrite_train_arrival(
    record: SpacesTrajectoryRecord | TrajectoryRecord, idx_failing_trains: list[int] | int
) -> SpacesTrajectoryRecord | TrajectoryRecord:
    """Overwrites the trajectory to simulate that the trains specified have not arrived.

    :param record: Trajectory to be overwritten.
    :param idx_failing_trains: List of trains IDs to simulate that have not arrived.
    :return: The updated trajectory.
    """
    n_trains = len(record.step_records[-1].substep_records[-1].info['state'])
    if isinstance(idx_failing_trains, int):
        idx_failing_trains = [idx_failing_trains]
    dist_to_target = np.zeros(n_trains)
    # set distance of 3 to target
    for idx in idx_failing_trains:
        dist_to_target[idx] = 3
        record.step_records[-1].substep_records[-1].info['state'][idx] = TrainState.MOVING
        # Set that the train is at a distance of 3 cells from its target.
    record.step_records[-1].substep_records[-1].info['mcts_node_info--dist_to_target'] = dist_to_target
    return record


def test_filter_only_arrived_trains():
    """Test that :trajectory_preprocessor.FilterOnlyArrivedTrains successfully
    removes only the substep records related to the selected trains.
    """

    output_path = run_rollout_and_collect_trajectories()

    with open(output_path, 'rb') as f:
        record = pickle.load(f)

    # overwrite the train with id 0 to 'not_arrived'
    idx_failing_train = 0
    overwrite_train_arrival(record, idx_failing_train)

    # get the ids of the original trajectory
    original_agent_ids = extract_actor_ids(record)
    # apply filter
    record = FilterOnlyArrivedTrains().pre_process(record)
    # get ids of the trajectory processed.
    refined_agent_ids = extract_actor_ids(record)

    offset = 0
    for idx, agent_id in enumerate(original_agent_ids):
        if agent_id == idx_failing_train:
            offset += 1
            continue
        assert agent_id == refined_agent_ids[idx - offset]
    assert idx_failing_train not in refined_agent_ids


def test_remove_imperfect_arrivals():
    """Test that :trajectory_preprocessor.RemoveImperfectArrivals successfully
    drops the trajectory records where not all trains have arrived.
    """

    output_path = run_rollout_and_collect_trajectories()

    with open(output_path, 'rb') as f:
        record = pickle.load(f)

    len_original_record = len(record)
    # Apply filtering.
    record = RemoveImperfectArrivals().pre_process(record)
    # Check that the trajectory remains the same
    assert len(record) == len_original_record

    # overwrite the train with id 0 to 'not_arrived'
    overwrite_train_arrival(record, 0)
    # Apply filtering.
    record = RemoveImperfectArrivals().pre_process(record)
    # Check that the full trajectory has been removed.
    assert len(record) == 0


def test_remove_all_lost_trains():
    """Test that :trajectory_preprocessor.RemoveAllLostTrains successfully
    drops the trajectory records where no trains have arrived.
    """

    output_path = run_rollout_and_collect_trajectories()
    with open(output_path, 'rb') as f:
        record = pickle.load(f)
    len_original_record = len(record)

    # overwrite the train with id 0 to 'not_arrived'
    overwrite_train_arrival(record, 0)
    # Apply filtering.
    record = RemoveAllLostTrains().pre_process(record)
    # Check that the full trajectory is unchanged.
    assert len(record) == len_original_record

    # overwrite all trains as 'not_arrived'
    overwrite_train_arrival(record, [0, 1, 2])
    # Apply filtering.
    record = RemoveAllLostTrains().pre_process(record)
    # Check that the full trajectory is now removed.
    assert len(record) == 0
