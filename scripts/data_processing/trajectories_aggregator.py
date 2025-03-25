"""File holding utils to aggregate trajectories."""
from __future__ import annotations

import argparse
import os
import pickle
import re

from maze.core.trajectory_recording.records.trajectory_record import SpacesTrajectoryRecord


def get_n_trains_and_traj_id(filename: str) -> tuple[int, str]:
    """Extract number of trains and trajectory id from a filename.
    :param filename: filename to extract trajectories from
    :return: Tuple with number of trains and trajectory id.
    """
    search_n_trains = re.search(r'_(\d+)_', filename)
    search_traj_name = re.search(r'--(.*).pkl', filename)
    assert search_n_trains is not None and search_traj_name is not None

    return search_n_trains.group(1), search_traj_name.group(1)


def aggregate_trajectories_by_n_trains(home_dir: str, out_dir: str):
    """Search the home dir and aggregates the trajectories by number of trains.
    :param home_dir: directory to look for trajectories.
    :param out_dir: directory to save aggregated trajectories.
    """
    trajectories_names_by_n_trains = {}
    for fname in os.listdir(home_dir):
        if not (fname.endswith('.pkl') and fname.startswith('trajectories')):
            continue
        n_trains, traj_id = get_n_trains_and_traj_id(fname)
        if n_trains not in trajectories_names_by_n_trains:
            trajectories_names_by_n_trains[n_trains] = {}

        with open(os.path.join(home_dir, fname), 'rb') as f:
            assert traj_id not in trajectories_names_by_n_trains[n_trains]
            loaded_trajectory = pickle.load(f)
            if isinstance(loaded_trajectory, dict):
                n_duplicates = len(
                    set(list(loaded_trajectory.keys())).intersection(trajectories_names_by_n_trains[n_trains])
                )
                assert n_duplicates == 0, f'Found {n_duplicates} duplicate trajectories in {fname}'
                trajectories_names_by_n_trains[n_trains].update(loaded_trajectory)
            elif isinstance(loaded_trajectory, SpacesTrajectoryRecord):
                trajectories_names_by_n_trains[n_trains][traj_id] = loaded_trajectory
            else:
                assert False, f'File {fname} with type {type(loaded_trajectory)} is not supported.'
    os.makedirs(out_dir, exist_ok=True)
    for n_trains, trajectories in trajectories_names_by_n_trains.items():
        aggregated_fname = f'trajectories_{n_trains}_trains--aggregated_{len(trajectories)}.pkl'
        with open(os.path.join(out_dir, aggregated_fname), 'wb') as f:
            pickle.dump(trajectories, f)
        print(f'[Dumped] {os.path.join(out_dir, aggregated_fname)}')


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    input_dir = args.input_dir
    assert os.path.isdir(input_dir)
    output_dir = os.path.join(input_dir, 'aggregated/')
    aggregate_trajectories_by_n_trains(input_dir, output_dir)
