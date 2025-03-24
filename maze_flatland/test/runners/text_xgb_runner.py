"""File holdings tests for XGB"""

from __future__ import annotations

import os
import re
import subprocess


def run_greedy_rollout(override: list[str], capture_output: bool = True) -> str | None:
    """Run greedy rollout for one episode"""

    result = subprocess.run(
        ['maze-run', '+experiment=multi_train/rollout/heuristic/simple_greedy'] + override,
        capture_output=capture_output,
        text=True,
        check=True,
    )

    if capture_output:
        match = re.search(r'Output directory: (.+)', result.stdout)

        if match:
            # Extract the output directory path
            rollout_output_dir = match.group(1)
            print(f'Rollout directory: {rollout_output_dir}')
        else:
            raise ValueError(
                f'Could not find the output directory in the stdout of the rollout runner: {result.stdout}'
            )
        return rollout_output_dir

    return None


def test_replay_rollout_runner_hydra():
    """Test the XGB training runner."""
    n_rollouts = 2

    input_path = run_greedy_rollout([f'runner.n_episodes={n_rollouts}', 'policy=random_policy'])

    output_path = run_greedy_rollout(
        ['runner=replay_rollout_runner', f'runner.input_dirs={input_path}', '~runner.n_episodes']
    )
    trajectory_folder = output_path + '/space_records'
    assert os.path.exists(trajectory_folder)
    assert len(os.listdir(trajectory_folder)) == n_rollouts

    overrides = ['+experiment=offline/train/xgboost_train', f'trajectories_data={trajectory_folder}']
    result = subprocess.run(['maze-run', '-cn', 'conf_train'] + overrides, capture_output=True, text=True, check=True)
    match = re.search(r'Model saved @ (.+)', result.stdout)
    assert match
    trained_policy_dir = match.group(1)
    _test_xgb_policy(trained_policy_dir)


def _test_xgb_policy(folder_path: str):
    """Test a XGB trained policy."""

    subprocess.run(
        [
            'maze-run',
            '+experiment=multi_train/rollout/xgboost',
            'env._.n_trains=3',
            'runner.n_episodes=1',
            'runner.n_processes=1',
            f'input_dir={folder_path}',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
