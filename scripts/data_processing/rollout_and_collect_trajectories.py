"""File holding the scripts to automatise the run and collection of trajectories."""
from __future__ import annotations

import argparse
import os
from datetime import datetime

from hydra import compose, initialize_config_module
from maze.maze_cli import maze_run
from maze.utils.bcolors import BColors


def run_hydra_config_with_overrides(overrides: list[str], working_dir: str) -> None:
    """Read assemble and run a hydra config, given the config module, name, and overrides.

    :param overrides: Overrides as kwargs, e.g. env="cartpole", configuration="test"
    :param working_dir: The working directory to run the experiment.
    : return: None.
    """
    with initialize_config_module('maze.conf'):
        cfg = compose('conf_rollout', overrides=overrides)
    cfg['wrappers']['maze.core.wrappers.monitoring_wrapper.MazeEnvMonitoringWrapper']['action_logging'] = True
    cfg['wrappers']['maze.core.wrappers.monitoring_wrapper.MazeEnvMonitoringWrapper']['reward_logging'] = True
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    os.chdir(working_dir)
    maze_run(cfg)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=31415)
parser.add_argument('--n_episodes', type=int, default=100)
parser.add_argument('--n_trains', type=int)
parser.add_argument('--policy_dir', type=str)
parser.add_argument('--use-challenge-setup', type=bool, default=False)
parser.add_argument('--n_simulations', type=int, default=2)

if __name__ == '__main__':
    args = parser.parse_args()
    use_challenge_setup = args.use_challenge_setup
    if use_challenge_setup:
        alert_msg = 'Fractional speeds and malfunctions are not yet supported.\nOnly the challenge reward is used. '
        BColors.print_colored(alert_msg, BColors.WARNING)
    seed = args.seed
    n_trains = args.n_trains
    n_processes = 30
    n_episodes = args.n_episodes
    non_lin = 'torch.nn.Tanh'
    n_simulations = args.n_simulations
    hidden_units = [512, 256]
    malfunction_rate_base = 0 if not use_challenge_setup else 90
    malfunction_duration_min = 10
    malfunction_duration_max = 30
    n_cities = 2
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    data_output_dir = os.path.abspath('.') + f'/flatland_offline_dataset/collected/{timestamp}'
    exp_output_dir = os.path.abspath('.') + f'/flatland_offline_dataset/tmp_exp_results/{timestamp}'
    input_dir = args.policy_dir
    assert os.path.isdir(input_dir)
    rollout_dir = exp_output_dir + f'_{n_trains}_rollout'
    malfunction_rate = 0  # if malfunction_rate_base == 0 else 1 / (n_trains * malfunction_rate_base)
    conf_rollout = [
        '+experiment=multi_train/rollout/az_mcts_gumbel',
        f'env._.n_trains={str(n_trains)}',
        f'model.policy.networks.train_move.non_lin={str(non_lin)}',
        f'model.policy.networks.train_move.hidden_units={str(hidden_units)}',
        f'input_dir={input_dir}',
        f'policy.n_simulations={n_simulations}',
        f'runner.n_processes={n_processes}',
        f'env._.malfunction_rate={malfunction_rate}',
        f'env._.malfunction_duration_min={malfunction_duration_min}',
        f'env._.malfunction_duration_max={malfunction_duration_max}',
        f'env._.n_cities={n_cities}',
        f'runner.n_episodes={n_episodes}',
    ]
    if use_challenge_setup:
        # conf_rollout.append('env._.speed_ratio_map={1.0: 0.25, 0.5: 0.25, 0.33: 0.25, 0.25: 0.25}')
        conf_rollout.append(
            'env.core_env.reward_aggregator={_target_:maze_flatland.reward.default_flatland_v3.ChallengeScore}'
        )
    # run the rollout with the policy specified.
    run_hydra_config_with_overrides(conf_rollout, rollout_dir)

    # now we need to "replay" the rollouts.
    trajectory_recording_rollout = [
        '+experiment=multi_train/rollout/heuristic/replay_rollout',
        f'runner.input_dirs={rollout_dir}',
    ]
    run_hydra_config_with_overrides(trajectory_recording_rollout, data_output_dir)
