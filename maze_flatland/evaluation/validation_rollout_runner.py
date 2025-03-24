"""Contains custom rollout runner for the flatland challenge that replicates
 the rounds undertaken during the challenge."""
from __future__ import annotations

import abc
import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.env.maze_env import MazeEnv
from maze.core.log_events.log_events_writer_registry import LogEventsWriterRegistry
from maze.core.log_events.log_events_writer_tsv import LogEventsWriterTSV
from maze.core.log_stats.event_decorators import define_episode_stats, define_epoch_stats
from maze.core.log_stats.log_stats import (
    LogStatsAggregator,
    LogStatsLevel,
    get_stats_logger,
    increment_log_step,
    register_log_stats_writer,
)
from maze.core.log_stats.log_stats_writer_console import LogStatsWriterConsole
from maze.core.rollout.rollout_runner import RolloutRunner
from maze.core.utils.factory import CollectionOfConfigType, ConfigType
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze_flatland.env import events
from maze_flatland.reward.default_flatland_v3 import ChallengeScore
from tqdm import tqdm


class ScoreTrackingEvents(abc.ABC):
    """
    Events to be triggered at the end of a rollout.
    """

    @define_epoch_stats(np.mean, output_name='avg_test_score')
    @define_episode_stats(max)
    def challenge_score(self, score: float):
        """
        Records the score for an episode.
        :param score: Normalised cumulative return in the range of [0,1].
        """


def load_round2_csv() -> pd.DataFrame:
    """Load round 2 dataset into a pandas dataframe.
    :return: dataframe with the aggergated seeds per level.
    """
    file_dir = os.path.dirname(__file__)
    df = pd.read_csv(f'{file_dir}/data/env_settings_round_2.csv', index_col=0)
    return _aggregate_seeds(df)


def _aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataset by aggregating the entries by their test_id value."""
    df['seeds'] = [np.array([]) for _ in range(len(df))]
    rnd_seeds = []
    for test_id in np.unique(df['test_id']):
        rnd_seeds.append(df.loc[df['test_id'] == test_id]['random_seed'].to_numpy())

    df.drop_duplicates('test_id', keep='first', inplace=True)
    df.drop(columns=['random_seed', 'env_id'], inplace=True)
    df['seeds'] = rnd_seeds
    return df


COLUMNS_NAME_KEY_DICT = {
    'test_id': None,
    'seed_used': None,
    'runtime': None,
    'score': None,
    'reward.count': (BaseEnvEvents.reward, 'count', None),
    'impossible_destination': (events.ScheduleEvents.impossible_dest, None, None),
    'count_steps_train_blocked': (events.TrainBlockEvents.train_blocked, None, None),
    'count_train_dead': (events.TrainBlockEvents.train_deadlocked, None, None),
    'deadlock_ratio': (events.TrainBlockEvents.count_deadlocks, None, None),
    'arrival_delay': (events.TrainMovementEvents.train_delay, None, None),
    'trains_arrived': (events.TrainMovementEvents.trains_arrived, None, None),
    'trains_arrived_possible': (events.TrainMovementEvents.trains_arrived_possible, None, None),
    'trains_cancelled': (events.TrainMovementEvents.trains_cancelled, None, None),
    'detour_delay': (events.TrainDetouringEvents.train_detouring, 'ep_sum_detour', None),
    'num_detours': (events.TrainDetouringEvents.train_detouring, 'ep_num_detours', None),
}

logger = logging.getLogger('ValidationRolloutRunner')
logger.setLevel(logging.INFO)


class ValidationRolloutRunner(RolloutRunner):
    """Custom sequential runner to run validate a policy on a set of challenge-like episodes.
       For round 1, we use the same setup as in round 2 but with a fixed speed of 1 for all the trains.
    :param deterministic: whether to sample the policy in a deterministic way
    :param round_level: Choice in [1, 2] based on the round to emulate.
    :param test_levels: if specified it defines the test_level or the test_levels to run from the validation set.
                        Otherwise, the full range of test levels will be evaluated.
    :param record_event_logs: Whether to record and dump the event log.
    :param time_limit: if given, it bounds the execution time to the given value.
    :param flat_step_limit: if given, it bounds the time for a flat step to the given value
                            despite the number of trains in the simulation.
    """

    def __init__(
        self,
        deterministic: bool,
        round_level: int,
        test_levels: list[int] | int | None,
        record_event_logs: bool,
        time_limit: Optional[int] = None,
        flat_step_limit: Optional[int] = None,
    ):
        self.progress_bar = None
        assert time_limit > 0 or time_limit is None
        self.time_limit = time_limit
        if test_levels is None:
            test_levels = np.arange(0, 15, 1)
        elif isinstance(test_levels, int):
            test_levels = [test_levels]
        assert round_level in [1, 2]
        self.round = round_level
        super().__init__(0, 0, deterministic, record_trajectory=False, record_event_logs=False)
        self.challenge_set = load_round2_csv()
        self.levels = test_levels
        self._n_seeds = 10
        self.epoch_stats = LogStatsAggregator(LogStatsLevel.EPOCH, get_stats_logger(f'validation round_{self.round}'))
        self.stats_df = pd.DataFrame(columns=COLUMNS_NAME_KEY_DICT.keys())
        self.simulation_time = 0.0
        self.flat_step_limit = flat_step_limit
        self.record_event_logs = record_event_logs

    def override_config(self, env: ConfigType, level: int) -> tuple[ConfigType, list[int]]:
        """Overrides the configuration defined in the config file with config from the selected level.
        :param env: Original config type from hydra.
        :param level: Identifier ot the test to run.
        :return: Tuple with updated env config and list of seeds.
        """
        config_dict_from_challenge = self.challenge_set.iloc[level].to_dict()
        assert config_dict_from_challenge['test_id'] == f'Test_{level}'

        assert isinstance(config_dict_from_challenge['speed_ratios'], str)
        # pylint: disable=eval-used
        speed_ratios = eval(config_dict_from_challenge['speed_ratios'])
        if self.round == 1:
            speed_ratios = {1: 1}  # Fix speed if 1st round

        env['_']['n_cities'] = config_dict_from_challenge['n_cities']
        env['_']['n_trains'] = config_dict_from_challenge['n_agents']
        env['_']['map_height'] = config_dict_from_challenge['y_dim']
        env['_']['map_width'] = config_dict_from_challenge['x_dim']
        env['_']['speed_ratio_map'] = speed_ratios

        # override malfunction generator
        env['core_env']['malfunction_generator']['parameters']['min_duration'] = config_dict_from_challenge[
            'malfunction_duration_min'
        ]
        env['core_env']['malfunction_generator']['parameters']['max_duration'] = config_dict_from_challenge[
            'malfunction_duration_max'
        ]
        malfunction_rate = 1 / config_dict_from_challenge['malfunction_interval']
        env['_']['malfunction_rate'] = malfunction_rate
        # override rail generator
        for k in ['grid_mode', 'max_rail_pairs_in_city', 'max_rails_between_cities']:
            env['core_env']['rail_generator'][k] = config_dict_from_challenge[k]

        return env, self.maze_seeding.get_explicit_env_seeds(self._n_seeds)

    @override(RolloutRunner)
    def run_with(self, env_config: ConfigType, wrappers: CollectionOfConfigType, agent: ConfigType) -> None:
        """Run the rollout sequentially in the main process.
        :param env_config: Plain configuration of the environment from the hydra file.
        :param wrappers: Collection of wrapper to be used
        :param agent: The config hydra setup for the agent
        """
        self.progress_bar = tqdm(desc='Episodes done', unit=' episodes', total=len(self.levels) * self._n_seeds)
        for test_id in self.levels:
            if self.record_event_logs:
                log_event_dir = f'./event_logs/round_{self.round}/test_{test_id}'
                LogEventsWriterRegistry.register_writer(LogEventsWriterTSV(log_dir=log_event_dir))
            env_config, seeds = self.override_config(env_config, test_id)
            env, policy = RolloutRunner.init_env_and_agent(
                env_config=env_config,
                wrappers_config=wrappers,
                max_episode_steps=self.max_episode_steps,
                agent_config=agent,
                input_dir=self.input_dir,
            )
            register_log_stats_writer(LogStatsWriterConsole())
            if not isinstance(env, LogStatsWrapper):
                env = LogStatsWrapper.wrap(env)

            rollout_time, time_limit_exceeded = self._evaluate(env, policy, seeds, test_id)
            self._summarise_results(test_id, rollout_time)

            # Dumps the stats to a csv file.
            self.stats_df.to_csv('./validation_stats.csv')
            if time_limit_exceeded:
                break
        self.progress_bar.close()
        self.plot_for_benchmarking()

    def plot_for_benchmarking(self) -> None:
        """Plot the most important metrics for copying to a benchmarking sheet"""
        name = os.path.abspath('').removeprefix(os.path.expanduser('~/'))
        summary = []
        rounds_considered = range(5)
        results = pd.DataFrame(columns=['score', 'trains_arrived_possible', 'runtime'])
        for test_id in map(lambda x: f'Test_{x}', rounds_considered):
            test_df = self.stats_df[self.stats_df.test_id == test_id]
            # if len(test_df) == 0:
            #     summary.append([0, 0, np.nan])
            #     continue
            # test_df = test_df.drop(columns=['test_id'])
            test_df = test_df[['score', 'trains_arrived_possible', 'runtime']]
            for i in range(len(test_df), 10):
                test_df = pd.concat(
                    [test_df, pd.DataFrame([dict(zip(test_df.columns, [0, 0, np.nan]))])], ignore_index=True
                )
            mean = test_df.mean()
            results = pd.concat([results, test_df], ignore_index=True)
            summary.append([mean.score, mean.trains_arrived_possible, mean.runtime])

        aggregated = results.mean().values
        output = [name] + sum(summary, []) + list(aggregated) + [results['runtime'].sum()]
        logger.info('output to copy to benchmarking sheet:')
        logger.info('\n\n' + '\t'.join(map(str, output)) + '\n\n' + '-' * 100)

    @classmethod
    def format_time(cls, unformatted_time: float) -> str:
        """Takes a float and returns the correct format for the time.
        :param unformatted_time: Float value of time in seconds.
        :return: formatted time as a string
        """
        seconds = int(unformatted_time % 60)
        hours = int(unformatted_time // 3600)
        minutes = int(unformatted_time // 60 - hours * 60)
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

    def _summarise_results(self, test_id: int, rollout_time: float) -> None:
        """Summarises the results over the same test_id.
        :param test_id: The index of the test to summarise the results for.
        :param rollout_time: The total time taken to complete the episode on the test_id.
        """
        env_settings = self.challenge_set.iloc[test_id]

        print(
            f'\n[Test_{test_id}] map_size: {env_settings.x_dim}x{env_settings.y_dim}'
            f', n_agents: {env_settings.n_agents}, number of rollout: {env_settings.n_envs_run},'
            f' elapsed time: {self.format_time(rollout_time)}'
        )
        increment_log_step()
        print('\n')

    @classmethod
    def get_df_stats_entry(cls, stats, level_id: int, seed: int, runtime: float, score: float) -> dict:
        """Converts raw stats into a dictionary compatible with a df.
        :param stats: the raw stats returned by the logstats wrapper
        :param level_id: the identifier of the test.
        :param seed: the seed used for the current run
        :param runtime: the run time of the current episode.
        :param score: the score for the current episode.
        :return: A dictionary with the relevant stats.
        """
        stats_entry = dict(zip(COLUMNS_NAME_KEY_DICT.keys(), [f'Test_{level_id}', seed, round(runtime, 3), score]))
        for k, v in COLUMNS_NAME_KEY_DICT.items():
            if v is not None:
                try:
                    stats_entry[k] = stats.get(v)
                except KeyError:
                    stats_entry[k] = 0
        return stats_entry

    def _evaluate(self, env: MazeEnv, policy: Policy, seeds: list[int], level_id: int) -> tuple[float, bool]:
        """Loops on the seeds to run the evaluation.
        :param env: the environment configured for the evaluation setup.
        :param policy: The policy to use for the evaluation.
        :param seeds: list of seeds to initialize the episodes
        :param level_id: Reference to the current test level.
        :return: A 2-element tuple with the time taken to run the evaluation rollouts and
                a flag indicating whether the time_limit is reached.
        """

        total_time = 0
        time_limit_exceeded = False
        score_event = self.epoch_stats.create_event_topic(ScoreTrackingEvents)
        policy_seeds = self.maze_seeding.get_explicit_agent_seeds(self._n_seeds)
        for i, seed in enumerate(seeds):
            t0 = time.time()
            score = np.round(self._run_rollout(env, policy, seed, policy_seeds[i]), 2)
            assert 0 <= score <= 1 or score == -1
            score_event.challenge_score(score=score)
            t1 = time.time()
            total_time += t1 - t0
            stats = env.get_stats(LogStatsLevel.EPISODE).reduce()
            self.stats_df.loc[len(self.stats_df)] = self.get_df_stats_entry(stats, level_id, seed, t1 - t0, score)
            self.epoch_stats.receive(stats)
            if self.time_limit is not None and self.simulation_time > self.time_limit:
                logger.info(
                    f'Early termination, time limit exceeded by '
                    f'{round(self.simulation_time - self.time_limit, 2)} seconds.',
                )
                time_limit_exceeded = True
                break
            # update the progress bar
            self.update_progress()
        return total_time, time_limit_exceeded

    def _run_rollout(self, env: MazeEnv, policy: Policy, env_seed: int, agent_seed: int) -> float:
        """Runs the rollout with the specified seed.
        :param env: The instance of the environment.
        :param policy: The policy to use to sample actions
        :param env_seed: The seed to initialise the environment
        :param agent_seed: The seed to initialise the policy
        :return: the score (defined in [0,1]) as per the flatland challenge 3.
        """
        _run_rollout_start_time = time.time()
        env.seed(env_seed)
        policy.seed(agent_seed)

        obs = env.reset()
        policy.reset()
        done = False
        score = 0
        start_time = time.time()
        while not done:
            action = policy.compute_action(
                observation=obs,
                actor_id=env.actor_id(),
                maze_state=env.get_maze_state() if policy.needs_state() else None,
                env=env if policy.needs_env() else None,
                deterministic=self.deterministic,
            )
            obs, rew, done, info = env.step(action)
            if env.is_flat_step():
                flat_step_time_taken = time.time() - start_time
                start_time = time.time()
                if self.flat_step_limit is not None and flat_step_time_taken > self.flat_step_limit:
                    score = -1  # penalty for timeout
                    logger.info(
                        f'Episode terminated early, flat step time limit exceeded by '
                        f'{round(flat_step_time_taken - self.flat_step_limit, 2)} seconds.',
                    )
                    break
                if (
                    self.time_limit is not None
                    and self.simulation_time + time.time() - _run_rollout_start_time > self.time_limit
                ):
                    break
        self.simulation_time += time.time() - _run_rollout_start_time
        if score != -1:
            score_instance = ChallengeScore()
            score = score_instance.to_scalar_reward(score_instance.summarize_reward(env.get_maze_state()))
        return score

    def update_progress(self):
        """Called on episode end to update a simple progress indicator."""
        self.progress_bar.update()
