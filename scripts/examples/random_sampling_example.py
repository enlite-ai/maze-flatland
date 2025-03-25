"""
Example for random sampling of actions for Flatland environment.
"""

from __future__ import annotations

import copy

from maze.core.agent.random_policy import MaskedRandomPolicy
from maze.core.trajectory_recording.utils.monitoring_setup import MonitoringSetup
from maze_flatland.env.masking.mask_builder import LogicMaskBuilder
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.test.env_instantation import create_env_for_testing
from maze_flatland.wrappers.masking_wrapper import FlatlandMaskingWrapper


def random_policy_example() -> None:
    """
    Demonstrates execution of random policy for Flatland environment.
    """

    env: FlatlandEnvironment = create_env_for_testing()
    env = FlatlandMaskingWrapper.wrap(env, mask_builder=LogicMaskBuilder())
    rnd_policy = MaskedRandomPolicy(copy.deepcopy(env.action_spaces_dict))
    with MonitoringSetup(env, log_dir='/tmp') as menv:
        obs = menv.reset()
        total_reward: int = 0
        step: int = 0
        max_n_steps = 1000
        done: bool = False

        while not done and step < max_n_steps:
            action = rnd_policy.compute_action(obs, maze_state=None)
            obs, rewards, done, info = menv.step(action)
            total_reward += rewards.sum()

            if (step + 1) % 4 == 0:
                print(f'== Round {step} ==\n')
                print('  Reward:', rewards)
                print('\n')
            step += 1

    print('Avg. reward =', total_reward / step)


if __name__ == '__main__':
    random_policy_example()
