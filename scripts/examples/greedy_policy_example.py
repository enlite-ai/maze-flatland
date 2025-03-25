"""
Example of application of a greedy policy for the Flatland environment.
"""

from __future__ import annotations

from maze.core.trajectory_recording.utils.monitoring_setup import MonitoringSetup
from maze_flatland.agents.greedy_policy import GreedyPolicy
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.test.env_instantation import create_env_for_testing


def greedy_policy_example() -> None:
    """
    Demonstrates execution of greedy policy for Flatland environment.
    """

    env: FlatlandEnvironment = create_env_for_testing()
    policy = GreedyPolicy()
    with MonitoringSetup(env) as menv:
        obs = menv.reset()
        total_reward: int = 0
        step: int = 0
        max_n_steps = 5000
        done: bool = False

        while not done and step < max_n_steps:
            action = policy.compute_action(obs, actor_id=env.actor_id(), maze_state=env.get_maze_state())
            obs, rewards, done, info = menv.step(action)
            total_reward += rewards.sum()

            if (step + 1) % 4 == 0:
                print(f'== Round {step} ==\n')
                print('  Reward:', rewards)
                print('\n')

                menv.render(
                    rail_env=menv.rail_env,
                    close_prior_windows=True,
                )

            step += 1

    print('Avg. reward:', total_reward / step)
    print('Total reward:', total_reward)
    print('Done:', done)


if __name__ == '__main__':
    greedy_policy_example()
