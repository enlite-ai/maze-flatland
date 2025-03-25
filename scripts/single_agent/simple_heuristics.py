"""
Application of a greedy single agent for the flatland environment.
"""

from __future__ import annotations

from maze_flatland.agents.greedy_policy import GreedyPolicy
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.space_interfaces.action_conversion.directional import DirectionalAC
from maze_flatland.space_interfaces.observation_conversion.positional import PositionalObservationConversion
from maze_flatland.test.env_instantation import create_core_env

if __name__ == '__main__':
    core_env = create_core_env(
        1,
        35,
        35,
        3,
        1 / 100,
        {1: 1},
    )
    env = FlatlandEnvironment(
        core_env=core_env,
        action_conversion={'train_move': DirectionalAC()},
        observation_conversion={'train_move': PositionalObservationConversion(False)},
    )
    policy = GreedyPolicy()
    env.seed(1234)
    obs = env.reset()
    step: int = 0
    done: bool = False
    while not done:
        action = policy.compute_action(obs, actor_id=env.actor_id(), maze_state=env.get_maze_state())
        obs, rewards, done, info = env.step(action)
        step += 1

    maze_state = env.get_maze_state()
    train_agent = maze_state.trains[0]
    print(
        f'Train status is: {train_agent.status.name} - '
        f'Departed from {train_agent.initial_position}. '
        f'Arrived at {train_agent.position}. ' + f'Arrival time is {train_agent.arrival_time}'
    )
