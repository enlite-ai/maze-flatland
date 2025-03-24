"""
Tests greedy policy agent for FlatlandEnvironment.
"""

from __future__ import annotations

import gym
import numpy as np
from maze.core.annotations import override
from maze.core.env.observation_conversion import ObservationConversionInterface, ObservationType
from maze_flatland.agents.greedy_policy import GreedyPolicy
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.env.maze_state import FlatlandMazeState
from maze_flatland.space_interfaces.observation_conversion.positional import PositionalObservationConversion
from maze_flatland.test.env_instantation import create_env_for_testing


def test_greedy_policy_is_successful():
    """
    Tests greedy policy in scenario in which it is successful.
    """

    env: FlatlandEnvironment = create_env_for_testing()
    env.seed(9999)
    obs = env.reset()
    policy = GreedyPolicy()
    step: int = 0
    max_n_steps = 305
    done: bool = False
    while not done and step < max_n_steps:
        action = policy.compute_action(obs, maze_state=env.get_maze_state(), actor_id=env.actor_id())
        obs, rewards, done, info = env.step(action)
        step += 1
    assert step == 304
    assert done


def test_greedy_policy_equivalency():
    """
    Tests equivalency of greedy policy's results when provided information by observations or maze states.
    """

    class NoShortestPathObsConv(PositionalObservationConversion):
        """
        Obs. conv. without information on shortest paths.
        """

        @override(PositionalObservationConversion)
        def space_to_maze(self, observation: ObservationType) -> FlatlandMazeState:
            """
            See :py:meth:`~maze.core.env.observation_conversion.ObservationConversionInterface.space_to_maze`.
            """
            raise NotImplementedError

        @override(ObservationConversionInterface)
        def maze_to_space(self, _maze_state: FlatlandMazeState) -> ObservationType:
            """
            Removes shortest path info from observation.
            """

            return {key: val for key, val in super().maze_to_space(_maze_state) if not key.startswith('shortest_path')}

        @override(PositionalObservationConversion)
        def space(self) -> gym.spaces.Dict:
            """
            Removes shortest path info from observation space.
            """

            return gym.spaces.Dict(
                {
                    key: subspace
                    for key, subspace in super().space().spaces.items()
                    if not key.startswith('shortest_path')
                }
            )

    env: FlatlandEnvironment = create_env_for_testing()
    obs = env.reset()
    policy_obs = GreedyPolicy()
    _obs_conv_no_short_path = NoShortestPathObsConv(
        False,
    )
    _obs_conv_no_short_path.reset(env.core_env)
    policy_state = GreedyPolicy()

    step: int = 0
    max_n_steps = 100
    done: bool = False

    while not done and step < max_n_steps:
        maze_state = env.get_maze_state()

        # Compare recommended actions.
        action_obs = policy_obs.compute_action(obs, maze_state=maze_state, actor_id=env.actor_id())
        action_state = policy_state.compute_action(obs, maze_state=maze_state, actor_id=env.actor_id())
        assert action_obs == action_state

        # Compare recommended top action candidates.
        actions_obs, scores_obs = policy_obs.compute_top_action_candidates(
            obs, num_candidates=3, maze_state=maze_state, actor_id=env.actor_id()
        )
        actions_state, scores_state = policy_state.compute_top_action_candidates(
            obs, num_candidates=3, maze_state=maze_state, actor_id=env.actor_id()
        )

        assert actions_obs == actions_state
        assert np.allclose(scores_obs, scores_state)

        obs, rewards, done, info = env.step(action_obs)
        step += 1
