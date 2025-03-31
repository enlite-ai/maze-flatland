"""
Greedy policy for Flatland environment.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np
from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID
from maze_flatland.env.maze_action import FlatlandMazeAction
from maze_flatland.env.maze_state import FlatlandMazeState


class GreedyPolicy(Policy):
    """
    Greedy policy for Flatland environment. Uses the shortest path for each individual train.

    Used to ensure that the observation space includes all properties necessary for the greedy policy.
    """

    def __init__(self):
        super().__init__()
        # Needs state if observation doesn't specify the shortest path information.
        self._needs_state = True

    @override(Policy)
    def needs_state(self) -> bool:
        """
        Implementation of :py:meth:`~maze.core.agent.policy.Policy.needs_state`.
        """

        return self._needs_state

    # pylint: disable=unused-argument
    @override(Policy)
    def compute_action(
        self,
        observation: ObservationType,
        maze_state: Optional[FlatlandMazeState],
        env: Optional[BaseEnv],
        actor_id: ActorID,
        deterministic: bool = False,
    ) -> ActionType:
        """
        Implementation of :py:meth:`~maze.core.agent.policy.Policy.compute_action`.
        Computes next action as the action that will generate the path with the shortest length. Note that this does not
        consider other agents in the environment, hence the resulting policy will be suboptimal.
        The returned action will always be one of FlatlandMazeAction.DEVIATE_LEFT, FlatlandMazeAction.GO_FORWARD,
        FlatlandMazeAction.DEVIATE_RIGHT, FlatlandMazeAction.DO_NOTHING.
        """
        assert actor_id is not None, 'ActorID must be given to compute the action.'
        step_key = actor_id.step_key
        mask_available = f'{step_key}_mask' in observation
        mask = observation[f'{step_key}_mask'] if mask_available else None
        # if there is mask with 1 single value, then return that value.
        if mask_available and sum(mask) == 1:
            return {step_key: np.where(mask == 1)[0][0]}

        assert maze_state is not None, 'FlatlandMazeState must be given.'
        train_id = int(maze_state.current_train_id)
        train = maze_state.trains[train_id]
        is_train_done = train.is_done()
        # train will not depart if it cannot arrive.
        if maze_state.trains[train_id].unsolvable:
            recommended_action = 0
        else:
            action_idx = np.asarray(
                list(action_state.goal_distance for action_state in train.actions_state.values())
            ).argmin()
            recommended_action = list(train.actions_state.keys())[action_idx]
        return {step_key: int(recommended_action) if not is_train_done else FlatlandMazeAction.STOP_MOVING.value}

    @override(Policy)
    def compute_top_action_candidates(
        self,
        observation: ObservationType,
        num_candidates: int,
        maze_state: Optional[FlatlandMazeState],
        env: Optional[BaseEnv],
        actor_id: ActorID = None,
        deterministic: bool = False,
    ) -> tuple[Sequence[ActionType], Sequence[float]]:
        """
        Implementation of :py:meth:`~maze.core.agent.policy.Policy.compute_top_candidates`.
        Computes probabilities as geodesic proximities to respective targets.
        """
        assert actor_id is not None, 'ActorID must be given to compute the action.'
        directions = np.asarray(
            [FlatlandMazeAction.DEVIATE_LEFT, FlatlandMazeAction.GO_FORWARD, FlatlandMazeAction.DEVIATE_RIGHT]
        )
        if not self._needs_state:
            assert 'current_train_id' in observation, (
                'flatland.agents.greedy_policy.GreedyPolicy requires information on the ID of the currently active '
                'train.'
            )
            train_id = int(observation['current_train_id'])
            # Possible shortest path directions in sequence in which they are provided by
            # ShortestPathObservationBuilder.
            # Proximities are computed as inverse distances.
            proximities = 1 / observation['shortest_path_distance_per_direction'][train_id]
        else:
            assert maze_state is not None, (
                "Observation doesn't provide shortest path information, hence " 'FlatlandMazeState must be specified.'
            )
            train_id = int(maze_state.current_train_id)
            train = maze_state.trains[train_id]
            proximities = 1 / np.nan_to_num(
                np.asarray([action_state.goal_distance for action_state in train.actions_state.values()]),
                # We replace infinite values with the max. path length + 1 to maintain consistency with
                # base.ObservationConversion.
                posinf=np.prod(maze_state.map_size) + 1,
            )

        # Compute scores as weighted proximities
        scores = np.nan_to_num(proximities / proximities.sum(), posinf=1)
        sort_idx = np.argsort(-scores)[:num_candidates]

        assert not np.isnan(scores).any(), proximities

        return [{actor_id.step_key: dirct for dirct in directions[sort_idx]}], scores[sort_idx]

    @override(Policy)
    def seed(self, seed: int) -> None:
        """
        GreedyPolicy is deterministic.
        """
