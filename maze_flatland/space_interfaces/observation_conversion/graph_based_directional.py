"""
    Graph inspired representation.
"""

from __future__ import annotations

import pickle

import gym
import numpy as np
from maze.core.annotations import override
from maze.core.env.observation_conversion import ObservationType
from maze_flatland.env.core_env import FlatlandCoreEnvironment
from maze_flatland.env.maze_state import FlatlandMazeState
from maze_flatland.env.observation_builders.tree_switches import Node, TreeSwitchObsRailEnv, Vertex
from maze_flatland.env.prediction_builders.malf_shortest_path_predictor import MalfShortestPathPredictorForRailEnv
from maze_flatland.space_interfaces.observation_conversion.base import BaseObservationConversion

PLACEHOLDER = np.asarray([-1])
CACHED_DUMMY_OBS = {}  # used to reduce the computation time when creating a dummy obs.

POSSIBLE_DIRECTIONS = ['L', 'F', 'R']


def expand_obs_with_ms_information(maze_state: FlatlandMazeState, obs: dict) -> None:
    """Expand the current observation with substep information contained within the maze_state.
    :param maze_state: The current maze_state.
    :param obs: The current observation.
    """
    for prefix_dir, action_state in zip(
        ['L', 'F', 'R'], maze_state.trains[maze_state.current_train_id].actions_state.values()
    ):
        # add info about actions that are obstructed.
        if action_state.obstructed_by is not None:
            dir_obstructed = 1
        elif action_state.target_cell is None:
            dir_obstructed = -1
        else:
            dir_obstructed = 0
        obs[f'{prefix_dir}--obstructed'] = np.asarray([dir_obstructed])


class GraphDirectionalObservationConversion(BaseObservationConversion):
    """
    ObservationConversion filtering a subset of features from the encoded positional information.
    Based on Flatland's global observations.
    See :py:meth:`positional.ObservationConversion`.

    NOTE: distance and target_distance on consecutive edges might be redundant as we know already the distances and
            target_distances from the next next edges.

    :param graph_depth: Depth of the graph. Default is 2.

    """

    @override(BaseObservationConversion)
    def __init__(
        self,
        serialize_representation: bool,
        graph_depth: int = 2,
    ):
        super().__init__(serialize_representation)
        self.observation_builders[TreeSwitchObsRailEnv] = TreeSwitchObsRailEnv(
            graph_depth=graph_depth, predictor=MalfShortestPathPredictorForRailEnv(False, True, 20)
        )
        assert len(self.observation_builders) == 1
        self.map_size = (0, 0)
        # used for temporary storing the observation which is eventually used for masking.
        self.current_obs = None
        self._spaces = self._spaces
        self._observation_representation = self._observation_representation

    @override(BaseObservationConversion)
    def maze_to_space(self, maze_state: FlatlandMazeState) -> ObservationType:
        """
        See :py:meth:`base_observation_conversion.ObservationConversion.maze_to_space` and
        :py:meth: 'positional_observation_conversion.ObservationConversion.maze_to_space'.
        """
        if len(self._observation_representation) == 0 or maze_state.current_train_id == 0:
            self.update_obs_builder(maze_state)
        super().maze_to_space(maze_state)
        self.current_obs = self.pop_observation_representation()[TreeSwitchObsRailEnv][1]

        graph_dict = self.generate_observation(self.current_obs, maze_state.current_train_id)
        obs = self.get_train_obs(maze_state) | graph_dict
        expand_obs_with_ms_information(maze_state, obs)
        return {'observation': np.concatenate(list(obs.values())).astype(np.float32)}

    def space_to_maze(self, observation: ObservationType) -> FlatlandMazeState:
        """
        We do not provide the conversion of space observations to Flatland's global observations as of this time.
        See :py:meth:`base_observation_conversion.ObservationConversion.space_to_maze`.
        """
        raise NotImplementedError

    def update_obs_builder(self, maze_state: FlatlandMazeState) -> None:
        """Update the tree obs builder with deadlock status.
        :param maze_state: Maze state holding deadlock information.
        """

        self.observation_builders[TreeSwitchObsRailEnv].set_deadlock_trains(
            [train.deadlock for train in maze_state.trains]
        )

    @override(BaseObservationConversion)
    def reset(self, core_env: FlatlandCoreEnvironment) -> None:
        """Intercepts ~BaseObservationConversion.reset"""
        super().reset(core_env)
        self.map_size = (core_env.rail_env.height, core_env.rail_env.width)
        self._spaces = gym.spaces.Dict(
            {
                **self._spaces.spaces,
                'observation': gym.spaces.Box(shape=(86,), low=-1, high=np.prod(self.map_size), dtype=np.float32),
            }
        )

    def space(self) -> gym.spaces.Dict:
        """
        Check https://flatland.aicrowd.com/getting-started/env/observations.html for more info.
        See :py:meth:`positional.ObservationConversion.space`.
        """
        return self._spaces

    @classmethod
    def generate_observation(cls, current_vertex: Vertex, train_handle: int, prefix: str = '') -> dict:
        """Navigate the graph and generate the observation.
        :param current_vertex: The starting vertex of the directed graph.
        :param train_handle: Handle of the train to generate the observation for.
        :param prefix: Str prefix to be added to the current identifier.
        :return: Dictionary of the graph observation.
        """
        # ignore the current node and get to the children.
        current_vertex_dict = {}
        if isinstance(current_vertex, float):
            return cls.dummy_obs_for_fake_edge(prefix)
        if current_vertex.depth != 0:
            current_vertex_dict = cls.extract_info_from_node(current_vertex, current_vertex.depth, train_handle, prefix)

        if current_vertex.depth == 2:  # stop branching.
            return current_vertex_dict
        # continue expanding..
        for idx_dir, _dir in enumerate(POSSIBLE_DIRECTIONS):
            _child_prefix = prefix + f'-{_dir}' if prefix != '' else _dir
            branching_dict = cls.generate_observation(current_vertex.edges[idx_dir], train_handle, _child_prefix)
            assert len(set(current_vertex_dict).intersection(branching_dict)) == 0
            current_vertex_dict.update(branching_dict)
        return current_vertex_dict

    @classmethod
    def dummy_obs_for_fake_edge(cls, prefix: str) -> dict:
        """Create an observation for a fake edge leading to non-existing vertex and return it as a dictionary.

        :param prefix: Str prefix to be added to the keys of the dictionary.
        :return: A dictionary with the needed information for a fake vertex.
        """
        # cache dummy obs for SP, DET prefixes.
        if prefix in CACHED_DUMMY_OBS:
            node_info = CACHED_DUMMY_OBS[prefix]
        else:
            if len(prefix.replace('-', '')) >= 2:  # depth of 2
                return {
                    f'{prefix}--distance': PLACEHOLDER,
                    f'{prefix}--target_dist': PLACEHOLDER,
                    f'{prefix}--deadlock': PLACEHOLDER,
                }
            node_info = {
                f'{prefix}--other_train_dist': PLACEHOLDER,
                f'{prefix}--potential_conflict_dist': PLACEHOLDER,
                f'{prefix}--conflicting_train_distance': PLACEHOLDER,
                f'{prefix}--clashing_agent_has_alternative_path': PLACEHOLDER,
                f'{prefix}--clashing_agent_expected_delay_alternative_path': PLACEHOLDER,
                f'{prefix}--clashing_agent_has_moving_priority': PLACEHOLDER,
                f'{prefix}--can_detour': PLACEHOLDER,
                f'{prefix}--expected_delay_to_detour': PLACEHOLDER,
                f'{prefix}--unusable_switch_dist': PLACEHOLDER,
                f'{prefix}--switch_could_be_used_by_conflicting_agent': PLACEHOLDER,
                f'{prefix}--delay_for_taking_the_switch_for_ca': PLACEHOLDER,
                f'{prefix}--distance': PLACEHOLDER,
                f'{prefix}--target_dist': PLACEHOLDER,
                f'{prefix}--num_agents_same_dir': PLACEHOLDER,
                f'{prefix}--num_agents_opposite_dir': PLACEHOLDER,
                f'{prefix}--deadlock': PLACEHOLDER,
                f'{prefix}--slowest_speed': PLACEHOLDER,
            }
            for _dir in POSSIBLE_DIRECTIONS:
                node_info.update(cls.dummy_obs_for_fake_edge(f'{prefix}-{_dir}'))
            assert prefix not in CACHED_DUMMY_OBS
            CACHED_DUMMY_OBS[prefix] = node_info
        return node_info

    @classmethod
    def extract_info_from_node(
        cls, vertex: Vertex, depth: int, train_handle: int, prefix: str
    ) -> dict[str, np.ndarray]:
        """Extract information from the given vertex and return it as a dictionary.

        :param vertex: The vertex to analyse.
        :param depth:  The depth of the vertex to analyse.
        :param prefix: Str prefix to be added to the keys of the dictionary.
        :param train_handle: Handle of the train to compute the observation.
        :return: A dictionary with the needed information for the given vertex.
        """
        if depth == 1:
            return cls._extract_info_from_node_depth_1(vertex, train_handle, prefix)
        # depth > 1
        node_info = {}
        node: Node = vertex.node
        # distance to the next branch from root vertex.
        node_info[f'{prefix}--distance'] = np.asarray([np.nan_to_num(node.dist_to_next_branch, posinf=-1)])
        # this is replaced to -1 as it is impossible to get to the target. May be confused by the plain policy
        # to dead ends as we are using -1 there as well.
        node_info[f'{prefix}--target_dist'] = np.asarray([np.nan_to_num(node.dist_min_to_target, posinf=-1)])
        node_info[f'{prefix}--deadlock'] = np.asarray([len(node.idx_deadlock_trains) > 0])
        return node_info

    @classmethod
    def _extract_info_from_node_depth_1(cls, vertex: Vertex, train_handle: int, prefix: str) -> dict[str, np.ndarray]:
        """Extract information from a given consecutive vertex and return it as a dictionary.

        :param vertex: The vertex to analyse.
        :param prefix: Str prefix to be added to the keys of the dictionary.
        :param train_handle: Handle of the train to compute the observation.
        :return: A dictionary with the needed information for the given vertex.
        """
        node_info = {}
        node: Node = vertex.node
        slowest_speed_observed = 0

        if (node.num_agents_same_direction + node.num_agents_opposite_direction) > 0:
            slowest_speed_observed = node.speed_min_fractional

        node_info[f'{prefix}--other_train_dist'] = np.asarray(
            [np.nan_to_num(node.dist_other_agent_encountered, posinf=-1)]
        )
        node_info[f'{prefix}--potential_conflict_dist'] = np.asarray(
            [np.nan_to_num(node.dist_potential_conflict, posinf=-1)]
        )
        if node.idx_conflicting_agent > -1:
            node_info[f'{prefix}--conflicting_train_distance'] = np.asarray(
                [node.dist_potential_conflict + node.dist_other_agent_to_conflict]
            )
        else:
            node_info[f'{prefix}--conflicting_train_distance'] = PLACEHOLDER
        node_info[f'{prefix}--clashing_agent_has_alternative_path'] = np.asarray(
            [node.clashing_agent_has_alternative if node.clashing_agent_has_alternative is not None else -1]
        )
        node_info[f'{prefix}--clashing_agent_expected_delay_alternative_path'] = np.asarray(
            [node.ca_expected_delay_for_alternative_path]
        )
        # -1 if no conflict.
        node_info[f'{prefix}--clashing_agent_has_moving_priority'] = np.asarray(
            [
                node.idx_conflicting_agent < train_handle
                if node.idx_conflicting_agent >= 0
                else node.idx_conflicting_agent
            ]
        )
        node_info[f'{prefix}--can_detour'] = np.asarray(
            [
                node.conflict_on_junction_with_multiple_path
                if node.conflict_on_junction_with_multiple_path is not None
                else -1
            ]
        )
        node_info[f'{prefix}--expected_delay_to_detour'] = np.asarray([node.expected_delay_for_2nd_best_path])

        node_info[f'{prefix}--unusable_switch_dist'] = np.asarray([np.nan_to_num(node.dist_unusable_switch, posinf=-1)])
        node_info[f'{prefix}--switch_could_be_used_by_conflicting_agent'] = np.asarray(
            [node.unusable_switch_usable_for_ca]
        )
        node_info[f'{prefix}--delay_for_taking_the_switch_for_ca'] = np.asarray([node.unusable_switch_delay_for_ca])
        node_info[f'{prefix}--distance'] = np.asarray([np.nan_to_num(node.dist_to_next_branch, posinf=-1)])
        # distance to the next branch from root vertex.
        node_info[f'{prefix}--target_dist'] = np.asarray([np.nan_to_num(node.dist_min_to_target, posinf=-1)])
        node_info[f'{prefix}--num_agents_same_dir'] = np.asarray([node.num_agents_same_direction])
        node_info[f'{prefix}--num_agents_opposite_dir'] = np.asarray([node.num_agents_opposite_direction])
        node_info[f'{prefix}--deadlock'] = np.asarray([len(node.idx_deadlock_trains) > 0])
        node_info[f'{prefix}--slowest_speed'] = np.asarray([slowest_speed_observed])
        return node_info

    @classmethod
    def get_train_obs(cls, maze_state: FlatlandMazeState) -> ObservationType:
        """Returns the observation for a selected train.

        :param maze_state: the current maze state
        :return: the observation for the current agent.
        """
        train = maze_state.trains[maze_state.current_train_id]
        return {
            'train_status': np.asarray([train.status]),
            'in_transition': np.asarray([train.in_transition]),
            'timesteps_left': np.asarray([train.max_episode_steps - train.env_time]),
            'train_speed': np.asarray([train.speed]),
            'time_left_to_latest_arrival': np.asarray([train.time_left_to_scheduled_arrival]),
        }

    @classmethod
    def convert_to_dict(cls, obs: ObservationType) -> dict:
        """Converts the observation into a dictionary where each value has associated a key.

        :param obs: the observation.
        :return: the osbervation as a key-value structure."""
        assert 'observation' in obs
        keys = [
            'train_status',
            'in_transition',
            'timesteps_left',
            'train_speed',
            'time_left_to_latest_arrival',
            'L--other_train_dist',
            'L--potential_conflict_dist',
            'L--conflicting_train_distance',
            'L--clashing_agent_has_alternative_path',
            'L--clashing_agent_expected_delay_alternative_path',
            'L--clashing_agent_has_moving_priority',
            'L--can_detour',
            'L--expected_delay_to_detour',
            'L--unusable_switch_dist',
            'L--switch_could_be_used_by_conflicting_agent',
            'L--delay_for_taking_the_switch_for_ca',
            'L--distance',
            'L--target_dist',
            'L--num_agents_same_dir',
            'L--num_agents_opposite_dir',
            'L--deadlock',
            'L--slowest_speed',
            'L-L--distance',
            'L-L--target_dist',
            'L-L--deadlock',
            'L-F--distance',
            'L-F--target_dist',
            'L-F--deadlock',
            'L-R--distance',
            'L-R--target_dist',
            'L-R--deadlock',
            'F--other_train_dist',
            'F--potential_conflict_dist',
            'F--conflicting_train_distance',
            'F--clashing_agent_has_alternative_path',
            'F--clashing_agent_expected_delay_alternative_path',
            'F--clashing_agent_has_moving_priority',
            'F--can_detour',
            'F--expected_delay_to_detour',
            'F--unusable_switch_dist',
            'F--switch_could_be_used_by_conflicting_agent',
            'F--delay_for_taking_the_switch_for_ca',
            'F--distance',
            'F--target_dist',
            'F--num_agents_same_dir',
            'F--num_agents_opposite_dir',
            'F--deadlock',
            'F--slowest_speed',
            'F-L--distance',
            'F-L--target_dist',
            'F-L--deadlock',
            'F-F--distance',
            'F-F--target_dist',
            'F-F--deadlock',
            'F-R--distance',
            'F-R--target_dist',
            'F-R--deadlock',
            'R--other_train_dist',
            'R--potential_conflict_dist',
            'R--conflicting_train_distance',
            'R--clashing_agent_has_alternative_path',
            'R--clashing_agent_expected_delay_alternative_path',
            'R--clashing_agent_has_moving_priority',
            'R--can_detour',
            'R--expected_delay_to_detour',
            'R--unusable_switch_dist',
            'R--switch_could_be_used_by_conflicting_agent',
            'R--delay_for_taking_the_switch_for_ca',
            'R--distance',
            'R--target_dist',
            'R--num_agents_same_dir',
            'R--num_agents_opposite_dir',
            'R--deadlock',
            'R--slowest_speed',
            'R-L--distance',
            'R-L--target_dist',
            'R-L--deadlock',
            'R-F--distance',
            'R-F--target_dist',
            'R-F--deadlock',
            'R-R--distance',
            'R-R--target_dist',
            'R-R--deadlock',
            'L--obstructed',
            'F--obstructed',
            'R--obstructed',
        ]
        in_keys = list(obs.keys())
        key_value_obs = dict(zip(keys, obs['observation']))
        in_keys.remove('observation')
        for missing_key in in_keys:
            key_value_obs[missing_key] = obs[missing_key]
        return key_value_obs

    @override(BaseObservationConversion)
    def serialize_state(self) -> bytes | None:
        """Intercepts ~base.BaseObservationConversion.serialize_state."""
        if not self.serialize_representation:
            return None
        return pickle.dumps(self._observation_representation)

    @override(BaseObservationConversion)
    def deserialize_state(self, serialised_state: bytes):
        """Intercepts ~base.BaseObservationConversion.serialize_state."""
        self._observation_representation = []
        if self.serialize_representation:
            self._observation_representation = pickle.loads(serialised_state)
