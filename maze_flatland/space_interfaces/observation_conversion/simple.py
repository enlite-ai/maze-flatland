"""
Base class for ObservationConversion.
"""
from __future__ import annotations

import flatland
import flatland.core.grid.grid4
import flatland.envs.agent_utils
import flatland.envs.line_generators
import flatland.envs.observations
import flatland.envs.rail_env
import gym
import numpy as np
from maze.core.annotations import override
from maze.core.env.observation_conversion import ObservationType
from maze_flatland.env.core_env import FlatlandCoreEnvironment
from maze_flatland.env.maze_action import FlatlandMazeAction
from maze_flatland.env.maze_state import FlatlandMazeState
from maze_flatland.space_interfaces.observation_conversion.base import BaseObservationConversion


class SimpleObservationConversion(BaseObservationConversion):
    """
    Base ObservationConversion for (multi-agent) Flatland environment.
    Incorporates a subset of information directly accessible via FlatlandMazeState that we want to ensure to be
    available in every ObservationConversion. This includes mostly primitive information (about trains'
    positions, targets, malfunctions, ...).
    """

    _map_height: int
    _highest_speed: float
    _max_duration: int
    _lowest_speed: float
    _highest_speed: float
    _n_trains: int
    _map_width: int

    def __init__(self, serialize_representation: bool):
        super().__init__(serialize_representation)
        self._spaces = self._spaces

    @override(BaseObservationConversion)
    def reset(self, core_env: FlatlandCoreEnvironment) -> None:
        """Intercepts ~BaseObservationConversion.reset"""
        super().reset(core_env)
        self._n_trains = core_env.n_trains
        self._map_height = core_env.rail_env.height
        self._map_width = core_env.rail_env.width
        all_speeds = [a.speed_counter.speed for a in core_env.rail_env.agents]
        self._lowest_speed = min(all_speeds)
        self._highest_speed = max(all_speeds)
        self._max_duration = core_env.rail_env.malfunction_generator.MFP.max_duration

        n_status = len(flatland.envs.agent_utils.TrainState)

        self._spaces = gym.spaces.Dict(
            {
                **self._spaces.spaces,
                # 0 if train has not yet started the transition between current and subsequent cell.
                'trains_in_transition': gym.spaces.Box(shape=(self._n_trains,), low=0, high=1, dtype=np.int32),
                # Map size. Can be arbitrary.
                'map_size': gym.spaces.Box(shape=(2,), low=0, high=np.iinfo(np.int32).max, dtype=np.int32),
                # ID of currently active train as discrete value and one-hot encoded.
                'current_train_id': gym.spaces.Box(shape=(1,), low=0, high=self._n_trains - 1, dtype=np.int32),
                'current_train_id_oh': gym.spaces.Box(shape=(self._n_trains,), low=0, high=1, dtype=np.int32),
                # Train targets.
                'target_x': gym.spaces.Box(shape=(self._n_trains,), low=0, high=self._map_width - 1, dtype=np.int32),
                'target_y': gym.spaces.Box(shape=(self._n_trains,), low=0, high=self._map_height - 1, dtype=np.int32),
                # Geodesic distances to target.
                'target_distances': gym.spaces.Box(
                    low=0, high=self._map_width * self._map_height, shape=(self._n_trains,), dtype=np.int32
                ),
                # Train positions.
                'position_x': gym.spaces.Box(shape=(self._n_trains,), low=0, high=self._map_width - 1, dtype=np.int32),
                'position_y': gym.spaces.Box(shape=(self._n_trains,), low=0, high=self._map_height - 1, dtype=np.int32),
                # Shortest path distances per direction left/forward/right.
                # The longest theoretically possible path is given by traversing each cell once.
                # Nonsensical paths (i. e. for left branch when no left branch is available)
                # are represented with max. possible path length + 1.
                'shortest_path_distance_per_direction': gym.spaces.Box(
                    shape=(self._n_trains, 3), low=0, high=self._map_width * self._map_height + 1, dtype=np.int32
                ),
                # Index for action for direction with shortest path.
                # 0 -> DEVIATE_LEFT, 1 -> GO_FORWARD, 2 -> DEVIATE_RIGHT.
                # Note: Complete shortest path is currently not included in observation.
                'shortest_path_direction_index': gym.spaces.Box(shape=(self._n_trains,), low=0, high=2, dtype=np.int32),
                # Reflects whether action is required. If agent is in current step _within_ a cell due to the train's
                # fractional speed, this is not the case and all issued action instructions are ignored until the agent
                # leaves the current cell. See
                # https://flatland.aicrowd.com/getting-started/env/speed_profiles.html#actions-and-observation-with-different-speed-levels.
                # Indicating which trains are malfunctioning.
                'malfunctions': gym.spaces.Box(shape=(self._n_trains,), low=0, high=self._max_duration, dtype=np.int32),
                # Train direction.
                'directions': gym.spaces.Box(shape=(self._n_trains,), low=0, high=3, dtype=np.int32),
                # Train status.
                'status': gym.spaces.Box(shape=(self._n_trains,), low=0, high=n_status, dtype=np.int32),
                # Current train speeds.
                'speeds': gym.spaces.Box(
                    shape=(self._n_trains,), low=self._lowest_speed, high=self._highest_speed, dtype=np.float32
                ),
                # Last modifying actions (i.e. without DO_NOTHING) per train.
                'last_modifying_actions': gym.spaces.Box(
                    shape=(self._n_trains,), low=0, high=len(FlatlandMazeAction) - 1, dtype=np.int32
                ),
                # Train blocks.
                'train_blocks': gym.spaces.Box(shape=(self._n_trains, self._n_trains), low=0, high=1, dtype=np.int32),
            }
        )

    @override(BaseObservationConversion)
    def maze_to_space(self, maze_state: FlatlandMazeState) -> ObservationType:
        """
        See :py:meth:`~maze.core.env.observation_conversion.ObservationConversionInterface.maze_to_space`.
        """
        super().maze_to_space(maze_state)
        # Fetch Flatland's global observation (see
        # https://flatland.aicrowd.com/getting-started/env/observations.html#global-observation) and the current RailEnv
        # instance.
        train_ids = list(range(self._n_trains))
        branch_directions = (
            FlatlandMazeAction.DEVIATE_LEFT,
            FlatlandMazeAction.GO_FORWARD,
            FlatlandMazeAction.DEVIATE_RIGHT,
        )
        trains_targets = [train.target for train in maze_state.trains]
        trains_positions = [train.position for train in maze_state.trains]
        train_blocks = np.zeros((maze_state.n_trains, maze_state.n_trains))
        for train in maze_state.trains:
            if not train.is_block:
                continue
            for action_state in train.actions_state.values():
                if action_state.is_safe():
                    train_blocks[train.handle][action_state.obstructed_by] = 1
        space_dict = {
            'trains_in_transition': np.asarray([train.in_transition for train in maze_state.trains]).astype(np.int32),
            'map_size': np.asarray(maze_state.map_size).astype(np.int32),
            'current_train_id': np.expand_dims(np.asarray(maze_state.current_train_id), 0).astype(np.int32),
            'current_train_id_oh': np.asarray([tid == maze_state.current_train_id for tid in train_ids]).astype(
                np.int32
            ),
            'target_x': np.asarray([tt[0] for tt in trains_targets]).astype(np.int32),
            'target_y': np.asarray([tt[1] for tt in trains_targets]).astype(np.int32),
            'target_distances': np.nan_to_num(
                np.asarray([train.target_distance for train in maze_state.trains], dtype=np.float32),
                posinf=maze_state.map_size[0] * maze_state.map_size[1] + 1,
            ).astype(np.int32),
            'position_x': np.asarray([tp[0] for tp in trains_positions]).astype(np.int32),
            'position_y': np.asarray([tp[1] for tp in trains_positions]).astype(np.int32),
            'shortest_path_distance_per_direction': np.nan_to_num(
                np.asarray(
                    [
                        [train.actions_state[act].goal_distance for act in branch_directions]
                        for train in maze_state.trains
                    ]
                ),
                posinf=self._map_width * self._map_height + 1,
            ).astype(np.int32),
            'shortest_path_direction_index': np.asarray(
                [
                    # Adjust action value to index within space boundaries.
                    np.argmin(list(action_state.goal_distance for action_state in train.actions_state.values()))
                    for train in maze_state.trains
                ]
            ).astype(np.int32),
            'malfunctions': np.asarray([train.malfunction_time_left for train in maze_state.trains]).astype(np.int32),
            'directions': np.asarray([train.direction for train in maze_state.trains]).astype(np.int32),
            'status': np.asarray([train.status.value for train in maze_state.trains]).astype(np.int32),
            'speeds': np.asarray([train.speed for train in maze_state.trains], dtype=np.float32),
            'last_modifying_actions': np.asarray([train.last_action.value for train in maze_state.trains]).astype(
                np.int32
            ),
            'train_blocks': np.asarray(train_blocks, dtype=np.int32),
        }

        return space_dict

    @override(BaseObservationConversion)
    def space(self) -> gym.spaces.Dict:
        """
        Check https://flatland.aicrowd.com/getting-started/env/observations.html for more info.
        See :py:meth:`~maze.core.env.observation_conversion.ObservationConversionInterface.space`.
        """

        # Ensure that agent directions and status options are numbered sequentially starting with 0.
        assert all(
            ({entry.value for entry in enum} == set(range(0, len(enum))))
            for enum in (flatland.core.grid.grid4.Grid4TransitionsEnum, flatland.envs.agent_utils.TrainState)
        ), 'Enum options are not sequentially numbered. Unclear how to specify observation space.'

        return self._spaces
