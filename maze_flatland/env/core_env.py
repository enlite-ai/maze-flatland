"""
CoreEnv for Flatland environment (https://flatland.aicrowd.com/intro.html).
Write-up with ideas for improvements:
https://blog.netcetera.com/leverage-reinforcement-learning-for-building-intelligent-trains-flatland-neurips-challenge-2020-6cc8882f7700
"""
from __future__ import annotations

import pickle
import random
import sys
from typing import Any, Optional, Union

import flatland
import flatland.envs.agent_utils
import flatland.envs.line_generators
import flatland.envs.malfunction_generators
import flatland.envs.rail_env
import flatland.envs.rail_generators
import numpy as np
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.step_utils.states import TrainState
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.core_env import CoreEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.structured_env import ActorID
from maze.core.events.pubsub import Pubsub
from maze.core.rendering.renderer import Renderer
from maze.core.utils.factory import ConfigType, Factory
from maze_flatland.env.backend_utils import get_transitions_map
from maze_flatland.env.events import (
    FlatlandDepartingEvents,
    FlatlandExecutionEvents,
    ScheduleEvents,
    SeedingEvent,
    TrainBlockEvents,
    TrainDetouringEvents,
    TrainMovementEvents,
)
from maze_flatland.env.kpi_calculator import FlatlandKPICalculator
from maze_flatland.env.maze_action import FlatlandMazeAction
from maze_flatland.env.maze_state import FlatlandMazeState, MazeTrainState
from maze_flatland.env.renderer import FlatlandRendererBase
from maze_flatland.env.termination_condition import BaseEarlyTermination, EarlyTerminationCondition
from maze_flatland.reward.flatland_reward import FlatlandReward
from omegaconf import ListConfig

MCTS_NODE_PREFIX_IDENTIFIER = 'mcts_node_info--'


class ImpossibleEpisodeException(Exception):
    """Class for exceptions raised when the generated RailEnv does not have a valid solution.
    :param exception_cause: Extra message for the reason of the exception"""

    def __init__(self, exception_cause=''):
        self.exception_cause = exception_cause
        super().__init__(self.exception_cause)


# pylint: disable=too-many-public-methods
class FlatlandCoreEnvironment(CoreEnv):
    """
    Environment for Flatland (https://flatland.aicrowd.com/intro.html). This environment wraps the original Flatland
    environment and is designed as multi-agent environment, i.e. it expects to receive one action per step and applies
    this action to the currently active agent. The active actor ID is updated after each step.

    Flatland offers customization for
        - observations (tree vs. global, custom),
        - malfunctions,
        - maps/rail networks.

    :param map_width: Map width.
    :param map_height: Map height.
    :param n_trains: Number of trains.
    :param reward_aggregator: Reward aggregator or configuration thereof.
    :param malfunction_generator: Generator for train malfunctions.
    :param line_generator: Generator for train schedules (including speeds).
    :param rail_generator: Generator for rail network.
    :param termination_conditions: List of termination conditions.
    :param renderer: Renderer used to render the maze-flatland environment.
    :param include_maze_state_in_serialization: [Default: True] Whether to include the maze state in the serialization.

    """

    # Create factories to dynamically build complex objects.
    factories = {
        ctype: Factory(base_type=ctype)
        for ctype in (
            FlatlandReward,
            flatland.envs.malfunction_generators.ParamMalfunctionGen,
            flatland.envs.line_generators.BaseLineGen,
            flatland.envs.rail_generators.RailGen,
            FlatlandRendererBase,
        )
    }

    def __init__(
        self,
        map_width: int,
        map_height: int,
        n_trains: int,
        reward_aggregator: Union[FlatlandReward, ConfigType],
        malfunction_generator: Union[flatland.envs.malfunction_generators.ParamMalfunctionGen, ConfigType],
        line_generator: Union[flatland.envs.line_generators.BaseLineGen, ConfigType],
        rail_generator: Union[flatland.envs.rail_generators.RailGen, ConfigType],
        termination_conditions: Union[EarlyTerminationCondition, list[Union[EarlyTerminationCondition, ConfigType]]],
        renderer: Union[FlatlandRendererBase, ConfigType],
        include_maze_state_in_serialization: bool = True,
    ):
        super().__init__()
        self._previous_best_target_distances = [None for _ in range(n_trains)]
        self._current_train_id: int = 0
        self.n_trains: int = n_trains
        # Currently selected random seed. Tracking it is necessary since it has to be provided to Flatland's RailEnv in
        # order to guarantee reproducible setups.
        self._random_seed: Optional[int] = random.randint(0, sys.maxsize)

        # Storage for actions of individual actions before wrapped Flatland environment is stepped through.
        self._actions: dict[int, FlatlandMazeAction] = {}
        # Keep track of last modifying action (i.e. everything except DO_NOTHING) for each train.
        self._last_modifying_actions: dict[int, FlatlandMazeAction] = {
            i: FlatlandMazeAction.STOP_MOVING for i in range(self.n_trains)
        }
        # Pubsub for event to reward routing.
        self._pubsub = Pubsub(self.context.event_service)

        # Instantiate complex objects.
        self._reward_aggregator = self._init_reward_aggregator(reward_aggregator)
        self._malfunction_generator = self.factories[
            flatland.envs.malfunction_generators.ParamMalfunctionGen
        ].instantiate(malfunction_generator)
        self._line_generator = self.factories[flatland.envs.line_generators.BaseLineGen].instantiate(line_generator)
        self._rail_generator = self.factories[flatland.envs.rail_generators.RailGen].instantiate(rail_generator)
        # Set up environment.
        (
            self._move_event_recorder,
            self._block_event_recorder,
            self._detouring_event_recorder,
            self._schedule_events,
            self._departing_events,
            self._rail_env,
            self._kpi_calculator,
            self._seeding_events,
            self._execution_events,
        ) = self._setup_env(map_width, map_height, n_trains)

        # init renderer
        self._renderer = self.factories[FlatlandRendererBase].instantiate(renderer)

        # Initial current maze state.
        self._current_maze_state = None
        self._include_maze_state_in_serialization = include_maze_state_in_serialization

        self._is_cloned = False
        self._rail_env_rnd_state_for_malfunctions = None

        self.departing_stats = None

        # add base termination condition by default.
        self.termination_conditions: list[EarlyTerminationCondition] = []
        if not isinstance(termination_conditions, (list, ListConfig)):
            termination_conditions = [termination_conditions]
        for tc in termination_conditions:
            self.termination_conditions.append(Factory(EarlyTerminationCondition).instantiate(tc))

        assert any(
            isinstance(tc, BaseEarlyTermination) for tc in self.termination_conditions
        ), 'Base termination conditions not found.'

    def _init_reward_aggregator(self, reward_aggregator: Union[FlatlandReward, ConfigType]) -> FlatlandReward:
        """
        Instantiate reward aggregator.
        :param reward_aggregator: BaseRewardAggregator configuration or instance.
        :return: BaseRewardAggregator instance.
        """

        reward_aggregator = self.factories[FlatlandReward].instantiate(reward_aggregator)
        self._pubsub.register_subscriber(reward_aggregator)

        return reward_aggregator

    def _setup_env(
        self, map_width: int, map_height: int, n_trains: int
    ) -> tuple[
        TrainMovementEvents,
        TrainBlockEvents,
        TrainDetouringEvents,
        ScheduleEvents,
        FlatlandDepartingEvents,
        flatland.envs.rail_env.RailEnv,
        FlatlandKPICalculator,
        SeedingEvent,
        FlatlandExecutionEvents,
    ]:
        """
        Initializes environment.
        :param map_width: Map width.
        :param map_height: Map height.
        :param n_trains: Number of trains.
        :return: Event recorder, RailEnv, KPI calculator.
        """

        self._actions = {}
        # Note: RailEnv could also be reset directly.
        rail_env = flatland.envs.rail_env.RailEnv(
            width=map_width,
            height=map_height,
            number_of_agents=n_trains,
            rail_generator=self._rail_generator,
            malfunction_generator=self._malfunction_generator,
            line_generator=self._line_generator,
            obs_builder_object=DummyObservationBuilder(),  # dummy obs builder to save computation
        )
        rail_env.reset(random_seed=self._random_seed)

        return (
            self._pubsub.create_event_topic(TrainMovementEvents),
            self._pubsub.create_event_topic(TrainBlockEvents),
            self._pubsub.create_event_topic(TrainDetouringEvents),
            self._pubsub.create_event_topic(ScheduleEvents),
            self._pubsub.create_event_topic(FlatlandDepartingEvents),
            rail_env,
            FlatlandKPICalculator(),
            self._pubsub.create_event_topic(SeedingEvent),
            self._pubsub.create_event_topic(FlatlandExecutionEvents),
        )

    @override(CoreEnv)
    def step(self, maze_action: FlatlandMazeAction) -> tuple[FlatlandMazeState, float, bool, dict[Any, Any]]:
        """
        Implementation of :py:meth:`~maze.core.env.core_env.CoreEnv.step`.
        Note that we expect the action to describe a directive for the single, currently active agent/train.
        """
        renv = self._rail_env
        self._actions[self._current_train_id] = maze_action
        if maze_action != maze_action.DO_NOTHING:
            self._last_modifying_actions[self._current_train_id] = maze_action
        # _current_maze_state must not be None
        assert self._current_maze_state is not None, 'Reset environment before stepping.'

        # One action for each train available: Step through actual Flatland environment.
        if len(self._actions) == renv.number_of_agents:
            if renv.number_of_agents > 1:
                self.context.increment_env_step()
            observations, _, _, info = self._rail_env.step({aid: self._actions[aid].value for aid in self._actions})

            # Reset current train ID to first one.
            self._current_train_id = 0

            # Get current state after stepping the backend
            self._current_maze_state = FlatlandMazeState(
                self._current_train_id,
                self._last_modifying_actions,
                self._rail_env,
            )

            # Check if any train has departed and if so, record the event.
            for ts in self._current_maze_state.trains:
                # If already departed then skip it.
                if self.departing_stats['trains_departed'][ts.handle]:
                    continue
                if ts.is_on_map():
                    self._record_departure_events(ts)

            # check for env can be terminated (early termination or all trains arrived)
            # (after calculating the new state)
            self._current_maze_state.terminate_episode = self._check_for_termination()

            # Log events (before calculating the reward)
            self.log_flat_step_events()

            # Reset actions after stepping wrapped Flatland environment.
            self._actions = {}
            agent_rewards = self._reward_aggregator.summarize_reward(self._current_maze_state)
        else:
            agent_rewards = np.zeros((renv.number_of_agents,))
            info = {}
            self._current_train_id += 1

            # Update current state
            self._current_maze_state.current_train_id = self._current_train_id
            self._current_maze_state.trains[self._current_train_id - 1].last_action = self._actions[
                self._current_train_id - 1
            ]

        # Summarise reward
        scalar_reward = self._reward_aggregator.to_scalar_reward(agent_rewards)

        # Overwrite the done flag to include early stopping.
        done = self._current_maze_state.terminate_episode
        if done:
            self.log_episode_end_events()

        # When all the trains with an existing valid solution reach their target location, the episode is flagged
        # as successfully solved. This is needed to inform mcts of the positive outcome.
        if self._current_maze_state.all_possible_trains_arrived():
            assert done
            info['Flatland.Done.successful'] = True

        # update mcts status after stepping the backend or if done.
        if self._current_train_id == 0 or done:
            self._update_info_dict_for_mcts(done, info)

        return self._current_maze_state, scalar_reward, done, info

    @override(CoreEnv)
    def get_actor_rewards(self) -> Optional[np.ndarray]:
        """Optional. If this is a multi-step or multi-agent environment, this method should return
        the last reward for all actors from the last structured step.

        NOTE: This needs to be disabled for now, since we might get problems otherwise in the final env step.
              Since the reset is called before collecting the get_actor_rewards in the rollout runner, the reward of
              the next episode might already be stored (if skipping is enabled).

        :return: The last rewards for each agent individually.
        """
        return None

    @override(CoreEnv)
    def reset(self) -> MazeStateType:
        """
        Implementation of :py:meth:`~maze.core.env.core_env.CoreEnv.reset`.
        """
        # override local params due to reset
        self._current_train_id = 0
        self._actions: dict[int, FlatlandMazeAction] = {}
        self._last_modifying_actions: dict[int, FlatlandMazeAction] = {
            i: FlatlandMazeAction.STOP_MOVING for i in range(self.n_trains)
        }

        # Reset environment.
        self._actions = {}
        self._rail_env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=self._random_seed)
        # Hard reset of predictor.
        self._rail_env.dev_pred_dict = {}
        self._current_maze_state = None

        self._renderer.reset()

        self._current_maze_state = FlatlandMazeState(
            self._current_train_id,
            self._last_modifying_actions,
            self._rail_env,
        )

        self._previous_best_target_distances = []
        for train in self._current_maze_state.trains:
            self._previous_best_target_distances.append(
                np.nan_to_num(train.target_distance, posinf=np.prod(self._current_maze_state.map_size) + 1)
            )

        self._current_maze_state.terminate_episode = self._check_for_termination()

        self._schedule_events.impossible_dest(
            sum(train.unsolvable for train in self._current_maze_state.trains) / self.n_trains
        )
        self._seeding_events.seed_used(self._random_seed)
        self._schedule_events.invalid_episode(self._current_maze_state.terminate_episode)
        if self._current_maze_state.terminate_episode:
            raise ImpossibleEpisodeException(f'Seed: {self._random_seed}')

        #  hard reset the malfunction generator.
        # pylint: disable=protected-access
        if self.rail_env.malfunction_generator._rand_idx != 0:
            self.rail_env.malfunction_generator._rand_idx = 0
            self.rail_env.malfunction_generator._cached_rand = None

        self._rail_env_rnd_state_for_malfunctions = self._rail_env.np_random.get_state()

        self.departing_stats = {'asap': 0, 'in_time': 0, 'delay': 0, 'trains_departed': np.zeros(self.n_trains)}
        return self._current_maze_state

    @override(CoreEnv)
    def seed(self, seed: int) -> None:
        """
        Implementation of :py:meth:`~maze.core.env.core_env.CoreEnv.seed`.
        """
        self._random_seed = seed
        self._rail_generator.seed = seed
        self._line_generator.seed = seed

    @override(CoreEnv)
    def close(self) -> None:
        """
        Implementation of :py:meth:`~maze.core.env.core_env.CoreEnv.close`.
        """

        self._renderer.close()

    @override(CoreEnv)
    def get_maze_state(self) -> FlatlandMazeState:
        """
        Implementation of :py:meth:`~maze.core.env.core_env.CoreEnv.get_maze_state`.
        """
        assert self._current_maze_state is not None
        return self._current_maze_state

    def get_renderer(self) -> Renderer:
        """
        Implementation of :py:meth:`~maze.core.env.core_env.CoreEnv.get_renderer`.
        """

        return self._renderer

    @override(CoreEnv)
    def actor_id(self) -> ActorID:
        """
        Implementation of :py:meth:`~maze.core.env.core_env.CoreEnv.actor_id`.
        """

        return ActorID('train_move', self._current_train_id)

    @property
    @override(CoreEnv)
    def agent_counts_dict(self) -> dict[Union[str, int], int]:
        """This is a single sub-step, multi-agent environment."""
        return {'train_move': self._rail_env.number_of_agents}

    @override(CoreEnv)
    def is_actor_done(self) -> bool:
        """
        We check whether current agent is done, since this is a single-policy environment.
        Implementation of :py:meth:`~maze.core.env.core_env.CoreEnv.is_actor_done`.
        """

        return self._rail_env.dones[self._current_train_id]

    @property
    def rail_env(self) -> flatland.envs.rail_env.RailEnv:
        """
        Returns Flatland's RailEnv.
        :return: Flatland's RailEnv instance for this core environment.
        """

        return self._rail_env

    def get_current_seed(self) -> int:
        """Returns the current seed used"""
        return self._random_seed

    @override(CoreEnv)
    def get_serializable_components(self) -> dict[str, Any]:
        pass

    @override(CoreEnv)
    def clone_from(self, env: FlatlandCoreEnvironment) -> None:
        """Initialise the environment to the given state."""
        assert isinstance(env, FlatlandCoreEnvironment)
        # or need to handle all the motions checks and events' tracker.
        assert env.n_trains == self.n_trains, 'Cloning is not possible - mismatching number of trains.'

        self.deserialize_state(env.serialize_state())

        # pylint: disable=protected-access
        assert (
            self.rail_env.malfunction_generator._cached_rand is None
            and env.rail_env.malfunction_generator._cached_rand is None
            or all(
                self.rail_env.malfunction_generator._cached_rand[:10]
                == env.rail_env.malfunction_generator._cached_rand[:10]
            )
        ), 'Not possible to clone safely. Mismatch in the malfunction generator'

    # pylint: disable=protected-access
    def _serialize_rail_env(self) -> list[any]:
        """Supports the serialization of the core_env.
        It decouples the serialization of the backend from the core_env.
        :return: A list of essential parameters needed to be cloned.
        """
        return [
            self.rail_env.agents,
            self.rail_env.np_random.get_state(),
            self.rail_env._elapsed_steps,
            self.rail_env.num_resets,
            self.rail_env.rail,
            self.rail_env.dev_pred_dict,
            self.rail_env.dev_obs_dict,
            self.rail_env.dones,
            self.rail_env._max_episode_steps,
            self.rail_env.active_agents,
            self.rail_env.distance_map.distance_map,
            self.rail_env.distance_map.agents_previous_computation,
            self.rail_env.malfunction_generator.MFP,
            self._rail_env_rnd_state_for_malfunctions,
            self.rail_env.malfunction_generator._rand_idx,
            self.rail_env.malfunction_generator._cached_rand is not None,
        ]

    @override(SimulatedEnvMixin)
    def serialize_state(self) -> bytes:
        """Serialize the current env state and return an object that can be used to deserialize the env again.
        The returned (serialised) object is a triple | (core_env_params, rail_env_params, reward_params).
        """
        assert not self._current_maze_state.terminate_episode, 'Never serialize a state that results in done.'
        core_env_param = (
            self._current_train_id,
            self._last_modifying_actions,
            self._actions,
            self._random_seed,
            (self.context.step_id, self.context.episode_id),
            self._include_maze_state_in_serialization,
            self._previous_best_target_distances,
            self.departing_stats,
        )
        renv_params = self._serialize_rail_env()
        reward_params = self._reward_aggregator.serialize_state()

        if self._include_maze_state_in_serialization:
            core_env_param += (self._current_maze_state,)

        return pickle.dumps((core_env_param, renv_params, reward_params))

    @override(SimulatedEnvMixin)
    def deserialize_state(self, serialised_state: bytes):
        """Deserialize the current env from the given env state
        :param serialised_state: the serialised state to be restored in the current environment.
        """
        old_renv_seed = self.rail_env.random_seed
        core_env_state, renv_state, reward_state = pickle.loads(serialised_state)
        self._current_train_id = core_env_state[0]
        self._last_modifying_actions = core_env_state[1]
        self._actions = core_env_state[2]
        self._random_seed = core_env_state[3]
        self.context.step_id, self.context._episode_id = core_env_state[4]
        self._include_maze_state_in_serialization = core_env_state[5]
        self._previous_best_target_distances = core_env_state[6]
        self.departing_stats = core_env_state[7]

        # end of core env params.
        self._reward_aggregator.deserialize_state(reward_state)
        self._deserialize_rail_env(renv_state)
        # if different seed then different rail topology, re-compute the transition map and store it in cache.
        if self._random_seed != old_renv_seed:
            _ = get_transitions_map(self.rail_env, use_cached=False)
        # As last step, restore/recompute the maze_state
        if self._include_maze_state_in_serialization:
            self._current_maze_state = core_env_state[-1]
        else:
            self._current_maze_state = FlatlandMazeState(
                self._current_train_id,
                self._last_modifying_actions,
                self._rail_env,
            )
            self._current_maze_state.terminate_episode = self._check_for_termination()

        self._renderer.reset()
        self._is_cloned = True

    # pylint: disable=protected-access
    def _deserialize_rail_env(self, serialised_rail_env: list[any]):
        """restore the state of the current rail_env to the state of the given one.
        :param serialised_rail_env: the serial
        """

        self._rail_env.random_seed = self._random_seed
        if self._rail_env.seed_history[-1] != self._random_seed:
            self._rail_env.seed_history.append(self._random_seed)
        self._rail_env.agents = serialised_rail_env[0]
        self._rail_env._elapsed_steps = serialised_rail_env[2]
        self._rail_env.num_resets = serialised_rail_env[3]
        self._rail_env.rail = serialised_rail_env[4]
        self._rail_env.dev_pred_dict = serialised_rail_env[5]
        self._rail_env.dev_obs_dict = serialised_rail_env[6]
        self._rail_env.dones = serialised_rail_env[7]
        self._rail_env._max_episode_steps = serialised_rail_env[8]
        self._rail_env.active_agents = serialised_rail_env[9]
        # deserialize distance map
        self._rail_env.distance_map.agents = self.rail_env.agents
        self._rail_env.distance_map.rail = self.rail_env.rail
        self._rail_env.distance_map.distance_map = serialised_rail_env[10]
        self._rail_env.distance_map.agents_previous_computation = serialised_rail_env[11]
        # deserialize MFP
        self._rail_env.malfunction_generator.MFP = serialised_rail_env[12]
        # borrow rnd generator and seed it with the state used to generate malfunctions data.
        self._rail_env_rnd_state_for_malfunctions = serialised_rail_env[13]
        self._rail_env.malfunction_generator._cached_rand = None
        if serialised_rail_env[-1]:
            self._rail_env.np_random.set_state(self._rail_env_rnd_state_for_malfunctions)
            self._rail_env.malfunction_generator.generate_rand_numbers(self._rail_env.np_random)
        self._rail_env.malfunction_generator._rand_idx = serialised_rail_env[14]

        # restore the true rnd generator
        self._rail_env.np_random.set_state(serialised_rail_env[1])
        # restore agent position
        self._rail_env.agent_positions = np.zeros((self.rail_env.height, self.rail_env.width), dtype=int) - 1
        for agent in self.rail_env.agents:
            if agent.position is not None:
                self._rail_env.agent_positions[agent.position] = agent.handle

    def render(
        self,
        action: None | FlatlandMazeAction | list[FlatlandMazeAction],
        save_in_dir: str | None = None,
    ):
        """Render the current state with the default parameters.

        :param action: The optional action to render. Or a list of actions for each train.
        :param save_in_dir: Optional dir path to save the image in. If None, it shows the image.
        """

        self.get_renderer().render(
            self.get_maze_state(),
            action,
            None,
            rail_env=self.rail_env,
            close_prior_windows=True,
            save_in_dir=save_in_dir,
        )

    def log_flat_step_events(self) -> None:
        """Helper to handle the recording of events after stepping the backend."""
        # block events
        for train in self._current_maze_state.trains:
            if train.is_block:
                self._block_event_recorder.train_blocked(train_id=train.handle)
            if train.deadlock:
                self._block_event_recorder.train_deadlocked(train_id=train.handle)
            # Fire events.
            self._move_event_recorder.train_moved(
                train_id=train.handle,
                goal_reached=train.is_done(),
                train_speed=train.speed,
                target_distance=train.target_distance,
            )
            updated_top_target_distance = np.nan_to_num(
                train.target_distance,
                posinf=np.prod(self._current_maze_state.map_size) + 1,
            )
            delta_detour_distance = updated_top_target_distance - self._previous_best_target_distances[train.handle]
            if train.status == TrainState.MOVING and delta_detour_distance > 0:
                self._detouring_event_recorder.train_detouring(delta_detour_distance)
            self._previous_best_target_distances[train.handle] = updated_top_target_distance

    def log_episode_end_events(self) -> None:
        """Log events at the end of the episode."""
        assert self._current_maze_state.terminate_episode

        arrived_trains = 0
        cancelled_trains = 0
        n_trains_deadlock = 0
        self._move_event_recorder.n_trains(n_trains=self.n_trains)
        for train in self._current_maze_state.trains:
            if train.is_done():
                arrived_trains += 1
                self._move_event_recorder.train_delay(train.arrival_delay)
            elif train.has_not_yet_departed():
                cancelled_trains += 1
            if train.deadlock:
                n_trains_deadlock += 1
        self._move_event_recorder.trains_arrived(arrived_trains / self.n_trains)
        self._move_event_recorder.trains_arrived_possible(
            arrived_trains / (self.n_trains - sum(train.unsolvable for train in self._current_maze_state.trains))
        )
        self._move_event_recorder.trains_cancelled(cancelled_trains / self.n_trains)
        self._block_event_recorder.count_deadlocks(dead_rate=n_trains_deadlock / self.n_trains)

        # Departing events.
        self._departing_events.departure_asap(self.departing_stats['asap'] / self.n_trains)
        self._departing_events.departure_in_time(self.departing_stats['in_time'] / self.n_trains)
        self._departing_events.departure_severe_delay(self.departing_stats['delay'] / self.n_trains)

    def _update_info_dict_for_mcts(self, done: bool, info: dict) -> None:
        """Updates the info dict returned by the environment to include stats to be logged at node level in mcts.
        :param done: Whether the env is done or not.
        :param info: dictionary returned by the backend.
        """
        trains_ready_to_depart = []
        trains_done = []
        deadlocks = []
        target_distances = []
        malfunctioning = []
        for train in self._current_maze_state.trains:
            if train.is_done():
                trains_done.append(train.handle)
            elif train.status == TrainState.READY_TO_DEPART:
                trains_ready_to_depart.append(train.handle)
            if train.status.is_malfunction_state():
                malfunctioning.append(train.handle)
            if train.deadlock:
                deadlocks.append(train.handle)
            target_distances.append(train.target_distance)

        if len(trains_ready_to_depart) > 0:
            info[MCTS_NODE_PREFIX_IDENTIFIER + 'ready_to_depart'] = tuple(trains_ready_to_depart)
        if done:
            info[MCTS_NODE_PREFIX_IDENTIFIER + 'arrived'] = trains_done
        if len(deadlocks) > 0:
            info[MCTS_NODE_PREFIX_IDENTIFIER + 'dead_trains'] = len(deadlocks)
        if len(malfunctioning) > 0:
            info[MCTS_NODE_PREFIX_IDENTIFIER + 'malf_trains'] = len(malfunctioning)

        info[MCTS_NODE_PREFIX_IDENTIFIER + 'dist_to_target'] = tuple(target_distances)

    def is_cloned(self) -> bool:
        """Return true if the environment was cloned or not."""
        return self._is_cloned

    def _check_for_termination(self) -> bool:
        """Helper to checks whether the environment can be terminated.
        :return: Boolean flag indicating whether the environment can be terminated.
        """
        for termination_condition in self.termination_conditions:
            if termination_condition.check_for_termination(self._current_maze_state):
                return True
        return False

    def record_action(self, action: ActionType):
        """Records the action taken.
        :param action: The action before conversion.
        """
        step_key = list(action.keys())[0]
        self._execution_events.action_taken(
            action=action[step_key], step_key=step_key, agent_id=self.actor_id().agent_id
        )

    def _record_departure_events(self, train_state: MazeTrainState):
        """Trigger at departure, it records the event for an agent that has departed.
            Departure is recorded when agent is put on rails.

        :param train_state: The state of the current train that has just left the station.
        """
        assert not self.departing_stats['trains_departed'][train_state.handle]
        self.departing_stats['trains_departed'][train_state.handle] = 1
        # -1 as train is now on map, means the departure signal was given in the previous time step.
        # Done in this way to handle potential last minute malfunctions off map.
        departure_delay = train_state.env_time - train_state.earliest_departure - 1
        eta = departure_delay + train_state.best_travel_time_to_target
        if departure_delay == 0:
            self.departing_stats['asap'] += 1

        elif eta <= train_state.latest_arrival:
            self.departing_stats['in_time'] += 1
        else:
            self.departing_stats['delay'] += 1

        # Delay based on estimated arrival time before departure. Disregarding the negative values.
        normalized_delay = max(0, eta - train_state.latest_arrival) / train_state.latest_arrival

        self._departing_events.departure_delay(normalized_delay)
