"""File holdings tests checking that there are no differences in the backend based
on the last checked version (v 4.0.1)"""
from __future__ import annotations

import os
import pickle
import unittest

import flatland
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs import malfunction_generators
from flatland.envs.malfunction_generators import MalfunctionParameters
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.space_interfaces.action_conversion.directional import DirectionalAC
from maze_flatland.space_interfaces.observation_conversion.simple import SimpleObservationConversion
from maze_flatland.test.env_instantation import create_core_env
from maze_flatland.test.test_utils import _compare


def test_rail():
    """Initialises an environment and compares its data against a serialised file.
    Supported flatland-rl Version: 4.0.4"""
    env = create_core_env(3, 35, 35, 3, 1 / 10, {1: 1})
    env = FlatlandEnvironment(
        env,
        action_conversion={'train_move': DirectionalAC()},
        observation_conversion={'train_move': SimpleObservationConversion(False)},
    )
    env.seed(1234)
    _ = env.reset()
    re = env.rail_env

    agents_info = [
        {
            'earliest_departure': agent.earliest_departure,
            'latest_arrival': agent.latest_arrival,
            'direction': agent.direction,
            'initial_position': agent.position,
            'target': agent.target,
        }
        for agent in re.agents
    ]

    malfunction_times = {tid: [] for tid in range(env.n_trains)}
    done = False
    action = {'train_move': 2}
    while not done:
        if env.is_flat_step():
            for train in env.get_maze_state().trains:
                if train.malfunction_time_left > 0:
                    malfunction_times[train.handle].append(train.env_time)
        # get the malfunctions and records the step.
        _, _, done, info = env.step(action)
    env_info = {
        'distance_maps': re.distance_map.distance_map,
        'agents_data': agents_info,
        'malfunction_times': malfunction_times,
    }

    pickled_datapath = os.path.split(os.path.dirname(__file__))[0] + '/serialised_data/env_info_v_4.0.4.pkl'
    with open(pickled_datapath, 'rb') as f:
        expected_agents_info = pickle.load(f)

    for k, v in env_info.items():
        assert k in expected_agents_info
        assert _compare(v, expected_agents_info[k]), (
            f'Expected key {k} not matching.\n'
            + f'maze-flatland supports version 4.0.4. Found flatland-rl version: {flatland.__version__}'
        )


class TestOnMapStateMachineSpeed1(unittest.TestCase):
    def setUp(self):
        """Setup testing RailEnv."""
        self.rail_env = RailEnv(
            width=30,
            height=30,
            number_of_agents=1,
            obs_builder_object=DummyObservationBuilder(),
            malfunction_generator=malfunction_generators.NoMalfunctionGen(),
            random_seed=1234,
        )
        _ = self.rail_env.reset(random_seed=1234)

    def fast_forward_to_on_map_moving(self):
        """Fast-forward to a state for testing."""
        agent = self.rail_env.agents[0]
        self.rail_env.step({0: RailEnvActions.DO_NOTHING})
        self.rail_env.step({0: RailEnvActions.MOVE_FORWARD})
        assert agent.state == TrainState.MOVING

    def test_on_map_from_moving(self):
        """Test the state machine of the flatland-agent in a moving state."""

        # (MOVING, DO_NOTHING) -> (MOVING)
        self.setUp()
        self.fast_forward_to_on_map_moving()
        agent = self.rail_env.agents[0]
        self.rail_env.step({0: RailEnvActions.DO_NOTHING})
        assert agent.state == TrainState.MOVING

        # (MOVING, MOVE) -> (MOVING)
        self.setUp()
        self.fast_forward_to_on_map_moving()
        agent = self.rail_env.agents[0]
        self.rail_env.step({0: RailEnvActions.MOVE_FORWARD})
        assert agent.state == TrainState.MOVING

        # (MOVING, STOP) -> (STOPPED)
        self.setUp()
        self.fast_forward_to_on_map_moving()
        agent = self.rail_env.agents[0]
        self.rail_env.step({0: RailEnvActions.STOP_MOVING})
        assert agent.state == TrainState.STOPPED

    def fast_forward_to_on_map_stopped(self):
        """Fast-forward to a state for testing."""
        agent = self.rail_env.agents[0]
        self.rail_env.step({0: RailEnvActions.DO_NOTHING})
        self.rail_env.step({0: RailEnvActions.MOVE_FORWARD})
        self.rail_env.step({0: RailEnvActions.STOP_MOVING})
        assert agent.state == TrainState.STOPPED

    def test_on_map_from_stopped(self):
        """Test the state machine of the flatland-agent in a stopped state."""

        # (STOPPED, DO_NOTHING) -> (STOPPED)
        self.setUp()
        self.fast_forward_to_on_map_stopped()
        agent = self.rail_env.agents[0]
        self.rail_env.step({0: RailEnvActions.DO_NOTHING})
        assert agent.state == TrainState.STOPPED

        # (STOPPED, MOVE) -> (MOVING)
        self.setUp()
        self.fast_forward_to_on_map_stopped()
        agent = self.rail_env.agents[0]
        self.rail_env.step({0: RailEnvActions.MOVE_FORWARD})
        assert agent.state == TrainState.MOVING

        # (STOPPED, STOP) -> (STOPPED)
        self.setUp()
        self.fast_forward_to_on_map_stopped()
        agent = self.rail_env.agents[0]
        self.rail_env.step({0: RailEnvActions.STOP_MOVING})
        assert agent.state == TrainState.STOPPED


class TestOffMapStateMachineSpeed1(unittest.TestCase):
    def setUp(self):
        """Setup testing RailEnv."""
        self.rail_env = RailEnv(
            width=30,
            height=30,
            number_of_agents=1,
            obs_builder_object=DummyObservationBuilder(),
            malfunction_generator=malfunction_generators.NoMalfunctionGen(),
            random_seed=1234,
        )
        _ = self.rail_env.reset(random_seed=1234)

    def fast_forward_to_ready_to_depart(self, action_to_use: RailEnvActions):
        """Fast-forward to a state for testing."""
        agent = self.rail_env.agents[0]
        assert agent.state == TrainState.WAITING
        self.rail_env.step({0: action_to_use})
        assert agent.state == TrainState.READY_TO_DEPART

    def test_off_map(self):
        """Test the state machine of the flatland-agent in a Ready to depart state."""
        for action in [RailEnvActions.MOVE_FORWARD, RailEnvActions.STOP_MOVING, RailEnvActions.DO_NOTHING]:
            # (READY_TO_DEPART, MOVING) -> (MOVING - ON MAP)
            self.setUp()
            self.fast_forward_to_ready_to_depart(action)
            agent = self.rail_env.agents[0]
            self.rail_env.step({0: RailEnvActions.MOVE_FORWARD})
            assert agent.state == TrainState.MOVING
            assert agent.state.is_on_map_state()

            # (READY_TO_DEPART, STOP) -> (READY_TO_DEPART - OFF MAP)
            self.setUp()
            self.fast_forward_to_ready_to_depart(action)
            agent = self.rail_env.agents[0]
            self.rail_env.step({0: RailEnvActions.STOP_MOVING})
            assert agent.state == TrainState.READY_TO_DEPART
            assert agent.state.is_off_map_state()

            # (READY_TO_DEPART, DO_NOTHING) -> (READY_TO_DEPART - OFF MAP)
            self.setUp()
            self.fast_forward_to_ready_to_depart(action)
            agent = self.rail_env.agents[0]
            self.rail_env.step({0: RailEnvActions.DO_NOTHING})
            assert agent.state == TrainState.READY_TO_DEPART
            assert agent.state.is_off_map_state()


class TestMalfunctionsOnMapSpeed1(unittest.TestCase):
    def setUp(self):
        """Setup testing RailEnv."""
        self.rail_env = RailEnv(
            width=30,
            height=30,
            number_of_agents=1,
            obs_builder_object=DummyObservationBuilder(),
            malfunction_generator=malfunction_generators.ParamMalfunctionGen(MalfunctionParameters(0.1, 5, 5)),
            random_seed=1234,
        )
        _ = self.rail_env.reset(random_seed=1234)

    def fast_forward_before_malfunction_with_do_nothing_in_moving_state(self):
        """Fast-forward to a state for testing."""
        # After 7 step the agent should be in a malfunction
        agent = self.rail_env.agents[0]
        self.rail_env.step({0: RailEnvActions.DO_NOTHING})
        self.rail_env.step({0: RailEnvActions.MOVE_FORWARD})
        for ii in range(5):
            self.rail_env.step({0: RailEnvActions.DO_NOTHING})
        assert agent.state == TrainState.MOVING

    def fast_forward_before_malfunction_with_do_nothing_in_stopped_state(self):
        """Fast-forward to a state for testing."""
        # After 7 step the agent should be in a malfunction
        agent = self.rail_env.agents[0]
        self.rail_env.step({0: RailEnvActions.DO_NOTHING})
        self.rail_env.step({0: RailEnvActions.MOVE_FORWARD})
        self.rail_env.step({0: RailEnvActions.STOP_MOVING})
        for ii in range(4):
            self.rail_env.step({0: RailEnvActions.DO_NOTHING})
        assert agent.state == TrainState.STOPPED

    def test_on_map_malfunction(self):
        """Test the behaviour for malfunctioning trains when last action is do nothing"""
        for fast_forward_method in [
            self.fast_forward_before_malfunction_with_do_nothing_in_stopped_state,
            self.fast_forward_before_malfunction_with_do_nothing_in_moving_state,
        ]:
            # The fast-forward method advances the env one ts before the malfunction will happen and the train state
            # is either stopped or moving.
            for last_action_before in [
                RailEnvActions.MOVE_FORWARD,
                RailEnvActions.STOP_MOVING,
                RailEnvActions.DO_NOTHING,
            ]:
                # The last action is the action that is (tried) to be performed when the agent is going into a
                # malfunction.
                for first_action in [
                    RailEnvActions.MOVE_FORWARD,
                    RailEnvActions.STOP_MOVING,
                    RailEnvActions.DO_NOTHING,
                ]:
                    # The first action is the action performed when the state == MALFUNCTION and
                    # malfunction_handler.in_malfunction == FALSE
                    self.setUp()
                    fast_forward_method()
                    agent = self.rail_env.agents[0]

                    pos_before_malfunction = agent.position
                    # Here is the action that will be saved!
                    self.rail_env.step({0: last_action_before})

                    assert agent.state == TrainState.MALFUNCTION
                    assert agent.malfunction_handler.malfunction_down_counter == 5

                    for _ in range(5):
                        self.rail_env.step({0: RailEnvActions.DO_NOTHING})

                    # The step before the malfunction
                    assert agent.state == TrainState.MALFUNCTION
                    assert not agent.malfunction_handler.in_malfunction
                    assert agent.malfunction_handler.malfunction_down_counter == 0
                    assert agent.state.is_on_map_state()

                    self.rail_env.step({0: first_action})
                    if first_action in [RailEnvActions.MOVE_FORWARD]:
                        assert agent.position == (pos_before_malfunction[0] + 1, pos_before_malfunction[1])
                        assert agent.state == TrainState.MOVING
                    else:
                        assert agent.position == pos_before_malfunction
                        assert agent.state == TrainState.STOPPED


class TestMalfunctionsOffMapSpeed1(unittest.TestCase):
    def setUp(self):
        """Setup testing RailEnv."""
        self.rail_env = RailEnv(
            width=30,
            height=30,
            number_of_agents=1,
            obs_builder_object=DummyObservationBuilder(),
            malfunction_generator=malfunction_generators.ParamMalfunctionGen(MalfunctionParameters(0.1, 5, 5)),
            random_seed=1234,
        )
        _ = self.rail_env.reset(random_seed=1234)

    def fast_forward_before_malfunction_with_do_nothing(self):
        """Fast-forward to a state for testing."""
        # After 7 step the agent should be in a malfunction
        agent = self.rail_env.agents[0]
        for ii in range(7):
            self.rail_env.step({0: RailEnvActions.DO_NOTHING})
        assert agent.state == TrainState.READY_TO_DEPART

    def fast_forward_before_malfunction_with_stop(self):
        """Fast-forward to a state for testing."""
        # After 7 step the agent should be in a malfunction
        agent = self.rail_env.agents[0]
        for ii in range(7):
            self.rail_env.step({0: RailEnvActions.STOP_MOVING})
        assert agent.state == TrainState.READY_TO_DEPART

    def test_off_map_malfunction_MOVE_STOP_after_malfunction(self):
        """Test the behaviour for malfunctioning trains going into a malfunction when the first action
        is move or stop.
        ==> The train will always depart. The state of the train depends on the last action before.
        """
        for fast_forward_method in [
            self.fast_forward_before_malfunction_with_do_nothing,
            self.fast_forward_before_malfunction_with_stop,
        ]:
            for last_action_before in [
                RailEnvActions.MOVE_FORWARD,
                RailEnvActions.STOP_MOVING,
                RailEnvActions.DO_NOTHING,
            ]:
                # The last action is the action that is (tried) to be performed when the agent is going into a
                # malfunction.
                for first_action in [RailEnvActions.MOVE_FORWARD, RailEnvActions.STOP_MOVING]:
                    # The first action is the action performed when the state == MALFUNCTION and
                    # malfunction_handler.in_malfunction == FALSE
                    self.setUp()
                    fast_forward_method()
                    agent = self.rail_env.agents[0]
                    assert agent.state == TrainState.READY_TO_DEPART

                    # Here is the action that will be saved!
                    self.rail_env.step({0: last_action_before})

                    assert agent.state == TrainState.MALFUNCTION_OFF_MAP
                    assert agent.malfunction_handler.malfunction_down_counter == 5

                    for _ in range(5):
                        self.rail_env.step({0: RailEnvActions.DO_NOTHING})

                    # The step before the malfunction
                    assert agent.state == TrainState.MALFUNCTION_OFF_MAP
                    assert not agent.malfunction_handler.in_malfunction
                    assert agent.malfunction_handler.malfunction_down_counter == 0
                    assert agent.position is None
                    assert agent.state.is_off_map_state()

                    self.rail_env.step({0: first_action})
                    if first_action == RailEnvActions.MOVE_FORWARD:
                        assert agent.state.is_on_map_state()
                        assert agent.state == TrainState.MOVING
                    else:
                        assert agent.state.is_on_map_state()
                        assert agent.state == TrainState.STOPPED

    def test_off_map_malfunction_DO_NOTHING_after_malfunction(self):
        """Test the behaviour for malfunctioning trains going into a malfunction when the first action
        is move or stop.
        ==> The train will either be placed on map or stay off map depending on the last_action_before.
        """
        for fast_forward_method in [
            self.fast_forward_before_malfunction_with_do_nothing,
            self.fast_forward_before_malfunction_with_stop,
        ]:
            # The last action is the action that is (tried) to be performed when the agent is going into a
            # malfunction.
            for last_action_before in [
                RailEnvActions.MOVE_FORWARD,
                RailEnvActions.STOP_MOVING,
                RailEnvActions.DO_NOTHING,
            ]:
                first_action = RailEnvActions.DO_NOTHING
                # The first action is the action performed when the state == MALFUNCTION and
                # malfunction_handler.in_malfunction == FALSE
                self.setUp()
                fast_forward_method()
                agent = self.rail_env.agents[0]
                assert agent.state == TrainState.READY_TO_DEPART

                # Here is the action that will be saved!
                self.rail_env.step({0: last_action_before})

                assert agent.state == TrainState.MALFUNCTION_OFF_MAP
                assert agent.malfunction_handler.malfunction_down_counter == 5

                for _ in range(5):
                    self.rail_env.step({0: RailEnvActions.DO_NOTHING})

                # The step before the malfunction
                assert agent.state == TrainState.MALFUNCTION_OFF_MAP
                assert not agent.malfunction_handler.in_malfunction
                assert agent.malfunction_handler.malfunction_down_counter == 0
                assert agent.position is None
                assert agent.state.is_off_map_state()

                self.rail_env.step({0: first_action})
                if last_action_before in [RailEnvActions.DO_NOTHING, RailEnvActions.STOP_MOVING]:
                    assert agent.state.is_off_map_state()
                    assert agent.state == TrainState.READY_TO_DEPART
                else:
                    assert agent.state.is_on_map_state()
                    assert agent.state == TrainState.MOVING

    def test_off_map_malfunction(self):
        """Test the behaviour for malfunctioning trains going into a malfunction."""
        for fast_forward_method in [
            self.fast_forward_before_malfunction_with_do_nothing,
            self.fast_forward_before_malfunction_with_stop,
        ]:
            for last_action_before in [
                RailEnvActions.MOVE_FORWARD,
                RailEnvActions.STOP_MOVING,
                RailEnvActions.DO_NOTHING,
            ]:
                # The last action is the action that is (tried) to be performed when the agent is going into a
                # malfunction.
                for first_action in [
                    RailEnvActions.MOVE_FORWARD,
                    RailEnvActions.STOP_MOVING,
                    RailEnvActions.DO_NOTHING,
                ]:
                    # The first action is the action performed when the state == MALFUNCTION and
                    # malfunction_handler.in_malfunction == FALSE
                    self.setUp()
                    fast_forward_method()
                    agent = self.rail_env.agents[0]
                    assert agent.state == TrainState.READY_TO_DEPART

                    # Here is the action that will be saved!
                    self.rail_env.step({0: last_action_before})

                    assert agent.state == TrainState.MALFUNCTION_OFF_MAP
                    assert agent.malfunction_handler.malfunction_down_counter == 5

                    for _ in range(5):
                        self.rail_env.step({0: RailEnvActions.DO_NOTHING})

                    # The step before the malfunction
                    assert agent.state == TrainState.MALFUNCTION_OFF_MAP
                    assert not agent.malfunction_handler.in_malfunction
                    assert agent.malfunction_handler.malfunction_down_counter == 0
                    assert agent.position is None
                    assert agent.state.is_off_map_state()

                    stored_action = agent.action_saver.is_action_saved
                    self.rail_env.step({0: first_action})
                    if first_action == RailEnvActions.MOVE_FORWARD:
                        assert agent.state.is_on_map_state()
                        assert agent.state == TrainState.MOVING
                    elif first_action == RailEnvActions.DO_NOTHING and stored_action:
                        assert agent.state.is_on_map_state()
                        assert agent.state == TrainState.MOVING
                    elif first_action == RailEnvActions.DO_NOTHING and not stored_action:
                        assert agent.state.is_off_map_state()
                        assert agent.state == TrainState.READY_TO_DEPART
                    else:
                        assert agent.state.is_on_map_state()
                        assert agent.state == TrainState.STOPPED


class TestMalfunctionsOffMapWaitingSpeed1(unittest.TestCase):
    def setUp(self):
        self.rail_env: RailEnv | None = None

    def setUp_v1(self):
        """Setup testing RailEnv."""
        self.rail_env = RailEnv(
            width=30,
            height=30,
            number_of_agents=5,
            obs_builder_object=DummyObservationBuilder(),
            malfunction_generator=malfunction_generators.ParamMalfunctionGen(MalfunctionParameters(0.1, 5, 5)),
            random_seed=1234,
        )
        _ = self.rail_env.reset(random_seed=1234)

    def fast_forward_before_malfunction_in_waiting(self):
        """Fast-forward to a state for testing."""
        # After 7 step the agent should be in a malfunction
        for ii in range(11):
            self.rail_env.step({aa.handle: RailEnvActions.STOP_MOVING for aa in self.rail_env.agents})

        agent = self.rail_env.agents[1]
        assert agent.state == TrainState.WAITING

    def test_waiting_malfunction(self):
        """Make sure that if a malfunction occurs while waiting state and then during the malfunction the train becomes
        ready to depart. Any action as the last and first action will result int the expected behaviour as if:
        MALFUNCTION_OFF_MAP AND malfunction_handler.in_malfunction == True and not agent.action_saver.is_action_saved.
        --> NO action will be saved!
        """
        # The first action is the action performed when the state == MALFUNCTION and
        # malfunction_handler.in_malfunction == FALSE
        for fast_forward_method in [
            self.fast_forward_before_malfunction_in_waiting,
        ]:
            for last_action_before in [
                RailEnvActions.MOVE_FORWARD,
                RailEnvActions.STOP_MOVING,
                RailEnvActions.DO_NOTHING,
            ]:
                # The last action is the action that is (tried) to be performed when the agent is going into a
                # malfunction.
                for first_action in [
                    RailEnvActions.MOVE_FORWARD,
                    RailEnvActions.STOP_MOVING,
                    RailEnvActions.DO_NOTHING,
                ]:
                    self.setUp_v1()
                    fast_forward_method()
                    agent = self.rail_env.agents[1]
                    assert agent.state == TrainState.WAITING

                    # Last action before the malfunction, this will not be saved when the agent is waiting.
                    self.rail_env.step({1: last_action_before})

                    assert agent.state == TrainState.MALFUNCTION_OFF_MAP
                    assert agent.malfunction_handler.malfunction_down_counter == 5
                    assert not agent.action_saver.is_action_saved

                    for _ in range(5):
                        self.rail_env.step({1: RailEnvActions.DO_NOTHING})

                    # The step before the malfunction
                    assert agent.state == TrainState.MALFUNCTION_OFF_MAP
                    assert not agent.malfunction_handler.in_malfunction
                    assert agent.malfunction_handler.malfunction_down_counter == 0
                    assert agent.position is None
                    assert agent.state.is_off_map_state()

                    self.rail_env.step({1: first_action})
                    print(last_action_before, first_action)
                    if first_action == RailEnvActions.MOVE_FORWARD:
                        assert agent.state.is_on_map_state()
                        assert agent.state == TrainState.MOVING
                    elif first_action == RailEnvActions.STOP_MOVING:
                        assert agent.state.is_on_map_state()
                        assert agent.state == TrainState.STOPPED
                    else:
                        assert agent.state.is_off_map_state()
                        assert agent.state == TrainState.READY_TO_DEPART

    def setUp_v2(self):
        """Setup testing RailEnv."""
        self.rail_env = RailEnv(
            width=30,
            height=30,
            number_of_agents=20,
            obs_builder_object=DummyObservationBuilder(),
            malfunction_generator=malfunction_generators.ParamMalfunctionGen(MalfunctionParameters(0.1, 5, 5)),
            random_seed=1234,
        )
        _ = self.rail_env.reset(random_seed=1234)

    def test_waiting_malfunction_back_to_waiting(self):
        """Make sure that if a malfunction occurs while waiting and the malfunction is shorter than the earliest
        departure the train will just go back to the waiting state.
        """
        for last_action_before in [
            RailEnvActions.MOVE_FORWARD,
            RailEnvActions.STOP_MOVING,
            RailEnvActions.DO_NOTHING,
        ]:
            # The last action is the action that is (tried) to be performed when the agent is going into a
            # malfunction.
            for first_action in [RailEnvActions.MOVE_FORWARD, RailEnvActions.STOP_MOVING, RailEnvActions.DO_NOTHING]:
                self.setUp_v2()
                agent = self.rail_env.agents[12]
                assert agent.state == TrainState.WAITING
                self.rail_env.step({12: last_action_before})
                assert agent.state == TrainState.MALFUNCTION_OFF_MAP
                assert agent.malfunction_handler.malfunction_down_counter == 5

                for _ in range(5):
                    self.rail_env.step({1: RailEnvActions.DO_NOTHING})

                # The step before the malfunction
                assert agent.state == TrainState.MALFUNCTION_OFF_MAP
                assert not agent.malfunction_handler.in_malfunction
                assert agent.malfunction_handler.malfunction_down_counter == 0
                assert agent.position is None
                assert agent.state.is_off_map_state()

                self.rail_env.step({1: first_action})
                assert agent.state.is_off_map_state()
                assert agent.state == TrainState.WAITING
                assert not agent.action_saver.is_action_saved
