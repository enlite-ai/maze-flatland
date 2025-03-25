"""Test file to check that the reward classes can be correctly instantiated without any error."""

from __future__ import annotations

from maze_flatland.env.core_env import FlatlandCoreEnvironment
from maze_flatland.env.maze_action import FlatlandMazeAction
from maze_flatland.reward.constant_reward import ConstantReward
from maze_flatland.reward.default_flatland_v2 import RewardAggregator as DefaultRewardAggregator
from maze_flatland.reward.distance_based import DeltaDistanceReward
from maze_flatland.reward.flatland_reward import FlatlandReward
from maze_flatland.test.env_instantation import create_core_env


def generate_core_env(reward_aggregator: FlatlandReward) -> FlatlandCoreEnvironment:
    """Generates CoreEnv by passing in attributes as classes.
    :param reward_aggregator: Instance of reward aggregator used by the core_env to compute rewards.
    :return: FlatlandCoreEnvironment instance.
    """

    return create_core_env(1, 30, 30, 2, 1 / 1000, {1: 1, 0.5: 0}, False, reward_aggregator=reward_aggregator)


def _step_the_environment(env: FlatlandCoreEnvironment):
    """Takes 10 steps on the given core environment.
    :param env: The core environment to step.
    """
    env.reset()
    for _ in range(10):
        _ = env.step(FlatlandMazeAction.GO_FORWARD)


def test_default_reward_aggregator():
    """Test with the default reward aggregator."""
    core_env = generate_core_env(
        DefaultRewardAggregator(
            alpha=1,
            beta=1,
            reward_for_goal_reached=10,
            penalty_for_start=0,
            penalty_for_stop=0,
            use_train_speed=True,
            penalty_for_block=5,
            penalty_for_deadlock=500,
            distance_penalty_weight=1 / 100,
        )
    )
    _step_the_environment(core_env)


def test_fixed_penalty_reward_aggregator():
    """Test with the fixed penalty reward aggregator."""
    core_env = generate_core_env(ConstantReward(value=-1))
    _step_the_environment(core_env)


def test_distance_based_reward_aggregator():
    """Test with the distance based reward aggregator."""
    core_env = generate_core_env(DeltaDistanceReward(detour_penalty_factor=1))
    _step_the_environment(core_env)
