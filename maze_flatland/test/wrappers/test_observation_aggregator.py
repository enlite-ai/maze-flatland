"""File holdings the tests for the observation aggregator wrappers."""


from __future__ import annotations

from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType
from maze_flatland.env.masking.mask_builder import LogicMaskBuilder
from maze_flatland.env.maze_env import FlatlandEnvironment
from maze_flatland.space_interfaces.action_conversion.directional import DirectionalAC
from maze_flatland.space_interfaces.observation_conversion.simple import SimpleObservationConversion
from maze_flatland.test.env_instantation import create_core_env
from maze_flatland.wrappers.masking_wrapper import FlatlandMaskingWrapper
from maze_flatland.wrappers.observation_aggregator_wrapper import (
    AggregationMode,
    ObservationAggregatorFlattening,
    ObservationAggregatorWithPaddingWrapper,
)


def create_example_env():
    """Create a small example env."""
    core_env = create_core_env(3, 30, 30, 2, 0, {1: 1}, False, 3, 3)
    env = FlatlandEnvironment(
        core_env,
        {'train_move': DirectionalAC()},
        {'train_move': SimpleObservationConversion(True)},
    )
    return env


def assertion_on_observation(obs: ObservationType, with_mask: bool):
    """Runs the assertions on a given observation.

    :param obs: the observation to check.
    :param with_mask: whether the assertions should consider the mask or not.
    """
    assert len(obs.keys()) == 1 + with_mask, f'Expected at most 2 keys found: {len(obs.keys())}'
    if with_mask:
        assert 'train_move_mask' in obs, 'Mask not found in observation.'
    assert 'observation' in obs, 'Observation not found in observation.'


def run_env_wrapped(env: MazeEnv):
    """Helper method to run the environment.

    :param env: the environment to run.
    """
    assert hasattr(env, 'exclude_mask_from_aggregation')
    with_mask = env.exclude_mask_from_aggregation
    env.seed(123)
    obs = env.reset()
    assertion_on_observation(obs, with_mask=with_mask)
    obs_space = env.observation_space.spaces
    assert len(obs_space) == 1 + with_mask
    for key, space in obs_space.items():
        assert obs[key] in space, f'Observation {key} not in observation space.'
    for _ in range(10):
        obs, _, _, _ = env.step({'train_move': 2})
        assertion_on_observation(obs, with_mask=with_mask)


def test_obs_aggregator_flattening():
    """Test the ObservationAggregatorFlattening. wrapper."""

    for exclude_mask in [True, False]:
        env = create_example_env()
        if exclude_mask:
            env = FlatlandMaskingWrapper.wrap(env, mask_builder=LogicMaskBuilder())
        run_env_wrapped(ObservationAggregatorFlattening.wrap(env, exclude_mask_from_aggregation=exclude_mask))


def test_obs_aggregator_padding():
    """Test the ObservationAggregatorFlattening. wrapper."""
    for aggr_mode in [AggregationMode.STACK, AggregationMode.CONCATENATE]:
        for exclude_mask in [True, False]:
            env = create_example_env()
            if exclude_mask:
                env = FlatlandMaskingWrapper.wrap(env, mask_builder=LogicMaskBuilder())
            run_env_wrapped(
                ObservationAggregatorWithPaddingWrapper.wrap(
                    env, exclude_mask_from_aggregation=exclude_mask, aggregation_mode=aggr_mode
                )
            )


def test_cloning():
    """Test the cloning functionality of the aggregator wrappers."""

    env1 = ObservationAggregatorWithPaddingWrapper.wrap(
        FlatlandMaskingWrapper(create_example_env(), mask_builder=LogicMaskBuilder()),
        exclude_mask_from_aggregation=True,
        aggregation_mode='stack',
    )
    env2 = ObservationAggregatorWithPaddingWrapper.wrap(
        create_example_env(), exclude_mask_from_aggregation=False, aggregation_mode='concatenate'
    )

    env1.seed(123)
    env2.seed(456)
    _ = [e.reset() for e in [env1, env2]]
    env2.clone_from(env1)

    assert env2.aggregation_mode == env1.aggregation_mode
    assert env2.observation_spaces_dict == env1.observation_spaces_dict
