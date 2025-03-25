"""Holds the wrapper to enable skipping in flatland."""

from __future__ import annotations

from typing import Any

import numpy as np
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID
from maze.core.log_events.skipping_events import SkipEvent
from maze.core.wrappers.wrapper import EnvType, Wrapper
from maze_flatland.wrappers.masking_wrapper import FlatlandMaskingWrapper


class EnvDoneInResetException(InterruptedError):
    """Exception raised if the env is already done in a reset"""


class FlatStepSkippingWrapper(Wrapper[MazeEnv]):
    """Wrapper used to enable the skipping based on the flat step.
    A step is skipped iff all the agents have no option in action selection.
    :param env: Environment to wrap.
    :param do_skipping_in_reset: Whether to enable the skipping on the reset.
    """

    def __init__(self, env: MazeEnv, do_skipping_in_reset: bool):
        super().__init__(env)
        assert isinstance(env, FlatlandMaskingWrapper), 'Env needs to be wrapped with FlatlandMaskingWrapper'
        self._step_events = self.core_env.context.event_service.create_event_topic(SkipEvent)
        self.do_skipping_in_reset = do_skipping_in_reset

    def reset(self) -> ObservationType:
        """Intercept ``BaseEnv.reset``"""
        obs = self.env.reset()
        if self.do_skipping_in_reset:
            skipped, observation, rewards, _done, _info = self.loop_skipping()
            if _done:
                raise EnvDoneInResetException
            if skipped:
                obs = observation
        return obs

    def check_skipping_conditions_and_get_actions(self) -> tuple[bool, list[dict]]:
        """Methods to check whether the flat step has to be skipped and gives the actions for the skipping.
        :return: a tuple containing: boolean indicating whether the flat_step should be skipped and the actions
                 for the skipped step.
        """

        masks = self.get_mask_for_flat_step()
        # This work under the following assumption: given train, its mask has at least 1 value set to true.
        # As a consequence, the minimum number of True flags matches the number of trains.
        skip_step = np.count_nonzero(masks) == self.env.n_trains
        if skip_step:
            actions = [self.action_conversion.action_from_masking(mask) for mask in masks]
        else:
            actions = [self.action_conversion.noop_action() for _ in masks]
        return skip_step, actions

    def loop_skipping(self) -> tuple[bool, ObservationType | None, list[float], bool, dict[Any, Any]]:
        """Do the skipping until at least an agent has a choice or the episode is finished.
        :return: Tuple with flag set to true if skipping has happened,
                    the last observation, cumulative reward, done flag and info.
        """

        observation = None
        rewards = []
        _done = False
        _info = {}
        skipped = False
        skip_step, actions = self.check_skipping_conditions_and_get_actions()
        # all trains have exactly 1 single option to follow
        while skip_step:
            for train_idx, a in enumerate(actions):
                assert self.env.get_maze_state().current_train_id == train_idx
                observation, _rew, _done, _info = self.env.step(a)
                rewards.append(_rew)
            # flag that skipping has happened
            skipped = True
            self._step_events.flat_step(flat_step_is_skipped=True)
            if _done:
                break
            skip_step, actions = self.check_skipping_conditions_and_get_actions()
        return skipped, observation, rewards, _done, _info

    # pylint: disable=protected-access
    def step(self, action) -> tuple[Any, Any, bool, dict[Any, Any]]:
        """Intercept ``BaseEnv.step`` and tracks the metrics.
        :param action: The action to take"""
        observation, reward, done, info = self.env.step(action)
        if self.env.is_flat_step():
            self._step_events.flat_step(flat_step_is_skipped=False)

        if done or not self.env.is_flat_step():
            return observation, reward, done, info

        has_skipped, skip_obs, skip_rews, skip_done, skip_info = self.loop_skipping()
        # update the parameters if skipping has happened.
        if has_skipped:
            assert len(skip_rews) > 0
            observation, done, info = skip_obs, skip_done, skip_info
            skip_rews.insert(0, reward)
            info['skipping.internal_rewards'] = skip_rews
            reward = sum(skip_rews)
        return observation, reward, done, info

    def clone_from(self, env: EnvType) -> None:
        """Overrides the clone_from method.
        :param env: The source env to be cloned."""
        self.env.clone_from(env.env)


class SubStepSkippingWrapper(Wrapper[MazeEnv]):
    """Wrapper used to enable the skipping at the substep level.
    A sub step is skipped iff the selected agent has no option in action selection.
    :param env: Environment to wrap.
    :param do_skipping_in_reset: Whether to enable the skipping on the reset.
    """

    def __init__(self, env: MazeEnv, do_skipping_in_reset: bool):
        super().__init__(env)
        assert isinstance(env, FlatlandMaskingWrapper), 'Env needs to be wrapped with FlatlandMaskingWrapper'
        self._step_events = self.core_env.context.event_service.create_event_topic(SkipEvent)
        self.do_skipping_in_reset = do_skipping_in_reset

    def reset(self) -> ObservationType:
        """Intercept ``BaseEnv.reset``"""
        obs = self.env.reset()
        if self.do_skipping_in_reset:
            skipped, observation, rewards, _done, _info = self.loop_skipping(obs)
            if _done:
                raise EnvDoneInResetException
            if skipped:
                obs = observation
        return obs

    @staticmethod
    def check_skipping_conditions_and_get_actions(obs: ObservationType, actor_id: ActorID) -> bool:
        """Checks if the choice for the current agent has to be skipped.
        :param obs: Current observation
        :param actor_id:Current actor id.

        :return: boolean indicating whether the step should be skipped.
        """
        mask = obs[f'{actor_id.step_key}_mask']
        skip_substep = np.count_nonzero(mask) == 1
        return skip_substep

    def loop_skipping(
        self, observation: ObservationType
    ) -> tuple[bool, ObservationType | None, list[float], bool, dict[Any, Any]]:
        """Do the skipping until at least an agent has a choice or the episode is finished.
        :return: Tuple with flag set to true if skipping has happened,
                    the last observation, cumulative reward, done flag and info.
        """
        rewards = []
        _done = False
        _info = {}
        skipped = False
        skip_step = self.check_skipping_conditions_and_get_actions(observation, self.actor_id())
        # all trains have exactly 1 single option to follow
        while skip_step:
            action = self.action_conversion.action_from_masking(observation[f'{self.actor_id().step_key}_mask'])
            observation, _rew, _done, _info = self.env.step(action)
            rewards.append(_rew)
            # flag that skipping has happened
            skipped = True
            self._step_events.sub_step(sub_step_is_skipped=True)
            if _done:
                break
            skip_step = self.check_skipping_conditions_and_get_actions(observation, self.actor_id())
        return skipped, observation, rewards, _done, _info

    # pylint: disable=protected-access
    def step(self, action) -> tuple[Any, Any, bool, dict[Any, Any]]:
        """Intercept ``BaseEnv.step`` and tracks the metrics.
        :param action: The action to take"""
        observation, reward, done, info = self.env.step(action)
        self._step_events.sub_step(sub_step_is_skipped=False)
        if done:
            return observation, reward, done, info

        has_skipped, skip_obs, skip_rews, skip_done, skip_info = self.loop_skipping(observation)
        # update the parameters if skipping has happened.
        if has_skipped:
            assert len(skip_rews) > 0
            observation, done, info = skip_obs, skip_done, skip_info
            skip_rews.insert(0, reward)
            info['skipping.internal_rewards'] = skip_rews
            reward = sum(skip_rews)
        return observation, reward, done, info

    def clone_from(self, env: EnvType) -> None:
        """Overrides the clone_from method.
        :param env: The source env to be cloned."""
        self.env.clone_from(env.env)
