"""File holding both the policy for xgboost and the related model."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID
from maze_flatland.env.maze_state import FlatlandMazeState
from xgboost import XGBClassifier, plot_importance, to_graphviz


class XGBoostPolicy(Policy):
    """Policy class that uses XGBOOST ensemble instead of nn.
    :param use_masking: Whether to use masking or not.
    :param **kwargs: Arbitrary keyword arguments, needed to maek it compatible with MCTS.
    """

    def __init__(self, use_masking: bool, **kwargs):
        _ = kwargs
        super().__init__()
        self.action_events = None
        self.model = XGBoostModel(
            n_estimators=1,
            max_depth=1,
            objective=None,
            lr=0,
            n_processes=1,
            seed=1,
            reg_alpha=None,
            gamma=None,
            reg_lambda=None,
        )
        self.use_masking = use_masking
        self.model.load_model(os.getcwd())

    @override(Policy)
    def needs_state(self) -> bool:
        """
        Implementation of :py:meth:`~maze.core.agent.policy.Policy.needs_state`.
        """

        return True

    @override(Policy)
    def seed(self, seed: int) -> None:
        """
        Trained XGBOOST is deterministic.
        """

    # pylint: disable=unused-argument
    @override(Policy)
    def compute_top_action_candidates(
        self,
        observation: ObservationType,
        num_candidates: int,
        maze_state: Optional[FlatlandMazeState],
        env: Optional[BaseEnv] = None,
        actor_id: ActorID = None,
        deterministic: bool = False,
    ) -> tuple[list[ActionType], np.ndarray]:
        """Implementation of :py:meth:`~maze.core.agent.policy.Policy.compute_top_action_candidates`."""
        obs = np.expand_dims(observation['observation'], axis=0)
        mask_name = actor_id.step_key + '_mask'
        mask = 1
        if self.use_masking:
            assert mask_name in observation
            mask = observation[mask_name]

        action_weights = self.model.predict_logit(obs)[0]
        masked_weights = action_weights * mask

        possible_actions = np.where(masked_weights != 0)[0]
        actions = [{'train_move': a} for a in possible_actions]
        soft_probs = stable_softmax(masked_weights[possible_actions])

        return actions, soft_probs

    # pylint: disable=unused-argument
    @override(Policy)
    def compute_action(
        self,
        observation: ObservationType,
        maze_state: Optional[FlatlandMazeState],
        env: Optional[BaseEnv] = None,
        actor_id: ActorID = None,
        deterministic: bool = False,
    ) -> ActionType:
        """
        Implementation of :py:meth:`~maze.core.agent.policy.Policy.compute_action`.
        """
        actions, probs = self.compute_top_action_candidates(
            observation=observation, num_candidates=1, maze_state=maze_state, actor_id=actor_id, env=env
        )
        if deterministic:
            action = actions[np.argmax(probs)]
        else:
            action = np.random.choice(actions, p=probs)
        return action


def stable_softmax(values: list[float] | np.ndarray) -> np.ndarray:
    """Implementation of softmax while avoiding exact values of 0 and 1"""
    # get standard softmax
    max_value = np.max(values)
    exps = np.exp(values - max_value)
    softmax_values = exps / exps.sum()
    # normalise to prevent 0 and 1 while keeping the sum ~ 1
    epsilon_val = 1e-8
    max_val = 1 - epsilon_val * len(values)
    # Adjust the softmax values to avoid 0 and 1
    return softmax_values * (max_val - epsilon_val) + epsilon_val


class XGBoostModel:
    """Class that implements a xgboost ensemble model.

    :param n_estimators: Number of ensemble estimators.
    :param max_depth: Maximum depth of each tree.
    :param objective: Specify the type of learning task and therefore objective.
    :param lr: Learning rate.
    :param n_processes: Number of processes for parallel train.
    :param seed: Used for the reproducibility in training.
    :param gamma: Min loss reduction required for a split on a none.
    :param reg_alpha: Regularization parameter L1.
    :param reg_lambda: Regularization parameter L2.
    """

    def __init__(
        self,
        n_estimators: int,
        max_depth: int,
        lr: float,
        objective: str | None,
        n_processes: int,
        seed: int,
        gamma: float | None,
        reg_alpha: float | None,
        reg_lambda: float | None,
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            objective=objective,
            n_jobs=n_processes,
            seed=seed,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
        )

    def set_feature_names(self, feature_names: list[str]):
        """Overwrites the feature names in the model based on the provided list.
        :param feature_names: list of feature names.
        """
        self.model.get_booster().feature_names = feature_names

    def get_feature_importance(self, importance_type: str = 'gain') -> dict[str:float]:
        """Returns the feature importance as list.
        :param importance_type: type used to determine the importance of each feature.
        :return: dictionary with feature id and its importance.
        """
        return self.model.get_booster().get_score(importance_type=importance_type)

    def plot_graphviz(self, dump_directory: str) -> None:
        """Plot the 'transparent' model.
        :param dump_directory: Directory where to store the figure"""
        tree_id = 0
        graph_model = to_graphviz(self.model, num_trees=tree_id)
        graph_model.render(os.path.join(dump_directory, 'transparent-model'))

    def plot_feature_importance(
        self, dump_directory: str, max_limit: int | None = None, fsize: tuple[int, int] = (12, 15)
    ):
        """Plots the feature importance for the first tree.
            this method needs further investigation and refinements

        :param dump_directory: Directory where to store the figure.
        :param max_limit: Maximum number of features to plot.
        :param fsize: Size of the figure in inches.
        """
        feature_importance_plot = plot_importance(self.model, max_num_features=max_limit)
        dump_fname = os.path.join(dump_directory, f'./{max_limit}_feature_importance.svg')
        if max_limit is None:
            dump_fname = os.path.join(dump_directory, '/all_feature_importance.svg')
        feature_importance_plot.figure.set_size_inches(fsize)
        feature_importance_plot.figure.savefig(dump_fname, bbox_inches='tight')

    def save_model(self, dump_directory: str):
        """Store the current model at the given path.
        :param dump_directory: directory where to save the model.
        """
        fname = 'xgboost.ubj'
        self.model.save_model(os.path.join(dump_directory, f'{fname}'))
        print(f'Model saved @ {dump_directory}')

    def load_model(self, folder: str):
        """Load a stored model from the given folder.
        :param folder: path to the folder.
        """
        _fname = os.path.join(folder, 'xgboost.ubj')
        assert os.path.isfile(_fname)
        self.model.load_model(_fname)

    def predict(self, observation: ObservationType | list[ObservationType]) -> list[int]:
        """Get the prediction based on the provided observation."""
        return self.model.predict(observation)

    def predict_logit(self, observation) -> list[float]:
        """Get the prediction based on the provided observation.
        :return: float values indicating the weights for class-membership.
        """
        return self.model.predict(observation, output_margin=True)
