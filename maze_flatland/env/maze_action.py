"""
MazeAction for Flatland environment.
"""

from __future__ import annotations

from enum import IntEnum

from maze_flatland.env.backend_utils import get_next_position


class FlatlandMazeAction(IntEnum):
    """
    Possible actions for trains as described in
    http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/intro_observation_actions.html#action-space and
    https://gitlab.aicrowd.com/flatland/flatland/blob/master/flatland/envs/rail_env.py.
    Contained values virtually identical to flatland.envs.rail_envs.RailEnvActions.
    """

    DO_NOTHING = 0
    DEVIATE_LEFT = 1
    GO_FORWARD = 2
    DEVIATE_RIGHT = 3
    STOP_MOVING = 4

    def __repr__(self):
        return f'Flatland action: Train will {self.name}.'

    @classmethod
    def global_directions_to_agent_view(cls, train_direction: int, pos_direction: int) -> FlatlandMazeAction:
        """Map the global directions in terms of 0: north, 1: east, 2: south, 3: west to forward, left and right.

        :param train_direction: The current train direction.
        :param pos_direction: The possible direction to change to.
        :return: The necessary action to turn from the current direction to the possible direction
        """
        if train_direction - pos_direction in [1, -3]:
            return cls.DEVIATE_LEFT
        if train_direction - pos_direction in [-1, 3]:
            return cls.DEVIATE_RIGHT

        return cls.GO_FORWARD

    @classmethod
    def action_to_global_position(
        cls, train_position: tuple[int, int], action: int, train_direction: int
    ) -> tuple[int, int]:
        """Get the resulting position after applying the action.

        :param train_position: The current train position on the grid.
        :param action: The possible action to consider.
        :param train_direction: The current train direction.
        :return: The position after applying the action.
        """
        if action == cls.DO_NOTHING:
            return get_next_position(*train_position, train_direction)

        if action == cls.STOP_MOVING:
            return train_position

        if action == cls.DEVIATE_RIGHT:
            new_direction = (train_direction + 1) % 4
            return get_next_position(*train_position, new_direction)

        if action == cls.DEVIATE_LEFT:
            new_direction = (4 + (train_direction - 1)) % 4
            return get_next_position(*train_position, new_direction)

        return get_next_position(*train_position, train_direction)
