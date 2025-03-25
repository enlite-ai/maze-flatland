"""Collections of relevant seeds for the scenario 1."""

from __future__ import annotations

DEAD_SEEDS = [1414794670, 1316168957, 1772476189, 1085572085, 747388822, 1774033989, 1921844764, 1179307154]

# Collection of seeds that required the observation space to have
# dummy values out of the distribution as the agents were detouring instead of going to their
# target under certain conditions.
WEIRD_SEEDS = [1224563176, 395393177, 905235680]


def challenging_deadlock_seeds() -> list[int]:
    """Testing seeds for the flatland environment that requires agents to manage deadlocks.
    :return: List of seeds."""
    return DEAD_SEEDS


def challenging_weird_seeds() -> list[int]:
    """Testing seeds for the flatland environment that are challenging to be solved by a policy.
    :return: List of seeds."""
    return WEIRD_SEEDS
