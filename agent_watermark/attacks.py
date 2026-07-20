"""Trajectory attacks for Agent-SAW robustness tests."""

from __future__ import annotations

import copy
import random

from .detector import TrajectoryStep


def drop_random_steps(steps: list[TrajectoryStep], drop_ratio: float, rng: random.Random) -> list[TrajectoryStep]:
    if not steps or drop_ratio <= 0:
        return copy.deepcopy(steps)
    keep = [step for step in steps if rng.random() > drop_ratio]
    return keep or [copy.deepcopy(steps[0])]


def rename_actions(steps: list[TrajectoryStep], rename_map: dict[str, str]) -> list[TrajectoryStep]:
    renamed: list[TrajectoryStep] = []
    for step in steps:
        action = rename_map.get(step.action, step.action)
        renamed.append(
            TrajectoryStep(
                observation=step.observation,
                action=action,
                candidate_actions=step.candidate_actions,
            )
        )
    return renamed


def paraphrase_observations(steps: list[TrajectoryStep], suffix: str = " (rephrased)") -> list[TrajectoryStep]:
    return [
        TrajectoryStep(
            observation=f"{step.observation}{suffix}",
            action=step.action,
            candidate_actions=step.candidate_actions,
        )
        for step in steps
    ]
