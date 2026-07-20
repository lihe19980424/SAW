"""Trajectory-only Agent-SAW detector (mean reconstructed noise)."""

from __future__ import annotations

from dataclasses import dataclass

from .action_space import ActionSpace
from .agent_saw import AgentSAW, AgentSAWConfig


@dataclass
class TrajectoryStep:
    observation: str
    action: str
    candidate_actions: list[str] | None = None


@dataclass
class DetectionResult:
    mean_score: float
    num_steps: int
    is_watermarked: bool
    # Backward-compatible alias used by older scripts.
    z_score: float = 0.0


class TrajectoryDetector:
    """Rebuild action-level noise from a saved trajectory without original scores."""

    def __init__(self, action_space: ActionSpace, config: AgentSAWConfig | None = None) -> None:
        self.action_space = action_space
        self.agent_saw = AgentSAW(action_space, config=config)

    def detect(self, steps: list[TrajectoryStep]) -> DetectionResult:
        history: list[str] = []
        selected_noise: list[float] = []

        for step in steps:
            noise_vector = self.agent_saw.build_hybrid_noise(step.observation, history)
            canonical = self.action_space.normalize(step.action)
            idx = self.action_space.index_of(canonical)
            selected_noise.append(float(noise_vector[idx]))
            history.append(canonical)

        mean_score = self.agent_saw.mean_noise_score(selected_noise)
        return DetectionResult(
            mean_score=mean_score,
            num_steps=len(steps),
            is_watermarked=self.agent_saw.is_watermarked(mean_score),
            z_score=mean_score,
        )
