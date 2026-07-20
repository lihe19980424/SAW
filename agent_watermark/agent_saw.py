"""Agent-level SAW: multiplicative scaling over planner action scores."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import numpy as np

from .action_space import ActionSpace


def derive_private_key(seed: int) -> bytes:
    rng = random.Random(seed)
    return rng.getrandbits(1024).to_bytes(128, "big")


@dataclass
class AgentSAWConfig:
    hash_key: int = 15485863
    private_key_seed: int = 42
    beta: float = 0.9
    std: float = 0.08
    mean: float = 1.0
    noise: str = "uniform"
    # Mean-noise threshold aligned with Token-SAW / paper (not z-score).
    mean_threshold: float = 1.03
    left_rand: float = 0.0
    # Softmax sampling is required for mean-noise detectability when action
    # spaces are small; argmax with large margins yields identical W/NW actions.
    selection_mode: str = "sample"
    temperature: float = 0.7


@dataclass
class StepRecord:
    prompt: str
    observation: str
    candidate_actions: list[str]
    original_scores: list[float]
    noise_values: dict[str, float]
    scaled_scores: dict[str, float]
    selected_action: str
    watermark_on: bool
    detection_score: float | None = None
    final_answer: str | None = None
    task_success: bool | None = None


class AgentSAW:
    """Apply SAW-style multiplicative scaling to discrete action scores."""

    def __init__(self, action_space: ActionSpace, config: AgentSAWConfig | None = None) -> None:
        self.action_space = action_space
        self.config = config or AgentSAWConfig()
        self.private_key = derive_private_key(self.config.private_key_seed)
        self._global_indices = self._build_global_indices()

    def _build_global_indices(self) -> np.ndarray:
        rng = np.random.default_rng(self.config.hash_key)
        permutation = rng.permutation(self.action_space.size)
        if self.config.beta >= 1.0:
            return permutation
        if self.config.beta <= 0.0:
            return np.array([], dtype=int)
        return permutation[: int(self.action_space.size * self.config.beta)]

    def _sample_noise(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if self.config.noise == "uniform":
            spread = self.config.std * np.sqrt(12.0)
            offset = 1.0 - spread / 2.0
            return rng.random(size) * spread + offset
        return rng.normal(self.config.mean, self.config.std, size=size)

    def _local_seed(self, observation: str, history: list[str]) -> int:
        payload = observation.encode("utf-8") + b"|" + "|".join(history).encode("utf-8")
        digest = hashlib.sha256()
        digest.update(payload)
        digest.update(self.private_key)
        return int.from_bytes(digest.digest(), "big") % (2**32 - 1)

    def build_hybrid_noise(self, observation: str, history: list[str]) -> np.ndarray:
        global_rng = np.random.default_rng(self.config.hash_key)
        global_noise = self._sample_noise(global_rng, self.action_space.size)
        local_rng = np.random.default_rng(self._local_seed(observation, history))
        local_noise = self._sample_noise(local_rng, self.action_space.size)
        noise = local_noise.copy()
        noise[self._global_indices] = global_noise[self._global_indices]
        if self.config.left_rand >= 0:
            noise = np.clip(noise, self.config.left_rand, None)
        return noise

    def _select(self, scores: dict[str, float], sample_seed: int) -> str:
        actions = list(scores.keys())
        vals = np.asarray([scores[a] for a in actions], dtype=np.float64)
        if self.config.selection_mode == "argmax":
            return actions[int(np.argmax(vals))]
        # Softmax sampling (LLM-like).
        temperature = max(self.config.temperature, 1e-6)
        logits = vals / temperature
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        rng = np.random.default_rng(sample_seed)
        return str(rng.choice(actions, p=probs))

    def scale_scores(
        self,
        observation: str,
        history: list[str],
        candidate_actions: list[str],
        original_scores: list[float],
        watermark_on: bool = True,
        sample_seed: int | None = None,
    ) -> tuple[dict[str, float], dict[str, float], str]:
        if len(candidate_actions) != len(original_scores):
            raise ValueError("candidate_actions and original_scores must have the same length")

        noise_vector = self.build_hybrid_noise(observation, history)
        noise_values: dict[str, float] = {}
        scaled_scores: dict[str, float] = {}
        for action, score in zip(candidate_actions, original_scores):
            idx = self.action_space.index_of(self.action_space.normalize(action))
            phi = float(noise_vector[idx])
            noise_values[action] = phi
            scaled_scores[action] = float(score * phi) if watermark_on else float(score)

        if sample_seed is None:
            sample_seed = self._local_seed(observation, history) ^ (1 if watermark_on else 0)
        selected_action = self._select(scaled_scores, sample_seed)
        return noise_values, scaled_scores, selected_action

    def mean_noise_score(self, selected_noise_values: list[float]) -> float:
        """Paper detection statistic: mean reconstructed noise of selected actions."""
        if not selected_noise_values:
            return 0.0
        return float(np.mean(selected_noise_values))

    def is_watermarked(self, mean_score: float) -> bool:
        return mean_score > self.config.mean_threshold

    def to_step_record(
        self,
        *,
        prompt: str,
        observation: str,
        candidate_actions: list[str],
        original_scores: list[float],
        watermark_on: bool,
        history: list[str],
        final_answer: str | None = None,
        task_success: bool | None = None,
        sample_seed: int | None = None,
    ) -> StepRecord:
        noise_values, scaled_scores, selected_action = self.scale_scores(
            observation,
            history,
            candidate_actions,
            original_scores,
            watermark_on=watermark_on,
            sample_seed=sample_seed,
        )
        selected_noise = [noise_values[selected_action]] if selected_action in noise_values else []
        detection_score = self.mean_noise_score(selected_noise)
        return StepRecord(
            prompt=prompt,
            observation=observation,
            candidate_actions=candidate_actions,
            original_scores=original_scores,
            noise_values=noise_values,
            scaled_scores=scaled_scores,
            selected_action=selected_action,
            watermark_on=watermark_on,
            detection_score=detection_score,
            final_answer=final_answer,
            task_success=task_success,
        )
