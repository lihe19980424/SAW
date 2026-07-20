"""Formal Agent-SAW scenario suites (procedurally generated, fixed seed)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .action_space import ActionSpace


@dataclass
class ToolSelectionScenario:
    scenario_id: str
    prompt: str
    observation: str
    candidate_actions: list[str]
    original_scores: list[float]
    gold_action: str
    # Optional multi-step session (preferred for formal detection).
    steps: list[dict] | None = None
    gold_actions: list[str] | None = None


@dataclass
class TravelPlanningScenario:
    scenario_id: str
    prompt: str
    steps: list[dict]
    gold_actions: list[str]


TOOL_INTENTS: list[tuple[str, str, str]] = [
    ("weather", "What is the weather in {loc} tomorrow?", "check_weather"),
    ("flight", "Find a flight from {src} to {dst} next week.", "book_flight"),
    ("calc", "Compute the total cost for {n} nights at ${price}/night.", "calculator"),
    ("web", "Search recent news about {topic}.", "search_web"),
    ("clarify", "I am not sure which city you mean for travel.", "ask_user"),
]

CITIES = [
    "Tokyo", "Beijing", "Shanghai", "Seoul", "Singapore", "Bangkok", "London",
    "Paris", "Berlin", "New York", "San Francisco", "Sydney", "Toronto", "Dubai",
    "Osaka", "Kyoto", "Hong Kong", "Taipei", "Los Angeles", "Chicago",
]

TOPICS = [
    "AI regulation", "typhoon alerts", "visa policy", "hotel strikes",
    "flight delays", "currency rates", "museum hours", "local festivals",
]

TRAVEL_STEP_TEMPLATES: list[tuple[str, str]] = [
    ("Need transportation options from {src} to {dst}.", "search_flight"),
    ("Flights shortlisted; need lodging near downtown {dst}.", "search_hotel"),
    ("Check outdoor conditions for sightseeing in {dst}.", "check_weather"),
    ("Find attractions suitable for a {days}-day itinerary in {dst}.", "search_attractions"),
    ("Compile the budget-constrained itinerary for the user.", "summarize_plan"),
]


def _softmax_like_scores(
    actions: list[str],
    gold: str,
    rng: np.random.Generator,
    margin: float = 0.10,
) -> list[float]:
    """
    Gold is preferred with a small margin so multiplicative SAW can flip
    near-ties toward high-noise actions (required for mean-noise detection).
    """
    scores = []
    for action in actions:
        if action == gold:
            scores.append(float(0.62 + margin + rng.uniform(0.0, 0.08)))
        elif action == "unknown_action":
            scores.append(float(rng.uniform(0.01, 0.05)))
        else:
            scores.append(float(rng.uniform(0.25, 0.70)))
    gold_idx = actions.index(gold)
    max_other = max(s for i, s in enumerate(scores) if i != gold_idx)
    if scores[gold_idx] <= max_other:
        scores[gold_idx] = max_other + margin
    return scores


def generate_tool_selection_scenarios(n: int, seed: int = 42, steps_per_session: int = 5) -> list[ToolSelectionScenario]:
    """
    Each scenario is a multi-query tool-use session (default 5 steps), so
    trajectory-level mean detection has enough samples for formal metrics.
    """
    space = ActionSpace.tool_selection()
    actions = [a for a in space.actions if a != "unknown_action"]
    rng = np.random.default_rng(seed)
    scenarios: list[ToolSelectionScenario] = []

    for i in range(n):
        steps = []
        gold_actions = []
        prompts = []
        for t in range(steps_per_session):
            intent, prompt_tmpl, gold = TOOL_INTENTS[(i + t) % len(TOOL_INTENTS)]
            loc = CITIES[(i + t) % len(CITIES)]
            src = CITIES[(i + t + 3) % len(CITIES)]
            dst = CITIES[(i + t + 7) % len(CITIES)]
            topic = TOPICS[(i + t) % len(TOPICS)]
            prompt = prompt_tmpl.format(
                loc=loc, src=src, dst=dst, n=2 + ((i + t) % 5), price=80 + 10 * ((i + t) % 8), topic=topic
            )
            observation = f"session={i:04d}; step={t}; intent={intent}; query={prompt}"
            scores = _softmax_like_scores(actions, gold, rng)
            steps.append(
                {
                    "observation": observation,
                    "candidate_actions": list(actions),
                    "original_scores": scores,
                    "gold_action": gold,
                    "prompt": prompt,
                }
            )
            gold_actions.append(gold)
            prompts.append(prompt)

        scenarios.append(
            ToolSelectionScenario(
                scenario_id=f"tool_session_{i:04d}",
                prompt=" | ".join(prompts),
                observation=steps[0]["observation"],
                candidate_actions=list(actions),
                original_scores=steps[0]["original_scores"],
                gold_action=gold_actions[0],
                steps=steps,
                gold_actions=gold_actions,
            )
        )
    return scenarios


def generate_travel_planning_scenarios(n: int, seed: int = 43) -> list[TravelPlanningScenario]:
    space = ActionSpace.travel_planning()
    actions = [a for a in space.actions if a != "unknown_action"]
    rng = np.random.default_rng(seed)
    scenarios: list[TravelPlanningScenario] = []

    for i in range(n):
        src = CITIES[i % len(CITIES)]
        dst = CITIES[(i + 5) % len(CITIES)]
        while dst == src:
            dst = CITIES[(i + 9) % len(CITIES)]
        days = 2 + (i % 4)
        # Fixed 5-step trajectories for stable trajectory-level detection.
        n_steps = 5
        templates = list(TRAVEL_STEP_TEMPLATES)
        # Cycle templates to fill 5 steps, always ending with summarize.
        chosen = [templates[j % len(templates)] for j in range(n_steps - 1)]
        rng.shuffle(chosen)
        chosen.append(TRAVEL_STEP_TEMPLATES[-1])

        steps = []
        gold_actions = []
        for step_i, (obs_tmpl, gold) in enumerate(chosen):
            obs = obs_tmpl.format(src=src, dst=dst, days=days) + f" | step={step_i}"
            scores = _softmax_like_scores(actions, gold, rng, margin=0.10)
            steps.append(
                {
                    "observation": obs,
                    "candidate_actions": list(actions),
                    "original_scores": scores,
                    "gold_action": gold,
                }
            )
            gold_actions.append(gold)

        scenarios.append(
            TravelPlanningScenario(
                scenario_id=f"travel_{i:04d}_{src.lower()}_{dst.lower()}",
                prompt=f"Plan a {days}-day trip from {src} to {dst} within a budget.",
                steps=steps,
                gold_actions=gold_actions,
            )
        )
    return scenarios


# Keep tiny handcrafted fixtures for unit smoke tests.
TOOL_SELECTION_SCENARIOS = generate_tool_selection_scenarios(2, seed=0)
TRAVEL_PLANNING_SCENARIOS = generate_travel_planning_scenarios(1, seed=1)
