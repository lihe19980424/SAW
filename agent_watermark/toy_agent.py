"""Rule-based toy agent used before wiring a real LLM planner."""

from __future__ import annotations

from dataclasses import dataclass, field

from .agent_saw import AgentSAW, StepRecord
from .action_space import ActionSpace
from .tasks import ToolSelectionScenario, TravelPlanningScenario


@dataclass
class ToyAgent:
    action_space: ActionSpace
    agent_saw: AgentSAW
    history: list[str] = field(default_factory=list)

    def act_tool_selection(self, scenario: ToolSelectionScenario, watermark_on: bool = True) -> StepRecord:
        """Single-step convenience API (kept for smoke tests)."""
        record = self.agent_saw.to_step_record(
            prompt=scenario.prompt,
            observation=scenario.observation,
            candidate_actions=scenario.candidate_actions,
            original_scores=scenario.original_scores,
            watermark_on=watermark_on,
            history=self.history.copy(),
            task_success=None,
        )
        self.history.append(record.selected_action)
        record.task_success = record.selected_action == scenario.gold_action
        return record

    def act_tool_session(self, scenario: ToolSelectionScenario, watermark_on: bool = True) -> list[StepRecord]:
        """Multi-step tool-use session used in formal experiments."""
        steps = scenario.steps
        if not steps:
            return [self.act_tool_selection(scenario, watermark_on=watermark_on)]
        records: list[StepRecord] = []
        golds = scenario.gold_actions or []
        for i, step in enumerate(steps):
            record = self.agent_saw.to_step_record(
                prompt=step.get("prompt", scenario.prompt),
                observation=step["observation"],
                candidate_actions=step["candidate_actions"],
                original_scores=step["original_scores"],
                watermark_on=watermark_on,
                history=self.history.copy(),
            )
            self.history.append(record.selected_action)
            if i < len(golds):
                record.task_success = record.selected_action == golds[i]
            records.append(record)
        return records

    def act_travel_plan(self, scenario: TravelPlanningScenario, watermark_on: bool = True) -> list[StepRecord]:
        records: list[StepRecord] = []
        for step in scenario.steps:
            record = self.agent_saw.to_step_record(
                prompt=scenario.prompt,
                observation=step["observation"],
                candidate_actions=step["candidate_actions"],
                original_scores=step["original_scores"],
                watermark_on=watermark_on,
                history=self.history.copy(),
            )
            self.history.append(record.selected_action)
            records.append(record)
        return records

    def reset(self) -> None:
        self.history.clear()
