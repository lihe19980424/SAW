"""Canonical action registry for Agent-SAW experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_ALIASES: dict[str, list[str]] = {
    "search_web": ["web_search", "browse_web", "internet_search"],
    "check_weather": ["weather", "get_weather", "weather_api"],
    "calculator": ["calc", "compute", "math_tool"],
    "book_flight": ["flight_booking", "reserve_flight"],
    "ask_user": ["clarify", "request_info", "user_clarification"],
    "search_flight": ["find_flight", "flight_search"],
    "search_hotel": ["find_hotel", "hotel_search"],
    "search_attractions": ["find_attractions", "poi_search"],
    "summarize_plan": ["summarize", "final_plan", "plan_summary"],
}


@dataclass
class ActionSpace:
    """Fixed action vocabulary with optional alias normalization."""

    actions: list[str]
    aliases: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_ALIASES))
    unk_action: str = "unknown_action"

    def __post_init__(self) -> None:
        self._alias_to_canonical: dict[str, str] = {}
        for action in self.actions:
            self._alias_to_canonical[action.lower()] = action
        for canonical, alias_list in self.aliases.items():
            if canonical not in self.actions:
                continue
            for alias in alias_list:
                self._alias_to_canonical[alias.lower()] = canonical

    @property
    def size(self) -> int:
        return len(self.actions)

    def index_of(self, action: str) -> int:
        canonical = self.normalize(action)
        if canonical not in self.actions:
            raise KeyError(f"Unknown action: {action}")
        return self.actions.index(canonical)

    def normalize(self, action: str) -> str:
        key = action.strip().lower()
        return self._alias_to_canonical.get(key, self.unk_action if self.unk_action in self.actions else action)

    @classmethod
    def tool_selection(cls) -> "ActionSpace":
        return cls(
            actions=[
                "search_web",
                "check_weather",
                "calculator",
                "book_flight",
                "ask_user",
                "unknown_action",
            ]
        )

    @classmethod
    def travel_planning(cls) -> "ActionSpace":
        return cls(
            actions=[
                "search_flight",
                "search_hotel",
                "check_weather",
                "search_attractions",
                "summarize_plan",
                "ask_user",
                "unknown_action",
            ]
        )
