from __future__ import annotations

import logging

from worldweaver_agent.backends import PlanningBackend, RuleBasedPlanningBackend
from worldweaver_agent.schemas import ClickEvent, ImageFrame, PerceptionResult, PlanningDecision, WorldState

logger = logging.getLogger(__name__)


class NarrativePlanningAgent:
    """Resolve interaction understanding into a story-level next-turn plan."""

    def __init__(self, backend: PlanningBackend):
        self.backend = backend
        self._fallback_backend = RuleBasedPlanningBackend()

    def plan_next_turn(
        self,
        *,
        image: ImageFrame,
        click: ClickEvent,
        state: WorldState,
        interaction: PerceptionResult,
    ) -> PlanningDecision:
        try:
            return self.backend.plan(image=image, click=click, state=state, perception=interaction)
        except Exception as exc:
            logger.warning(
                "Narrative planning backend failed; falling back to rule-based planning for this turn: %s",
                exc,
                exc_info=True,
            )
            return self._fallback_backend.plan(image=image, click=click, state=state, perception=interaction)

    def decide(
        self,
        *,
        image: ImageFrame,
        click: ClickEvent,
        state: WorldState,
        perception: PerceptionResult,
    ) -> PlanningDecision:
        return self.plan_next_turn(image=image, click=click, state=state, interaction=perception)

    def draft_story_outline(self, *, root_topic: str, style_guide: str, world_summary: str) -> str:
        draft = getattr(self.backend, "draft_story_outline", None)
        if draft is None:
            return self._fallback_story_outline(root_topic=root_topic)
        try:
            return draft(root_topic=root_topic, style_guide=style_guide, world_summary=world_summary)
        except Exception as exc:
            logger.warning("Story outline drafting failed; using fallback outline: %s", exc, exc_info=True)
            return self._fallback_story_outline(root_topic=root_topic)

    @staticmethod
    def _fallback_story_outline(*, root_topic: str) -> str:
        return (
            f"- Establish the world of {root_topic} and its central mystery.\n"
            "- Let the user discover a concrete clue through local exploration.\n"
            "- Open a connected path that expands the setting without losing continuity.\n"
            "- Reveal a hidden cause or character motive behind the first mystery.\n"
            "- Converge the explored details into a meaningful turning point."
        )


class PlanningAgent(NarrativePlanningAgent):
    """Backward-compatible alias for the narrative planning stage."""
