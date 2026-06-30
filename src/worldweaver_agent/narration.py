from __future__ import annotations

import logging

from worldweaver_agent.backends import NarrationBackend, RuleBasedNarrationBackend
from worldweaver_agent.schemas import (
    ClickEvent,
    ImageFrame,
    NarrationResult,
    PerceptionResult,
    PlanningDecision,
    RenderResult,
    WorldState,
)

logger = logging.getLogger(__name__)


class NarrationAgent:
    def __init__(self, backend: NarrationBackend):
        self.backend = backend
        self._fallback_backend = RuleBasedNarrationBackend()

    def narrate(
        self,
        *,
        image: ImageFrame,
        click: ClickEvent,
        state: WorldState,
        perception: PerceptionResult,
        plan: PlanningDecision,
        render_result: RenderResult,
    ) -> NarrationResult:
        try:
            return self.backend.narrate(
                image=image,
                click=click,
                state=state,
                perception=perception,
                plan=plan,
                render_result=render_result,
            )
        except Exception as exc:
            logger.warning(
                "Narration backend failed; falling back to rule-based narration for this turn: %s",
                exc,
                exc_info=True,
            )
            return self._fallback_backend.narrate(
                image=image,
                click=click,
                state=state,
                perception=perception,
                plan=plan,
                render_result=render_result,
            )
