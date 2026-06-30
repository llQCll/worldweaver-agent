from __future__ import annotations

import logging

from worldweaver_agent.backends import PerceptionBackend, RuleBasedPerceptionBackend
from worldweaver_agent.schemas import ClickEvent, ImageFrame, PerceptionResult, WorldState

logger = logging.getLogger(__name__)


class InteractionUnderstandingAgent:
    """Infer user interaction semantics from the current frame and session state."""

    def __init__(self, backend: PerceptionBackend):
        self.backend = backend
        self._fallback_backend = RuleBasedPerceptionBackend()

    def infer(
        self,
        *,
        image: ImageFrame,
        click: ClickEvent,
        state: WorldState,
    ) -> PerceptionResult:
        try:
            return self.backend.analyze(image=image, click=click, state=state)
        except Exception as exc:
            logger.warning(
                "Interaction understanding backend failed; falling back to rule-based perception for this turn: %s",
                exc,
                exc_info=True,
            )
            return self._fallback_backend.analyze(image=image, click=click, state=state)

    def analyze(self, image: ImageFrame, click: ClickEvent, state: WorldState) -> PerceptionResult:
        return self.infer(image=image, click=click, state=state)


class PerceptionAgent(InteractionUnderstandingAgent):
    """Backward-compatible alias for the interaction understanding stage."""

