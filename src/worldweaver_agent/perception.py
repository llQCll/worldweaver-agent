from __future__ import annotations

from worldweaver_agent.interaction import InteractionUnderstandingAgent
from worldweaver_agent.schemas import ClickEvent, ImageFrame, PerceptionResult, WorldState


class PerceptionAgent:
    """Narrative-perception stage that converts interaction understanding into a planning-ready perception result.

    The current project keeps this stage as a thin wrapper around the interaction
    model so the high-level pipeline can read clearly as:
    interaction -> perception -> planning -> rendering.
    """

    def __init__(self, interaction_agent: InteractionUnderstandingAgent):
        self.interaction_agent = interaction_agent

    def perceive(
        self,
        *,
        image: ImageFrame,
        click: ClickEvent,
        state: WorldState,
    ) -> PerceptionResult:
        return self.interaction_agent.infer(image=image, click=click, state=state)

    def infer(
        self,
        *,
        image: ImageFrame,
        click: ClickEvent,
        state: WorldState,
    ) -> PerceptionResult:
        return self.perceive(image=image, click=click, state=state)

    def analyze(self, image: ImageFrame, click: ClickEvent, state: WorldState) -> PerceptionResult:
        return self.perceive(image=image, click=click, state=state)


__all__ = ["InteractionUnderstandingAgent", "PerceptionAgent"]
