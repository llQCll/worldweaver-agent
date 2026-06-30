from __future__ import annotations

from worldweaver_agent.memory import MemoryAgent
from worldweaver_agent.schemas import ImageFrame, PerceptionResult, PlanningDecision, RenderRequest, RetrievedFact


class FramePlanningAgent:
    """Translate story-level decisions into a renderable next-frame request."""

    def __init__(self, memory_agent: MemoryAgent):
        self.memory_agent = memory_agent

    def compose_next_frame(
        self,
        *,
        image: ImageFrame,
        interaction: PerceptionResult,
        narrative_plan: PlanningDecision,
        retrieved_facts: list[RetrievedFact],
    ) -> RenderRequest:
        return self.memory_agent.build_render_request(
            image_page_id=image.page_id,
            image_uri=image.image_uri,
            perception=interaction,
            plan=narrative_plan,
            retrieved_facts=retrieved_facts,
        )

    def build_render_request(
        self,
        *,
        image: ImageFrame,
        perception: PerceptionResult,
        plan: PlanningDecision,
        retrieved_facts: list[RetrievedFact],
    ) -> RenderRequest:
        return self.compose_next_frame(
            image=image,
            interaction=perception,
            narrative_plan=plan,
            retrieved_facts=retrieved_facts,
        )
