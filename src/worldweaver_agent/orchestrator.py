from __future__ import annotations

from worldweaver_agent.frame_planning import FramePlanningAgent
from worldweaver_agent.interaction import InteractionUnderstandingAgent
from worldweaver_agent.memory import MemoryAgent
from worldweaver_agent.narrative_planning import NarrativePlanningAgent
from worldweaver_agent.narration import NarrationAgent
from worldweaver_agent.perception import PerceptionAgent
from worldweaver_agent.rendering import RenderingAgent
from worldweaver_agent.retrieval import RetrievalAgent
from worldweaver_agent.schemas import ClickEvent, ImageFrame, PerceptionResult, PlanningDecision, ExplorationTurn


class InteractiveWorldExplorer:
    def __init__(
        self,
        *,
        perception_agent: PerceptionAgent,
        memory_agent: MemoryAgent,
        planning_agent: NarrativePlanningAgent,
        retrieval_agent: RetrievalAgent,
        rendering_agent: RenderingAgent,
        narration_agent: NarrationAgent,
        frame_planning_agent: FramePlanningAgent | None = None,
        interaction_agent: InteractionUnderstandingAgent | None = None,
    ):
        self.interaction_agent = interaction_agent or perception_agent.interaction_agent
        self.perception_agent = perception_agent
        self.memory_agent = memory_agent
        self.narrative_planning_agent = planning_agent
        self.planning_agent = planning_agent
        self.retrieval_agent = retrieval_agent
        self.rendering_agent = rendering_agent
        self.narration_agent = narration_agent
        self.frame_planning_agent = frame_planning_agent or FramePlanningAgent(memory_agent)

    def step(self, *, image: ImageFrame, click: ClickEvent) -> ExplorationTurn:
        state = self.memory_agent.state
        interaction = self.interaction_agent.infer(image=image, click=click, state=state)
        perception = self.perception_agent.perceive(image=image, click=click, state=state)
        plan = self.narrative_planning_agent.plan_next_turn(
            image=image,
            click=click,
            state=state,
            interaction=perception,
        )
        retrieved_facts = self.retrieval_agent.search(plan.retrieval_queries, state)
        render_request = self.frame_planning_agent.compose_next_frame(
            image=image,
            interaction=perception,
            narrative_plan=plan,
            retrieved_facts=retrieved_facts,
        )
        render_result = self.rendering_agent.render(render_request=render_request, state=state)
        self.memory_agent.commit_turn_without_narration(
            source_page_id=image.page_id,
            click=click,
            perception=perception,
            plan=plan,
            retrieved_facts=retrieved_facts,
            render_result=render_result,
        )
        return ExplorationTurn(
            perception=perception,
            plan=plan,
            retrieved_facts=retrieved_facts,
            render_request=render_request,
            render_result=render_result,
            narration=None,
            world_state=self.memory_agent.state.to_dict(),
        )

    def analyze_intent(self, *, image: ImageFrame, click: ClickEvent) -> tuple[PerceptionResult, PlanningDecision]:
        state = self.memory_agent.state
        interaction = self.interaction_agent.infer(image=image, click=click, state=state)
        perception = self.perception_agent.perceive(image=image, click=click, state=state)
        plan = self.narrative_planning_agent.plan_next_turn(
            image=image,
            click=click,
            state=state,
            interaction=perception,
        )
        return perception, plan
