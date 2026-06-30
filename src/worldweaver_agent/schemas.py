from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any
from uuid import uuid4


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    return value


class ExplorationAction(str, Enum):
    ZOOM_IN = "zoom_in"
    BRANCH_OUT = "branch_out"
    REVEAL = "reveal"
    REFRAME = "reframe"


@dataclass(frozen=True)
class NormalizedBox:
    left: float
    top: float
    width: float
    height: float


@dataclass(frozen=True)
class ClickEvent:
    x: float
    y: float
    user_hint: str | None = None
    interaction_type: str = "click"
    selection_box: NormalizedBox | None = None
    press_duration_ms: int | None = None


@dataclass(frozen=True)
class ImageFrame:
    page_id: str
    image_uri: str | None = None
    prompt: str | None = None
    width: int | None = None
    height: int | None = None
    summary: str | None = None


@dataclass(frozen=True)
class CandidateEntity:
    name: str
    description: str
    region: NormalizedBox | None = None
    confidence: float = 0.0
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IntentOption:
    action: ExplorationAction
    label: str
    description: str
    confidence: float = 0.0
    target_label: str | None = None


@dataclass
class UserPreferenceSignal:
    axis: str
    label: str
    score: float = 0.0
    evidence: list[str] = field(default_factory=list)


@dataclass
class UserFeedbackRecord:
    page_id: str
    feedback_type: str
    label: str
    created_at: float
    axes: dict[str, float] = field(default_factory=dict)
    note: str | None = None
    plan_action: str | None = None
    target_label: str | None = None


@dataclass(frozen=True)
class EntityReference:
    image_uri: str
    source_page_id: str | None = None
    caption: str | None = None
    region: NormalizedBox | None = None
    reference_type: str = "crop"


@dataclass(frozen=True)
class EntityConditioning:
    entity_id: str
    entity_name: str
    visual_signature: str | None = None
    reference_images: list[EntityReference] = field(default_factory=list)


@dataclass(frozen=True)
class SceneReference:
    image_uri: str
    page_id: str
    summary: str | None = None
    reference_type: str = "full_frame"


@dataclass
class SceneMemory:
    scene_id: str
    scene_name: str
    description: str
    branch_id: str
    first_seen_page_id: str | None = None
    last_seen_page_id: str | None = None
    page_ids: list[str] = field(default_factory=list)
    references: list[SceneReference] = field(default_factory=list)
    update_notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PerceptionResult:
    target_label: str
    region_caption: str
    clicked_region: NormalizedBox | None
    action: ExplorationAction
    action_description: str
    user_profile_summary: str
    intent_hint: str
    confidence: float
    focus_type: str = "region"
    focus_subject: str | None = None
    story_role: str | None = None
    emotion_hint: str | None = None
    next_panel_expectation: str | None = None
    interaction_intent: str | None = None
    interaction_reason: str | None = None
    suggested_story_direction: str | None = None
    intent_options: list[IntentOption] = field(default_factory=list)
    intent_confidence_threshold: float = 0.7
    requires_user_confirmation: bool = True
    accepted_intent_option: IntentOption | None = None
    candidate_entities: list[CandidateEntity] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievedFact:
    query: str
    snippet: str
    source_title: str
    source_url: str | None = None
    confidence: float = 0.0


@dataclass(frozen=True)
class PlanningDecision:
    action: ExplorationAction
    rationale: str
    branch_id: str
    branch_label: str
    world_update: str
    render_prompt: str
    negative_prompt: str = ""
    continuity_mode: str | None = None
    scene_location: str | None = None
    primary_actor: str | None = None
    primary_action: str | None = None
    supporting_subject: str | None = None
    next_story_beat: str | None = None
    story_purpose: str | None = None
    shot_type: str | None = None
    narrative_function: str | None = None
    transition_type: str | None = None
    prompt_subject: str | None = None
    prompt_action: str | None = None
    prompt_environment_change: str | None = None
    prompt_new_element: str | None = None
    prompt_background_continuity: str | None = None
    protected_entities: list[str] = field(default_factory=list)
    continuity_notes: list[str] = field(default_factory=list)
    retrieval_queries: list[str] = field(default_factory=list)
    style_directives: list[str] = field(default_factory=list)
    user_profile_summary: str | None = None
    global_intent_alignment: str | None = None
    global_intent_rationale: str | None = None
    story_outline_update: str | None = None
    narrative_beat: str | None = None
    narration_brief: str | None = None
    narration_style: str | None = None


@dataclass(frozen=True)
class RenderRequest:
    session_id: str
    branch_id: str
    source_page_id: str
    target_label: str
    action: ExplorationAction
    render_prompt: str
    negative_prompt: str
    world_summary: str
    branch_summary: str
    retrieved_facts: list[RetrievedFact] = field(default_factory=list)
    entity_conditioning: list[EntityConditioning] = field(default_factory=list)
    reference_image_uri: str | None = None
    conditioning: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderResult:
    page_id: str
    image_uri: str
    revised_prompt: str
    page_summary: str
    world_facts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NarrationResult:
    story_text: str
    summary_text: str = ""
    caption_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageRecord:
    page_id: str
    branch_id: str
    scene_id: str | None
    source_page_id: str | None
    image_uri: str
    prompt: str
    page_summary: str
    action: ExplorationAction
    target_label: str
    click: ClickEvent
    perception: PerceptionResult
    plan: PlanningDecision | None = None
    retrieved_facts: list[RetrievedFact] = field(default_factory=list)
    world_facts: list[str] = field(default_factory=list)
    narration: NarrationResult | None = None
    narration_status: str = "ready"
    narration_error: str | None = None
    render_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BranchState:
    branch_id: str
    label: str
    page_ids: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class WorldEntity:
    entity_id: str
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    mentions: int = 1
    visual_signature: str | None = None
    first_seen_page_id: str | None = None
    last_seen_page_id: str | None = None
    reference_bank: list[EntityReference] = field(default_factory=list)


@dataclass
class WorldState:
    session_id: str
    root_topic: str
    style_guide: str
    world_summary: str
    user_profile_summary: str = "用户刚开始探索这个互动世界，画像仍在形成中。"
    user_preference_signals: dict[str, UserPreferenceSignal] = field(default_factory=dict)
    user_feedback_history: list[UserFeedbackRecord] = field(default_factory=list)
    story_outline: str = ""
    story_outline_source: str = "none"
    story_outline_revision: int = 0
    pages: dict[str, PageRecord] = field(default_factory=dict)
    branches: dict[str, BranchState] = field(default_factory=dict)
    scenes: dict[str, SceneMemory] = field(default_factory=dict)
    entities: dict[str, WorldEntity] = field(default_factory=dict)
    current_page_id: str | None = None
    current_branch_id: str = "main"
    current_scene_id: str | None = None
    history: list[str] = field(default_factory=list)

    @classmethod
    def create(cls, root_topic: str, style_guide: str) -> "WorldState":
        session_id = uuid4().hex
        return cls(
            session_id=session_id,
            root_topic=root_topic,
            style_guide=style_guide,
            world_summary=f"Interactive world centered on: {root_topic}.",
            branches={
                "main": BranchState(
                    branch_id="main",
                    label="Main Branch",
                    summary=f"Primary exploration path for {root_topic}.",
                )
            },
        )

    def ensure_branch(self, branch_id: str, label: str) -> BranchState:
        branch = self.branches.get(branch_id)
        if branch is None:
            branch = BranchState(branch_id=branch_id, label=label, summary=f"Branch about {label}.")
            self.branches[branch_id] = branch
        return branch

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class ExplorationTurn:
    perception: PerceptionResult
    plan: PlanningDecision
    retrieved_facts: list[RetrievedFact]
    render_request: RenderRequest
    render_result: RenderResult
    narration: NarrationResult | None
    world_state: dict[str, Any]
