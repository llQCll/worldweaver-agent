from __future__ import annotations

import argparse
import copy
import html
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from worldweaver_agent import (  # noqa: E402
    ClickEvent,
    ExplorationAction,
    FluxRenderingBackend,
    ImageFrame,
    InMemoryRetrievalBackend,
    InteractionUnderstandingAgent,
    InteractiveWorldExplorer,
    LlmNarrationBackend,
    LlmPerceptionBackend,
    LlmPlanningBackend,
    MemoryAgent,
    MockRenderingBackend,
    NarrationAgent,
    OpenAICompatibleJsonBackend,
    PerceptionAgent,
    PlanningAgent,
    PlanningDecision,
    RenderRequest,
    RenderResult,
    RenderingAgent,
    RetrievalAgent,
    RetrievedFact,
    RuleBasedNarrationBackend,
    RuleBasedPerceptionBackend,
    RuleBasedPlanningBackend,
    NormalizedBox,
    PerceptionResult,
)

logger = logging.getLogger("worldweaver_web_demo")


def serialize_turn_from_page(current_page: Any) -> dict[str, Any] | None:
    if current_page is None or current_page.plan is None:
        return None
    return {
        "perception": {
            "target_label": current_page.perception.target_label,
            "region_caption": current_page.perception.region_caption,
            "action": current_page.perception.action.value,
            "action_description": current_page.perception.action_description,
            "user_profile_summary": current_page.perception.user_profile_summary,
            "intent_hint": current_page.perception.intent_hint,
            "confidence": current_page.perception.confidence,
            "clicked_region": (
                {
                    "left": current_page.perception.clicked_region.left,
                    "top": current_page.perception.clicked_region.top,
                    "width": current_page.perception.clicked_region.width,
                    "height": current_page.perception.clicked_region.height,
                }
                if current_page.perception.clicked_region is not None
                else None
            ),
            "focus_type": current_page.perception.focus_type,
            "focus_subject": current_page.perception.focus_subject,
            "story_role": current_page.perception.story_role,
            "emotion_hint": current_page.perception.emotion_hint,
            "next_panel_expectation": current_page.perception.next_panel_expectation,
            "interaction_intent": current_page.perception.interaction_intent,
            "interaction_reason": current_page.perception.interaction_reason,
            "suggested_story_direction": current_page.perception.suggested_story_direction,
            "intent_options": [
                {
                    "action": option.action.value,
                    "label": option.label,
                    "description": option.description,
                    "confidence": option.confidence,
                    "target_label": option.target_label,
                }
                for option in current_page.perception.intent_options
            ],
            "intent_confidence_threshold": current_page.perception.intent_confidence_threshold,
            "requires_user_confirmation": current_page.perception.requires_user_confirmation,
            "accepted_intent_option": (
                {
                    "action": current_page.perception.accepted_intent_option.action.value,
                    "label": current_page.perception.accepted_intent_option.label,
                    "description": current_page.perception.accepted_intent_option.description,
                    "confidence": current_page.perception.accepted_intent_option.confidence,
                    "target_label": current_page.perception.accepted_intent_option.target_label,
                }
                if current_page.perception.accepted_intent_option is not None
                else None
            ),
            "notes": current_page.perception.notes,
        },
        "plan": {
            "action": current_page.plan.action.value,
            "branch_id": current_page.plan.branch_id,
            "branch_label": current_page.plan.branch_label,
            "rationale": current_page.plan.rationale,
            "continuity_mode": current_page.plan.continuity_mode,
            "scene_location": current_page.plan.scene_location,
            "primary_actor": current_page.plan.primary_actor,
            "primary_action": current_page.plan.primary_action,
            "supporting_subject": current_page.plan.supporting_subject,
            "next_story_beat": current_page.plan.next_story_beat,
            "story_purpose": current_page.plan.story_purpose,
            "shot_type": current_page.plan.shot_type,
            "narrative_function": current_page.plan.narrative_function,
            "transition_type": current_page.plan.transition_type,
            "prompt_subject": current_page.plan.prompt_subject,
            "prompt_action": current_page.plan.prompt_action,
            "prompt_environment_change": current_page.plan.prompt_environment_change,
            "prompt_new_element": current_page.plan.prompt_new_element,
            "prompt_background_continuity": current_page.plan.prompt_background_continuity,
            "protected_entities": list(current_page.plan.protected_entities),
            "continuity_notes": current_page.plan.continuity_notes,
            "render_prompt": current_page.plan.render_prompt,
            "user_profile_summary": current_page.plan.user_profile_summary,
            "global_intent_alignment": current_page.plan.global_intent_alignment,
            "global_intent_rationale": current_page.plan.global_intent_rationale,
            "story_outline_update": current_page.plan.story_outline_update,
            "narrative_beat": current_page.plan.narrative_beat,
            "narration_brief": current_page.plan.narration_brief,
            "narration_style": current_page.plan.narration_style,
        },
        "render_result": {
            "page_id": current_page.page_id,
            "image_uri": current_page.image_uri,
            "page_summary": current_page.page_summary,
            "metadata": current_page.render_metadata,
        },
        "narration": {
            "status": current_page.narration_status,
            "error": current_page.narration_error,
            "story_text": current_page.narration.story_text if current_page.narration else "",
            "summary_text": current_page.narration.summary_text if current_page.narration else "",
            "caption_text": current_page.narration.caption_text if current_page.narration else "",
            "metadata": current_page.narration.metadata if current_page.narration else {},
        },
    }


def serialize_intent_turn(
    *,
    image: ImageFrame,
    perception: PerceptionResult,
    plan: PlanningDecision,
) -> dict[str, Any]:
    return {
        "perception": {
            "target_label": perception.target_label,
            "region_caption": perception.region_caption,
            "action": perception.action.value,
            "action_description": perception.action_description,
            "user_profile_summary": perception.user_profile_summary,
            "intent_hint": perception.intent_hint,
            "confidence": perception.confidence,
            "clicked_region": (
                {
                    "left": perception.clicked_region.left,
                    "top": perception.clicked_region.top,
                    "width": perception.clicked_region.width,
                    "height": perception.clicked_region.height,
                }
                if perception.clicked_region is not None
                else None
            ),
            "focus_type": perception.focus_type,
            "focus_subject": perception.focus_subject,
            "story_role": perception.story_role,
            "emotion_hint": perception.emotion_hint,
            "next_panel_expectation": perception.next_panel_expectation,
            "interaction_intent": perception.interaction_intent,
            "interaction_reason": perception.interaction_reason,
            "suggested_story_direction": perception.suggested_story_direction,
            "intent_options": [
                {
                    "action": option.action.value,
                    "label": option.label,
                    "description": option.description,
                    "confidence": option.confidence,
                    "target_label": option.target_label,
                }
                for option in perception.intent_options
            ],
            "intent_confidence_threshold": perception.intent_confidence_threshold,
            "requires_user_confirmation": perception.requires_user_confirmation,
            "accepted_intent_option": (
                {
                    "action": perception.accepted_intent_option.action.value,
                    "label": perception.accepted_intent_option.label,
                    "description": perception.accepted_intent_option.description,
                    "confidence": perception.accepted_intent_option.confidence,
                    "target_label": perception.accepted_intent_option.target_label,
                }
                if perception.accepted_intent_option is not None
                else None
            ),
            "notes": perception.notes,
        },
        "plan": {
            "action": plan.action.value,
            "branch_id": plan.branch_id,
            "branch_label": plan.branch_label,
            "rationale": plan.rationale,
            "continuity_mode": plan.continuity_mode,
            "scene_location": plan.scene_location,
            "primary_actor": plan.primary_actor,
            "primary_action": plan.primary_action,
            "supporting_subject": plan.supporting_subject,
            "next_story_beat": plan.next_story_beat,
            "story_purpose": plan.story_purpose,
            "shot_type": plan.shot_type,
            "narrative_function": plan.narrative_function,
            "transition_type": plan.transition_type,
            "prompt_subject": plan.prompt_subject,
            "prompt_action": plan.prompt_action,
            "prompt_environment_change": plan.prompt_environment_change,
            "prompt_new_element": plan.prompt_new_element,
            "prompt_background_continuity": plan.prompt_background_continuity,
            "protected_entities": list(plan.protected_entities),
            "continuity_notes": plan.continuity_notes,
            "render_prompt": plan.render_prompt,
            "user_profile_summary": plan.user_profile_summary,
            "global_intent_alignment": plan.global_intent_alignment,
            "global_intent_rationale": plan.global_intent_rationale,
            "story_outline_update": plan.story_outline_update,
            "narrative_beat": plan.narrative_beat,
            "narration_brief": plan.narration_brief,
            "narration_style": plan.narration_style,
        },
        "render_result": {
            "page_id": image.page_id,
            "image_uri": image.image_uri,
            "page_summary": image.summary,
            "metadata": {
                "intent_only": True,
                "render_skipped": True,
            },
        },
        "narration": {
            "status": "skipped",
            "error": None,
            "story_text": "",
            "summary_text": image.summary or "",
            "caption_text": "",
            "metadata": {"intent_only": True},
        },
    }


def apply_confirmed_intent(
    perception: PerceptionResult,
    *,
    option_index: int | None,
    user_hint: str | None,
) -> PerceptionResult:
    options = perception.intent_options or []
    option = None
    if option_index is not None and 0 <= option_index < len(options):
        option = options[option_index]
    if option is None and len(options) == 1:
        option = options[0]
    if option is None:
        return perception
    description = user_hint or option.description or option.label
    return PerceptionResult(
        target_label=option.target_label or perception.target_label,
        region_caption=perception.region_caption,
        clicked_region=perception.clicked_region,
        action=option.action,
        action_description=description,
        user_profile_summary=perception.user_profile_summary,
        intent_hint=description,
        confidence=option.confidence,
        focus_type=perception.focus_type,
        focus_subject=option.target_label or perception.focus_subject,
        story_role=perception.story_role,
        emotion_hint=perception.emotion_hint,
        next_panel_expectation=perception.next_panel_expectation,
        interaction_intent=description,
        interaction_reason=f"User confirmed intent option: {option.label}",
        suggested_story_direction=description,
        intent_options=perception.intent_options,
        intent_confidence_threshold=perception.intent_confidence_threshold,
        requires_user_confirmation=False,
        accepted_intent_option=option,
        candidate_entities=perception.candidate_entities,
        notes=[*perception.notes, "User confirmed intent before rendering."],
    )


class CreateSessionRequest(BaseModel):
    topic: str
    style_guide: str = (
        "Continuous comic panels with cinematic framing, strong character consistency, expressive lighting, "
        "and clean environmental storytelling."
    )
    root_image_path: str | None = None
    story_outline: str | None = None
    auto_generate_story_outline: bool = True


class ClickRequest(BaseModel):
    x: float
    y: float
    user_hint: str | None = None
    interaction_type: str = "click"
    selection_box: dict[str, float] | None = None
    press_duration_ms: int | None = None


class ConfirmIntentRequest(BaseModel):
    option_index: int | None = None
    user_hint: str | None = None


class RestoreRequest(BaseModel):
    mode: str = "preview"


@dataclass
class SessionCheckpoint:
    label: str
    memory_state: Any
    image: ImageFrame


@dataclass
class PendingIntentConfirmation:
    source_image: ImageFrame
    click: ClickEvent
    perception: PerceptionResult
    plan: PlanningDecision


@dataclass
class SessionRuntime:
    session_id: str
    memory: MemoryAgent
    explorer: InteractiveWorldExplorer
    current_image: ImageFrame
    retrieval_corpus: list[RetrievedFact]
    checkpoints: list[SessionCheckpoint]
    preview_image: ImageFrame | None = None
    pending_confirmation: PendingIntentConfirmation | None = None


def generate_narration_for_page(
    runtime: SessionRuntime,
    *,
    source_image: ImageFrame,
    click: ClickEvent,
    page_id: str,
    state_snapshot: Any,
    page_snapshot: Any,
) -> None:
    page = page_snapshot
    if page is None or page.plan is None:
        return

    try:
        narration = runtime.explorer.narration_agent.narrate(
            image=source_image,
            click=click,
            state=state_snapshot,
            perception=page.perception,
            plan=page.plan,
            render_result=RenderResult(
                page_id=page.page_id,
                image_uri=page.image_uri,
                revised_prompt=page.prompt,
                page_summary=page.page_summary,
                world_facts=list(page.world_facts),
                metadata=dict(page.render_metadata),
            ),
        )
        runtime.memory.attach_narration(page_id=page_id, narration=narration)
    except Exception as exc:
        logger.warning("Background narration failed for page %s: %s", page_id, exc, exc_info=True)
        try:
            fallback = RuleBasedNarrationBackend().narrate(
                image=source_image,
                click=click,
                state=state_snapshot,
                perception=page.perception,
                plan=page.plan,
                render_result=RenderResult(
                    page_id=page.page_id,
                    image_uri=page.image_uri,
                    revised_prompt=page.prompt,
                    page_summary=page.page_summary,
                    world_facts=list(page.world_facts),
                    metadata=dict(page.render_metadata),
                ),
            )
            fallback.metadata = {
                **dict(fallback.metadata),
                "sidecar_fallback": True,
                "sidecar_error": str(exc),
            }
            runtime.memory.attach_narration(page_id=page_id, narration=fallback)
        except Exception as fallback_exc:
            logger.warning("Fallback narration also failed for page %s: %s", page_id, fallback_exc, exc_info=True)
            runtime.memory.fail_narration(page_id=page_id, error=str(exc))


def schedule_root_page_narration(runtime: SessionRuntime, background_tasks: BackgroundTasks) -> None:
    current_page = runtime.memory.state.pages.get(runtime.current_image.page_id)
    if current_page is None:
        return
    runtime.memory.mark_page_narration_pending(page_id=current_page.page_id)
    state_snapshot = copy.deepcopy(runtime.memory.state)
    page_snapshot = copy.deepcopy(current_page)
    source_image = ImageFrame(
        page_id=current_page.page_id,
        image_uri=current_page.image_uri,
        prompt=current_page.prompt,
        summary=current_page.page_summary,
    )
    root_click = current_page.click
    root_plan = PlanningDecision(
        action=current_page.action,
        rationale="Root page opening narration.",
        branch_id=current_page.branch_id,
        branch_label=runtime.memory.state.branches.get(current_page.branch_id).label
        if runtime.memory.state.branches.get(current_page.branch_id)
        else "Main Branch",
        world_update="Introduce the opening world page.",
        render_prompt=current_page.prompt,
        continuity_mode="root_page",
        scene_location=runtime.memory.state.root_topic,
        primary_actor=current_page.target_label,
        primary_action="introduce the world",
        next_story_beat=current_page.page_summary,
        story_purpose="establish opening setting",
        shot_type="establishing shot",
        narrative_function="introduce_world",
        prompt_subject=current_page.target_label,
        prompt_action="establish the opening scene",
        prompt_background_continuity="opening page",
        user_profile_summary=runtime.memory.state.user_profile_summary,
        narrative_beat=current_page.page_summary,
        narration_brief="Write one concise opening paragraph that introduces the world and invites exploration.",
        narration_style="elegant, atmospheric, image-rich, and restrained",
    )
    page_snapshot.plan = root_plan
    background_tasks.add_task(
        generate_narration_for_page,
        runtime,
        source_image=source_image,
        click=root_click,
        page_id=current_page.page_id,
        state_snapshot=state_snapshot,
        page_snapshot=page_snapshot,
    )


def render_confirmed_turn(
    runtime: SessionRuntime,
    *,
    source_image: ImageFrame,
    click: ClickEvent,
    perception: PerceptionResult,
    plan: PlanningDecision,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    runtime.checkpoints.append(
        SessionCheckpoint(
            label=runtime.current_image.page_id,
            memory_state=copy.deepcopy(runtime.memory.state),
            image=copy.deepcopy(runtime.current_image),
        )
    )
    retrieved_facts = runtime.explorer.retrieval_agent.search(plan.retrieval_queries, runtime.memory.state)
    render_request = runtime.explorer.frame_planning_agent.compose_next_frame(
        image=source_image,
        interaction=perception,
        narrative_plan=plan,
        retrieved_facts=retrieved_facts,
    )
    render_result = runtime.explorer.rendering_agent.render(render_request=render_request, state=runtime.memory.state)
    runtime.memory.commit_turn_without_narration(
        source_page_id=source_image.page_id,
        click=click,
        perception=perception,
        plan=plan,
        retrieved_facts=retrieved_facts,
        render_result=render_result,
    )
    runtime.current_image = ImageFrame(
        page_id=render_result.page_id,
        image_uri=render_result.image_uri,
        prompt=render_result.revised_prompt,
        summary=render_result.page_summary,
    )
    page_snapshot = copy.deepcopy(runtime.memory.state.pages.get(render_result.page_id))
    state_snapshot = copy.deepcopy(runtime.memory.state)
    background_tasks.add_task(
        generate_narration_for_page,
        runtime,
        source_image=source_image,
        click=click,
        page_id=render_result.page_id,
        state_snapshot=state_snapshot,
        page_snapshot=page_snapshot,
    )
    payload = serialize_session(runtime)
    payload["latest_turn"] = serialize_turn_from_page(runtime.memory.state.pages.get(render_result.page_id))
    payload["intent_only"] = False
    payload["interaction_mode"] = "full"
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a clickable web demo for the WorldWeaver agent.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--interaction-mode",
        choices=("intent_only", "full"),
        default="intent_only",
        help="In intent_only mode, clicks stop after perception and planning without rendering a new page.",
    )
    parser.add_argument(
        "--planning-backend",
        choices=("rule", "llm"),
        default="rule",
    )
    parser.add_argument(
        "--llm-endpoint",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--llm-model",
        default="Qwen3_8B",
    )
    parser.add_argument(
        "--llm-api-key",
        default="EMPTY",
    )
    parser.add_argument(
        "--llm-timeout-seconds",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--narration-backend",
        choices=("rule", "llm"),
        default="llm",
    )
    parser.add_argument(
        "--perception-backend",
        choices=("rule", "llm"),
        default="llm",
    )
    parser.add_argument(
        "--perception-llm-endpoint",
        default="http://127.0.0.1:8001",
    )
    parser.add_argument(
        "--perception-llm-model",
        default="Qwen3_VL_8B",
    )
    parser.add_argument(
        "--perception-llm-api-key",
        default="EMPTY",
    )
    parser.add_argument(
        "--perception-llm-timeout-seconds",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--narration-llm-endpoint",
        default=None,
    )
    parser.add_argument(
        "--narration-llm-model",
        default=None,
    )
    parser.add_argument(
        "--narration-llm-api-key",
        default=None,
    )
    parser.add_argument(
        "--narration-llm-timeout-seconds",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--rendering-backend",
        choices=("mock", "flux"),
        default="flux",
    )
    parser.add_argument(
        "--flux-model-path",
        default="/c20250509/ZhongzhengWang/model/FLUX.1-dev",
    )
    parser.add_argument(
        "--flux-device",
        default="cuda",
    )
    parser.add_argument(
        "--flux-output-dir",
        default="output",
    )
    parser.add_argument(
        "--flux-steps",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--flux-guidance-scale",
        type=float,
        default=3.5,
    )
    parser.add_argument(
        "--flux-height",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--flux-width",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--flux-image-conditioning-mode",
        choices=("off", "img2img", "ip_adapter"),
        default="off",
    )
    parser.add_argument(
        "--flux-img2img-strength",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--flux-ip-adapter-model-path",
        default=None,
    )
    parser.add_argument(
        "--flux-ip-adapter-weight-name",
        default=None,
    )
    parser.add_argument(
        "--flux-ip-adapter-subfolder",
        default="",
    )
    parser.add_argument(
        "--flux-ip-adapter-image-encoder-path",
        default=None,
    )
    parser.add_argument(
        "--flux-ip-adapter-image-encoder-subfolder",
        default="",
    )
    parser.add_argument(
        "--flux-ip-adapter-scale",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--flux-disable-cpu-offload",
        dest="flux_disable_cpu_offload",
        action="store_true",
        help="Keep FLUX resident on GPU. This is now the default behavior.",
    )
    parser.add_argument(
        "--flux-enable-cpu-offload",
        dest="flux_disable_cpu_offload",
        action="store_false",
        help="Enable CPU offload for FLUX to save VRAM, at the cost of slower interaction.",
    )
    parser.add_argument(
        "--flux-disable-cudnn",
        action="store_true",
    )
    parser.add_argument(
        "--frontend-title",
        default="WorldWeaver",
    )
    parser.add_argument(
        "--preload-models",
        dest="preload_models",
        action="store_true",
        help="Preload rendering models during server startup. This is now the default behavior.",
    )
    parser.add_argument(
        "--lazy-load-models",
        dest="preload_models",
        action="store_false",
        help="Delay rendering model loading until first use. This reduces startup time but makes the first interaction slower.",
    )
    parser.set_defaults(preload_models=True, flux_disable_cpu_offload=True)
    return parser.parse_args()


def build_planning_agent(args: argparse.Namespace) -> PlanningAgent:
    if args.planning_backend == "rule":
        return PlanningAgent(RuleBasedPlanningBackend())

    planning_backend = OpenAICompatibleJsonBackend(
        endpoint=args.llm_endpoint,
        model=args.llm_model,
        api_key=args.llm_api_key,
        timeout_seconds=args.llm_timeout_seconds,
    )
    return PlanningAgent(LlmPlanningBackend(reasoning_backend=planning_backend))


def build_perception_agent(args: argparse.Namespace) -> PerceptionAgent:
    if args.perception_backend == "rule":
        return PerceptionAgent(InteractionUnderstandingAgent(RuleBasedPerceptionBackend()))

    perception_backend = OpenAICompatibleJsonBackend(
        endpoint=args.perception_llm_endpoint,
        model=args.perception_llm_model,
        api_key=args.perception_llm_api_key,
        timeout_seconds=args.perception_llm_timeout_seconds,
    )
    return PerceptionAgent(
        InteractionUnderstandingAgent(LlmPerceptionBackend(reasoning_backend=perception_backend))
    )


def build_narration_agent(args: argparse.Namespace) -> NarrationAgent:
    if args.narration_backend == "rule":
        return NarrationAgent(RuleBasedNarrationBackend())

    narration_backend = OpenAICompatibleJsonBackend(
        endpoint=args.narration_llm_endpoint or args.llm_endpoint,
        model=args.narration_llm_model or args.llm_model,
        api_key=args.narration_llm_api_key or args.llm_api_key,
        timeout_seconds=args.narration_llm_timeout_seconds,
    )
    return NarrationAgent(LlmNarrationBackend(reasoning_backend=narration_backend))


def build_rendering_agent(args: argparse.Namespace) -> RenderingAgent:
    if args.rendering_backend == "mock":
        return RenderingAgent(MockRenderingBackend())

    flux_backend = FluxRenderingBackend(
        model_path=args.flux_model_path,
        output_dir=args.flux_output_dir,
        device=args.flux_device,
        num_inference_steps=args.flux_steps,
        guidance_scale=args.flux_guidance_scale,
        height=args.flux_height,
        width=args.flux_width,
        image_conditioning_mode=args.flux_image_conditioning_mode,
        img2img_strength=args.flux_img2img_strength,
        ip_adapter_model_path=args.flux_ip_adapter_model_path,
        ip_adapter_weight_name=args.flux_ip_adapter_weight_name,
        ip_adapter_subfolder=args.flux_ip_adapter_subfolder,
        ip_adapter_image_encoder_path=args.flux_ip_adapter_image_encoder_path,
        ip_adapter_image_encoder_subfolder=args.flux_ip_adapter_image_encoder_subfolder,
        ip_adapter_scale=args.flux_ip_adapter_scale,
        enable_cpu_offload=not args.flux_disable_cpu_offload,
        force_disable_cudnn=args.flux_disable_cudnn,
    )
    if args.preload_models:
        flux_backend.preload_models()
    return RenderingAgent(flux_backend)


def build_retrieval_corpus(topic: str) -> list[RetrievedFact]:
    return [
        RetrievedFact(
            query=topic,
            snippet=f"{topic} often benefits from branching into subtopics and hidden context.",
            source_title="Seed Knowledge",
            source_url=None,
            confidence=0.7,
        ),
        RetrievedFact(
            query=topic,
            snippet=f"A coherent world about {topic} should preserve entities, style, and exploration history.",
            source_title="World Design Note",
            source_url=None,
            confidence=0.75,
        ),
    ]


def prepare_story_outline(
    request: CreateSessionRequest,
    *,
    planning_agent: PlanningAgent,
) -> tuple[str | None, str]:
    provided_outline = (request.story_outline or "").strip()
    if provided_outline:
        return provided_outline, "user"
    if not request.auto_generate_story_outline:
        return None, "none"
    draft = planning_agent.draft_story_outline(
        root_topic=request.topic,
        style_guide=request.style_guide,
        world_summary=f"Interactive world centered on: {request.topic}.",
    ).strip()
    return (draft or None), "planning_agent" if draft else "none"


def build_root_image_frame(
    args: argparse.Namespace,
    *,
    topic: str,
    style_guide: str,
    root_image_path: str | None,
    memory: MemoryAgent,
    rendering_agent: RenderingAgent,
) -> ImageFrame:
    prompt = f"Opening page for {topic}"
    summary = f"Entry page introducing the world of {topic}."

    if root_image_path:
        root_path = Path(root_image_path)
        if not root_path.is_file():
            raise FileNotFoundError(f"Root image path does not exist: {root_path}")
        memory.seed_root_page(
            page_id="root_page",
            image_uri=str(root_path),
            prompt=prompt,
            summary=summary,
        )
        return ImageFrame(
            page_id="root_page",
            image_uri=str(root_path),
            prompt=prompt,
            summary=summary,
        )

    if args.rendering_backend == "flux":
        root_request = RenderRequest(
            session_id=memory.state.session_id,
            branch_id="main",
            source_page_id="root_page",
            target_label=topic,
            action=ExplorationAction.REFRAME,
            render_prompt=(
                f"Create an opening WorldWeaver-style world page for {topic}. "
                f"{style_guide}. Show several visually distinct and semantically meaningful regions "
                f"that invite further exploration."
            ),
            negative_prompt="blurry, low detail, empty composition, unrelated objects",
            world_summary=memory.state.world_summary,
            branch_summary=memory.state.branches["main"].summary,
            retrieved_facts=[],
            reference_image_uri=None,
            conditioning={"stage": "root_page"},
        )
        root_result = rendering_agent.render(root_request, memory.state)
        memory.seed_root_page(
            page_id="root_page",
            image_uri=root_result.image_uri,
            prompt=root_result.revised_prompt,
            summary=root_result.page_summary,
        )
        return ImageFrame(
            page_id="root_page",
            image_uri=root_result.image_uri,
            prompt=root_result.revised_prompt,
            summary=root_result.page_summary,
        )

    memory.seed_root_page(
        page_id="root_page",
        image_uri="mock://root",
        prompt=prompt,
        summary=summary,
    )
    return ImageFrame(
        page_id="root_page",
        image_uri="mock://root",
        prompt=prompt,
        summary=summary,
    )


def create_session_runtime_with_agents(
    args: argparse.Namespace,
    request: CreateSessionRequest,
    *,
    planning_agent: PlanningAgent,
    perception_agent: PerceptionAgent,
    rendering_agent: RenderingAgent,
    narration_agent: NarrationAgent,
) -> SessionRuntime:
    story_outline, story_outline_source = prepare_story_outline(
        request,
        planning_agent=planning_agent,
    )
    memory = MemoryAgent.create(
        root_topic=request.topic,
        style_guide=request.style_guide,
        story_outline=story_outline,
        story_outline_source=story_outline_source,
    )
    retrieval_corpus = build_retrieval_corpus(request.topic)
    explorer = InteractiveWorldExplorer(
        perception_agent=perception_agent,
        memory_agent=memory,
        planning_agent=planning_agent,
        retrieval_agent=RetrievalAgent(InMemoryRetrievalBackend(retrieval_corpus)),
        rendering_agent=rendering_agent,
        narration_agent=narration_agent,
    )
    root_image = build_root_image_frame(
        args,
        topic=request.topic,
        style_guide=request.style_guide,
        root_image_path=request.root_image_path,
        memory=memory,
        rendering_agent=explorer.rendering_agent,
    )
    return SessionRuntime(
        session_id=memory.state.session_id,
        memory=memory,
        explorer=explorer,
        current_image=root_image,
        retrieval_corpus=retrieval_corpus,
        checkpoints=[
            SessionCheckpoint(
                label="root_page",
                memory_state=copy.deepcopy(memory.state),
                image=copy.deepcopy(root_image),
            )
        ],
    )


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="WorldWeaver Web Demo")
    sessions: dict[str, SessionRuntime] = {}
    frontend_path = PROJECT_ROOT / "web" / "index.html"
    shared_planning_agent = build_planning_agent(args)
    shared_perception_agent = build_perception_agent(args)
    shared_rendering_agent = build_rendering_agent(args)
    shared_narration_agent = build_narration_agent(args)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        if not frontend_path.is_file():
            raise HTTPException(status_code=500, detail="Frontend file not found.")
        return HTMLResponse(frontend_path.read_text(encoding="utf-8"))

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        return {
            "title": args.frontend_title,
            "interaction_mode": args.interaction_mode,
            "planning_backend": args.planning_backend,
            "perception_backend": args.perception_backend,
            "narration_backend": args.narration_backend,
            "rendering_backend": args.rendering_backend,
        }

    @app.post("/api/sessions")
    def create_session(request: CreateSessionRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
        runtime = create_session_runtime_with_agents(
            args,
            request,
            planning_agent=shared_planning_agent,
            perception_agent=shared_perception_agent,
            rendering_agent=shared_rendering_agent,
            narration_agent=shared_narration_agent,
        )
        sessions[runtime.session_id] = runtime
        if args.perception_backend == "llm" and (
            not runtime.current_image.image_uri or runtime.current_image.image_uri.startswith("mock://")
        ):
            logger.warning(
                "Perception backend is 'llm' but the session root image is mock. "
                "Use FLUX rendering or provide root_image_path for image-aware perception."
            )
        schedule_root_page_narration(runtime, background_tasks)
        return serialize_session(runtime)

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        runtime = sessions.get(session_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="Unknown session id.")
        return serialize_session(runtime)

    @app.post("/api/sessions/{session_id}/click")
    def click_session(session_id: str, request: ClickRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
        runtime = sessions.get(session_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="Unknown session id.")
        runtime.preview_image = None
        runtime.pending_confirmation = None
        source_image = copy.deepcopy(runtime.current_image)
        selection_box = None
        if request.selection_box is not None:
            selection_box = NormalizedBox(
                left=float(request.selection_box.get("left", 0.0)),
                top=float(request.selection_box.get("top", 0.0)),
                width=float(request.selection_box.get("width", 0.0)),
                height=float(request.selection_box.get("height", 0.0)),
            )
        click = ClickEvent(
            x=request.x,
            y=request.y,
            user_hint=request.user_hint,
            interaction_type=request.interaction_type,
            selection_box=selection_box,
            press_duration_ms=request.press_duration_ms,
        )

        if args.interaction_mode == "intent_only":
            perception, plan = runtime.explorer.analyze_intent(image=source_image, click=click)
            runtime.memory.update_story_outline_from_plan(plan)
            payload = serialize_session(runtime)
            payload["latest_turn"] = serialize_intent_turn(
                image=source_image,
                perception=perception,
                plan=plan,
            )
            payload["intent_only"] = True
            payload["interaction_mode"] = args.interaction_mode
            return payload

        perception, plan = runtime.explorer.analyze_intent(image=source_image, click=click)
        if perception.requires_user_confirmation:
            runtime.pending_confirmation = PendingIntentConfirmation(
                source_image=source_image,
                click=click,
                perception=perception,
                plan=plan,
            )
            payload = serialize_session(runtime)
            payload["latest_turn"] = serialize_intent_turn(
                image=source_image,
                perception=perception,
                plan=plan,
            )
            payload["pending_intent_confirmation"] = True
            payload["intent_only"] = True
            payload["interaction_mode"] = args.interaction_mode
            return payload

        return render_confirmed_turn(
            runtime,
            source_image=source_image,
            click=click,
            perception=perception,
            plan=plan,
            background_tasks=background_tasks,
        )

    @app.post("/api/sessions/{session_id}/confirm-intent")
    def confirm_intent(session_id: str, request: ConfirmIntentRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
        runtime = sessions.get(session_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="Unknown session id.")
        pending = runtime.pending_confirmation
        if pending is None:
            raise HTTPException(status_code=400, detail="No pending intent confirmation.")
        perception = apply_confirmed_intent(
            pending.perception,
            option_index=request.option_index,
            user_hint=request.user_hint,
        )
        plan = runtime.explorer.narrative_planning_agent.plan_next_turn(
            image=pending.source_image,
            click=pending.click,
            state=runtime.memory.state,
            interaction=perception,
        )
        runtime.pending_confirmation = None
        return render_confirmed_turn(
            runtime,
            source_image=pending.source_image,
            click=pending.click,
            perception=perception,
            plan=plan,
            background_tasks=background_tasks,
        )

    @app.post("/api/sessions/{session_id}/restore")
    def restore_session(session_id: str, request: RestoreRequest) -> dict[str, Any]:
        runtime = sessions.get(session_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="Unknown session id.")
        if not runtime.checkpoints:
            raise HTTPException(status_code=400, detail="No previous world state is available.")

        latest_checkpoint = runtime.checkpoints[-1]
        if request.mode == "preview":
            runtime.preview_image = copy.deepcopy(latest_checkpoint.image)
            payload = serialize_session(runtime)
            payload["restore"] = {
                "mode": "preview",
                "message": f"Previewing previous page: {latest_checkpoint.image.page_id}",
            }
            return payload

        if request.mode == "rollback":
            runtime.memory.state = copy.deepcopy(latest_checkpoint.memory_state)
            runtime.current_image = copy.deepcopy(latest_checkpoint.image)
            runtime.preview_image = None
            runtime.checkpoints.pop()
            payload = serialize_session(runtime)
            payload["restore"] = {
                "mode": "rollback",
                "message": f"Rolled back to page: {latest_checkpoint.image.page_id}",
            }
            return payload

        raise HTTPException(status_code=400, detail="Unsupported restore mode. Use 'preview' or 'rollback'.")

    @app.post("/api/sessions/{session_id}/resume")
    def resume_current_session(session_id: str) -> dict[str, Any]:
        runtime = sessions.get(session_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="Unknown session id.")
        runtime.preview_image = None
        payload = serialize_session(runtime)
        payload["restore"] = {
            "mode": "resume",
            "message": f"Returned to current page: {runtime.current_image.page_id}",
        }
        return payload

    @app.get("/api/sessions/{session_id}/image")
    def get_current_image(session_id: str) -> Response:
        runtime = sessions.get(session_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="Unknown session id.")

        active_image = runtime.preview_image or runtime.current_image
        image_uri = active_image.image_uri
        if image_uri:
            image_path = Path(image_uri)
            if image_path.is_file():
                return FileResponse(image_path)
        return Response(
            content=build_mock_svg(runtime),
            media_type="image/svg+xml",
        )

    @app.get("/api/sessions/{session_id}/pages/{page_id}/narration")
    def get_page_narration(session_id: str, page_id: str) -> dict[str, Any]:
        runtime = sessions.get(session_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="Unknown session id.")
        page = runtime.memory.state.pages.get(page_id)
        if page is None:
            raise HTTPException(status_code=404, detail="Unknown page id.")
        return {
            "session_id": session_id,
            "page_id": page_id,
            "narration": {
                "status": page.narration_status,
                "error": page.narration_error,
                "story_text": page.narration.story_text if page.narration else "",
                "summary_text": page.narration.summary_text if page.narration else "",
                "caption_text": page.narration.caption_text if page.narration else "",
                "metadata": page.narration.metadata if page.narration else {},
            },
        }

    return app


def serialize_session(runtime: SessionRuntime) -> dict[str, Any]:
    state = runtime.memory.state
    active_image = runtime.preview_image or runtime.current_image
    current_page_id = active_image.page_id
    current_page = state.pages.get(current_page_id)
    image_version = f"{len(state.history)}-{active_image.page_id}-{'preview' if runtime.preview_image else 'live'}"
    payload = {
        "session_id": runtime.session_id,
        "topic": state.root_topic,
        "style_guide": state.style_guide,
        "world_summary": state.world_summary,
        "user_profile_summary": state.user_profile_summary,
        "story_outline": state.story_outline,
        "story_outline_source": state.story_outline_source,
        "story_outline_revision": state.story_outline_revision,
        "current_page": {
            "page_id": active_image.page_id,
            "image_uri": active_image.image_uri,
            "prompt": active_image.prompt,
            "summary": active_image.summary,
            "image_url": f"/api/sessions/{runtime.session_id}/image?v={image_version}",
            "narration_status": current_page.narration_status if current_page else "ready",
            "narration_error": current_page.narration_error if current_page else None,
            "story_text": current_page.narration.story_text if current_page and current_page.narration else "",
            "caption_text": current_page.narration.caption_text if current_page and current_page.narration else "",
        },
        "live_page_id": runtime.current_image.page_id,
        "is_previewing": runtime.preview_image is not None,
        "pending_intent_confirmation": runtime.pending_confirmation is not None,
        "can_restore_previous": len(runtime.checkpoints) > 0,
        "history": list(state.history),
        "branches": {
            branch_id: {
                "label": branch.label,
                "summary": branch.summary,
                "page_ids": list(branch.page_ids),
            }
            for branch_id, branch in state.branches.items()
        },
        "scenes": {
            scene_id: {
                "scene_name": scene.scene_name,
                "description": scene.description,
                "branch_id": scene.branch_id,
                "first_seen_page_id": scene.first_seen_page_id,
                "last_seen_page_id": scene.last_seen_page_id,
                "page_ids": list(scene.page_ids),
                "references": [
                    {
                        "image_uri": reference.image_uri,
                        "page_id": reference.page_id,
                        "summary": reference.summary,
                        "reference_type": reference.reference_type,
                    }
                    for reference in scene.references
                ],
                "update_notes": list(scene.update_notes),
            }
            for scene_id, scene in state.scenes.items()
        },
        "current_scene_id": state.current_scene_id,
        "entities": [
            {
                "entity_id": entity.entity_id,
                "name": entity.name,
                "description": entity.description,
                "tags": list(entity.tags),
                "mentions": entity.mentions,
                "visual_signature": entity.visual_signature,
                "first_seen_page_id": entity.first_seen_page_id,
                "last_seen_page_id": entity.last_seen_page_id,
                "reference_count": len(entity.reference_bank),
                "reference_bank": [
                    {
                        "image_uri": reference.image_uri,
                        "source_page_id": reference.source_page_id,
                        "caption": reference.caption,
                        "reference_type": reference.reference_type,
                    }
                    for reference in entity.reference_bank
                ],
            }
            for entity in state.entities.values()
        ],
        "page_summary": current_page.page_summary if current_page else active_image.summary,
        "latest_turn": serialize_turn_from_page(current_page),
    }
    return payload


def build_mock_svg(runtime: SessionRuntime) -> str:
    state = runtime.memory.state
    summary = runtime.current_image.summary or "No image has been rendered yet."
    prompt = runtime.current_image.prompt or state.root_topic
    history_text = " -> ".join(state.history[-4:]) or "root_page"
    safe_topic = html.escape(state.root_topic)
    safe_summary = html.escape(summary)
    safe_prompt = html.escape(prompt)
    safe_history = html.escape(history_text)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 768">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1a3a5f"/>
      <stop offset="50%" stop-color="#256f63"/>
      <stop offset="100%" stop-color="#d9923b"/>
    </linearGradient>
  </defs>
  <rect width="1024" height="768" fill="url(#bg)"/>
  <circle cx="820" cy="180" r="120" fill="rgba(255,255,255,0.12)"/>
  <circle cx="180" cy="620" r="160" fill="rgba(255,255,255,0.10)"/>
  <rect x="72" y="72" width="880" height="624" rx="36" fill="rgba(7,18,28,0.50)" stroke="rgba(255,255,255,0.20)"/>
  <text x="112" y="170" font-size="48" font-family="Space Grotesk, Avenir Next, Segoe UI, sans-serif" fill="#f8f4ea">{safe_topic}</text>
  <text x="112" y="240" font-size="24" font-family="IBM Plex Sans, Segoe UI, sans-serif" fill="#d7e7ec">{safe_summary}</text>
  <text x="112" y="340" font-size="20" font-family="IBM Plex Mono, Consolas, monospace" fill="#ffd89a">Current prompt</text>
  <text x="112" y="378" font-size="18" font-family="IBM Plex Sans, Segoe UI, sans-serif" fill="#f8f4ea">{safe_prompt}</text>
  <text x="112" y="478" font-size="20" font-family="IBM Plex Mono, Consolas, monospace" fill="#ffd89a">History</text>
  <text x="112" y="516" font-size="18" font-family="IBM Plex Sans, Segoe UI, sans-serif" fill="#f8f4ea">{safe_history}</text>
  <text x="112" y="628" font-size="18" font-family="IBM Plex Sans, Segoe UI, sans-serif" fill="#d7e7ec">Mock image mode: the webpage can still capture clicks, but real image-conditioned perception needs FLUX or a local root image.</text>
</svg>"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
