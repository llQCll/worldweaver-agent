from __future__ import annotations

import argparse
import copy
import html
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
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
    OpenAIImageRenderingBackend,
    PerceptionAgent,
    PerceptionResult,
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
)

logger = logging.getLogger("worldweaver_openai_web_demo")


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
                "intent_confirmation_pending": True,
                "render_skipped": True,
            },
        },
        "narration": {
            "status": "skipped",
            "error": None,
            "story_text": "",
            "summary_text": image.summary or "",
            "caption_text": "",
            "metadata": {"intent_confirmation_pending": True},
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


class ConfirmIntentRequest(BaseModel):
    option_index: int | None = None
    user_hint: str | None = None


class FeedbackRequest(BaseModel):
    page_id: str | None = None
    feedback_type: str
    label: str
    axes: dict[str, float] = {}
    note: str | None = None


class NextRoundRequest(BaseModel):
    prompt: str


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
    trajectory_log_dir: Path | None = None
    preview_image: ImageFrame | None = None
    pending_confirmation: PendingIntentConfirmation | None = None


def create_trajectory_log_dir(session_id: str) -> Path:
    log_dir = PROJECT_ROOT / "user_trajectory_logs" / session_id
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _format_markdown_value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, int | bool):
        return str(value)
    if isinstance(value, dict | list | tuple):
        return "\n```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"
    return str(value)


def append_trajectory_event(
    runtime: SessionRuntime,
    event_name: str,
    fields: dict[str, Any],
) -> None:
    if runtime.trajectory_log_dir is None:
        return
    runtime.trajectory_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime.trajectory_log_dir / "trajectory.md"
    lines = [f"\n## {_now_iso()} - {event_name}\n"]
    for key, value in fields.items():
        lines.append(f"- {key}: {_format_markdown_value(value)}\n")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.writelines(lines)


def _serialize_runtime_signals(runtime: SessionRuntime) -> list[dict[str, Any]]:
    return [
        {
            "axis": signal.axis,
            "label": signal.label,
            "score": round(signal.score, 3),
            "evidence": list(signal.evidence),
        }
        for signal in sorted(
            runtime.memory.state.user_preference_signals.values(),
            key=lambda item: abs(item.score),
            reverse=True,
        )
    ]


def _top_signal_labels(signals: list[dict[str, Any]], axes: set[str], polarity: int = 1) -> list[str]:
    selected = []
    for signal in signals:
        if signal["axis"] not in axes:
            continue
        score = float(signal.get("score") or 0)
        if polarity > 0 and score <= 0.05:
            continue
        if polarity < 0 and score >= -0.05:
            continue
        selected.append(f"{signal['label']} ({score:+.2f})")
    return selected[:4]


def _build_chinese_profile_summary(signals: list[dict[str, Any]], fallback: str) -> str:
    positive = [item for item in signals if float(item.get("score") or 0) > 0.05]
    negative = [item for item in signals if float(item.get("score") or 0) < -0.05]
    if not positive and not negative:
        if fallback and not fallback.lower().startswith("the user"):
            return fallback
        return "用户刚开始探索这个互动世界，画像仍在形成中。"

    exploration_axes = {"exploration_drive", "novelty_surprise", "mystery_tension", "detail_orientation"}
    narrative_axes = {"narrative_alignment", "continuity", "pacing_preference", "character_focus", "world_focus", "social_focus"}
    emotional_axes = {"affective_alignment", "warmth_safety", "emotional_intensity", "comfort_with_ambiguity", "closure_need"}
    parts = []
    exploration = [item["label"] for item in positive if item["axis"] in exploration_axes][:3]
    narrative = [item["label"] for item in positive if item["axis"] in narrative_axes][:3]
    emotional = [item["label"] for item in positive if item["axis"] in emotional_axes][:3]
    if exploration:
        parts.append("探索方式偏向" + "、".join(exploration))
    if narrative:
        parts.append("叙事偏好偏向" + "、".join(narrative))
    if emotional:
        parts.append("情绪体验偏向" + "、".join(emotional))
    if negative:
        parts.append("对" + "、".join(item["label"] for item in negative[:2]) + "耐受较低")
    return "用户" + "；".join(parts) + "。"


def build_user_analysis(runtime: SessionRuntime) -> dict[str, Any]:
    state = runtime.memory.state
    signals = _serialize_runtime_signals(runtime)
    narrative_axes = {
        "narrative_alignment",
        "continuity",
        "character_focus",
        "world_focus",
        "mystery_tension",
        "novelty_surprise",
        "pacing_preference",
        "detail_orientation",
    }
    emotional_axes = {
        "affective_alignment",
        "warmth_safety",
        "emotional_intensity",
        "comfort_with_ambiguity",
        "closure_need",
        "mystery_tension",
    }
    personality_axes = {
        "agency_preference",
        "exploration_drive",
        "detail_orientation",
        "social_focus",
        "comfort_with_ambiguity",
        "closure_need",
        "novelty_surprise",
    }
    resistance = _top_signal_labels(
        signals,
        narrative_axes | emotional_axes | personality_axes | {"visual_alignment", "overall_alignment"},
        polarity=-1,
    )
    story_progress = {
        "pages_generated": max(0, len(state.history) - 1),
        "story_outline_revision": state.story_outline_revision,
        "current_arc": state.story_outline or state.world_summary,
        "current_world_summary": state.world_summary,
    }
    profile_summary = _build_chinese_profile_summary(signals, state.user_profile_summary)
    return {
        "summary": profile_summary,
        "narrative_preferences": _top_signal_labels(signals, narrative_axes),
        "emotional_tendencies": _top_signal_labels(signals, emotional_axes),
        "personality_style": _top_signal_labels(signals, personality_axes),
        "resistance_or_low_alignment": resistance,
        "story_progress": story_progress,
        "confidence_note": (
            "这是基于点击与反馈推断出的创作型互动画像，可用于个性化生成；"
            "它不是临床诊断，也不代表固定不变的人格结论。"
        ),
        "signals": signals,
    }


def _page_window(runtime: SessionRuntime, window_size: int = 5) -> list[Any]:
    state = runtime.memory.state
    generated_ids = [page_id for page_id in state.history if page_id != "root_page"]
    return [
        state.pages[page_id]
        for page_id in generated_ids[-window_size:]
        if page_id in state.pages
    ]


def build_stage_report(runtime: SessionRuntime, window_size: int = 5) -> dict[str, Any]:
    state = runtime.memory.state
    pages = _page_window(runtime, window_size=window_size)
    generated_count = max(0, len([page_id for page_id in state.history if page_id != "root_page"]))
    user_analysis = build_user_analysis(runtime)
    feedback_by_page = {
        record.page_id: record
        for record in state.user_feedback_history
    }
    dimension_data = [
        {
            "axis": signal["axis"],
            "label": signal["label"],
            "score": signal["score"],
            "evidence": signal["evidence"][-3:],
        }
        for signal in user_analysis["signals"][:12]
    ]
    page_summaries = [
        {
            "page_id": page.page_id,
            "image_uri": page.image_uri,
            "summary": page.page_summary,
            "action": page.action.value if page.action else "",
            "target_label": page.target_label,
            "feedback": feedback_by_page.get(page.page_id).label if feedback_by_page.get(page.page_id) else None,
        }
        for page in pages
    ]
    story_phrases = [page.page_summary for page in pages if page.page_summary]
    first_target = pages[0].target_label if pages else state.root_topic
    last_target = pages[-1].target_label if pages else state.root_topic
    narrative_text = (
        f"\u8fd9\u4e94\u6b21\u4e92\u52a8\u50cf\u4e00\u7ec4\u8fde\u7eed\u7684\u5206\u955c\uff1a\u4f60\u4ece\u201c{first_target}\u201d\u51fa\u53d1\uff0c"
        f"\u4e00\u8def\u628a\u6ce8\u610f\u529b\u63a8\u5411\u201c{last_target}\u201d\u3002"
        f"\u753b\u9762\u91cc\u7684\u9009\u62e9\u4e0d\u65ad\u5728\u6545\u4e8b\u63a8\u8fdb\u548c\u81ea\u6211\u504f\u597d\u4e4b\u95f4\u4ea4\u6362\u4fe1\u53f7\uff0c"
        f"\u9010\u6e10\u663e\u9732\u51fa\u4f60\u66f4\u613f\u610f\u8ffd\u968f\u600e\u6837\u7684\u7ebf\u7d22\u3001\u505c\u7559\u5728\u54ea\u7c7b\u60c5\u7eea\u91cc\u3002"
        if pages
        else "\u8fd8\u6ca1\u6709\u8db3\u591f\u7684\u51fa\u56fe\u5f62\u6210\u9636\u6bb5\u753b\u50cf\uff1b\u7ee7\u7eed\u751f\u6210\u5230\u4e94\u5f20\u540e\uff0c\u8fd9\u91cc\u4f1a\u51fa\u73b0\u4e00\u6bb5\u9636\u6bb5\u6027\u827a\u672f\u603b\u7ed3\u3002"
    )
    if story_phrases:
        narrative_text += " \u8fd9\u4e00\u9636\u6bb5\u7684\u6545\u4e8b\u7eb9\u7406\u5305\u62ec\uff1a" + " / ".join(story_phrases[-3:]) + "\u3002"

    suggestions = []
    if user_analysis["narrative_preferences"]:
        suggestions.append("\u4e0b\u4e00\u8f6e\u53ef\u4ee5\u7ee7\u7eed\u5f3a\u5316\uff1a" + "\u3001".join(user_analysis["narrative_preferences"][:3]) + "\u3002")
    if user_analysis["emotional_tendencies"]:
        suggestions.append("\u60c5\u7eea\u8868\u8fbe\u4e0a\u53ef\u4ee5\u66f4\u660e\u786e\u5730\u56de\u5e94\uff1a" + "\u3001".join(user_analysis["emotional_tendencies"][:3]) + "\u3002")
    if user_analysis["resistance_or_low_alignment"]:
        suggestions.append("\u9700\u8981\u907f\u514d\u6216\u51cf\u5f31\uff1a" + "\u3001".join(user_analysis["resistance_or_low_alignment"][:2]) + "\u3002")
    if not suggestions:
        suggestions.append("\u5efa\u8bae\u4e0b\u4e00\u8f6e\u591a\u5c1d\u8bd5\u4e0d\u540c\u7c7b\u578b\u7684\u70b9\u51fb\u548c\u53cd\u9988\uff0c\u8ba9\u753b\u50cf\u66f4\u5b8c\u6574\u3002")

    next_round_seed = (
        "\u5ef6\u7eed\u5f53\u524d\u6545\u4e8b\u7ebf\uff0c\u4f46\u8ba9\u4e0b\u4e00\u9636\u6bb5\u66f4\u5168\u9762\u5730\u6d4b\u8bd5\u7528\u6237\u504f\u597d\uff1a"
        "\u4fdd\u7559\u5f53\u524d\u4f4d\u7f6e\u548c\u70b9\u51fb\u610f\u56fe\uff0c\u4f46\u8ba9\u4e0b\u4e00\u5e27\u4e3b\u52a8\u63a2\u7d22\u4e0d\u540c\u60c5\u7eea\u4e0e\u98ce\u683c\u3002"
    )
    completed = generated_count > 0 and generated_count % window_size == 0
    return {
        "window_size": window_size,
        "generated_count": generated_count,
        "completed": completed,
        "remaining_until_next_report": 0 if completed else window_size - (generated_count % window_size),
        "title": f"\u9636\u6bb5\u753b\u50cf\u62a5\u544a\uff1a\u7b2c {max(1, (generated_count - 1) // window_size + 1)} \u8f6e",
        "artistic_summary": narrative_text,
        "dimension_data": dimension_data,
        "page_summaries": page_summaries,
        "user_analysis": user_analysis,
        "suggestions": suggestions,
        "next_round_seed": next_round_seed,
    }

def write_profile_snapshot(runtime: SessionRuntime) -> None:
    if runtime.trajectory_log_dir is None:
        return
    state = runtime.memory.state
    user_analysis = build_user_analysis(runtime)
    stage_report = build_stage_report(runtime)
    history = [
        {
            "page_id": record.page_id,
            "feedback_type": record.feedback_type,
            "label": record.label,
            "created_at": record.created_at,
            "axes": dict(record.axes),
            "note": record.note,
            "plan_action": record.plan_action,
            "target_label": record.target_label,
        }
        for record in state.user_feedback_history
    ]
    snapshot = {
        "session_id": runtime.session_id,
        "updated_at": _now_iso(),
        "topic": state.root_topic,
        "user_profile_summary": state.user_profile_summary,
        "user_analysis": user_analysis,
        "stage_report": stage_report,
        "user_preference_signals": user_analysis["signals"],
        "user_feedback_history": history,
    }
    profile_path = runtime.trajectory_log_dir / "profile_snapshot.json"
    profile_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def _feedback_option(
    feedback_type: str,
    label: str,
    description: str,
    axes: dict[str, float],
) -> dict[str, Any]:
    return {
        "type": feedback_type,
        "feedback_type": feedback_type,
        "label": label,
        "description": description,
        "axes": axes,
    }


def build_personalized_feedback(runtime: SessionRuntime) -> dict[str, Any]:
    state = runtime.memory.state
    active_image = runtime.preview_image or runtime.current_image
    page = state.pages.get(active_image.page_id)
    plan = page.plan if page else None
    perception = page.perception if page else None
    target = (
        getattr(perception, "target_label", None)
        or getattr(page, "target_label", None)
        or active_image.summary
        or state.root_topic
        or "\u8fd9\u4e00\u5e27"
    )
    action = getattr(plan, "action", None)
    action_value = action.value if hasattr(action, "value") else str(action or "")
    mood = (
        getattr(perception, "emotion_hint", None)
        or getattr(plan, "narration_style", None)
        or ""
    )
    summary = getattr(page, "page_summary", None) or active_image.summary or ""
    signals = sorted(
        state.user_preference_signals.values(),
        key=lambda item: abs(item.score),
        reverse=True,
    )
    dominant_signal = signals[0] if signals and abs(signals[0].score) >= 0.08 else None

    prompt_parts = [f"\u8fd9\u5f20\u56fe\u56f4\u7ed5\u201c{target}\u201d\u63a8\u8fdb\u3002"]
    if summary:
        prompt_parts.append(f"\u5f53\u524d\u7ed3\u679c\uff1a{summary}")
    if state.user_profile_summary:
        prompt_parts.append(f"\u6211\u4f1a\u7ed3\u5408\u4f60\u7684\u753b\u50cf\uff1a{state.user_profile_summary}")
    if mood:
        prompt_parts.append(f"\u8fd9\u6b21\u4e5f\u60f3\u786e\u8ba4\u6c1b\u56f4\u662f\u5426\u8d34\u8fd1\u4f60\u8981\u7684\u201c{mood}\u201d\u3002")
    prompt = " ".join(prompt_parts)

    options = [
        _feedback_option(
            "aligned_current_panel",
            f"\u559c\u6b22\uff0c\u7ee7\u7eed\u6cbf\u7740\u201c{target}\u201d\u63a8\u8fdb",
            "\u5f3a\u5316\u5f53\u524d\u70b9\u51fb\u610f\u56fe\u3001\u53d9\u4e8b\u65b9\u5411\u3001\u753b\u9762\u98ce\u683c\u548c\u89d2\u8272/\u573a\u666f\u8fde\u7eed\u6027\u3002",
            {
                "overall_alignment": 1.0,
                "narrative_alignment": 0.65,
                "visual_alignment": 0.45,
                "continuity": 0.4,
            },
        )
    ]

    if action_value == ExplorationAction.REVEAL.value:
        options.append(
            _feedback_option(
                "mystery_level_right",
                "\u7ec6\u8282\u5c55\u5f00\u5f97\u6709\u5438\u5f15\u529b",
                "\u5f3a\u5316\u9690\u85cf\u7ebf\u7d22\u3001\u60ac\u5ff5\u548c\u60ca\u559c\uff0c\u4f46\u4e0d\u4e00\u6b21\u6027\u8bb2\u900f\u3002",
                {
                    "mystery_tension": 0.9,
                    "novelty_surprise": 0.55,
                    "narrative_alignment": 0.35,
                    "comfort_with_ambiguity": 0.35,
                    "exploration_drive": 0.25,
                },
            )
        )
    elif action_value == ExplorationAction.BRANCH_OUT.value:
        options.append(
            _feedback_option(
                "branch_interest",
                "\u7ec6\u8282\u5c55\u5f00\u5f97\u6709\u5438\u5f15\u529b",
                "\u5f3a\u5316\u7531\u5f53\u524d\u5bf9\u8c61\u5ef6\u4f38\u51fa\u7684\u65b0\u5730\u70b9\u3001\u65b0\u4eba\u7269\u6216\u65b0\u4e8b\u4ef6\u3002",
                {
                    "world_focus": 0.8,
                    "novelty_surprise": 0.55,
                    "narrative_alignment": 0.35,
                    "exploration_drive": 0.55,
                    "agency_preference": 0.25,
                },
            )
        )
    elif action_value == ExplorationAction.ZOOM_IN.value:
        options.append(
            _feedback_option(
                "detail_focus",
                "\u7ec6\u8282\u5c55\u5f00\u5f97\u6709\u5438\u5f15\u529b",
                "\u5f3a\u5316\u5c40\u90e8\u89c2\u5bdf\u3001\u89d2\u8272/\u7269\u4ef6\u7ec6\u8282\uff0c\u4ee5\u53ca\u66f4\u8fd1\u8ddd\u79bb\u7684\u89c6\u89c9\u53d9\u4e8b\u3002",
                {
                    "character_focus": 0.55,
                    "visual_alignment": 0.5,
                    "continuity": 0.35,
                    "detail_orientation": 0.55,
                    "closure_need": 0.2,
                },
            )
        )
    else:
        options.append(
            _feedback_option(
                "transition_continuity",
                "\u8fd9\u4e2a\u8f6c\u573a\u548c\u4e0a\u4e00\u5e27\u8fde\u5f97\u4e0a",
                "\u5f3a\u5316\u6545\u4e8b\u8fde\u7eed\u3001\u955c\u5934\u8c03\u5ea6\u548c\u4e0a\u4e0b\u6587\u627f\u63a5\u3002",
                {
                    "continuity": 0.8,
                    "narrative_alignment": 0.55,
                    "visual_alignment": 0.25,
                    "pacing_preference": 0.3,
                },
            )
        )

    if dominant_signal is not None and dominant_signal.score > 0:
        options.append(
            _feedback_option(
                "lean_into_profile",
                f"\u7ee7\u7eed\u5f3a\u5316\u6211\u7684\u504f\u597d\uff1a{dominant_signal.label}",
                "\u628a\u5f53\u524d\u7528\u6237\u753b\u50cf\u91cc\u6700\u660e\u663e\u7684\u6b63\u5411\u504f\u597d\u66f4\u660e\u786e\u5730\u5e26\u5165\u540e\u7eed\u753b\u9762\u3002",
                {
                    dominant_signal.axis: 0.85,
                    "overall_alignment": 0.35,
                },
            )
        )
    elif dominant_signal is not None:
        options.append(
            _feedback_option(
                "avoid_profile_mismatch",
                f"\u5c11\u4e00\u70b9\uff1a{dominant_signal.label}",
                "\u4fdd\u7559\u5f53\u524d\u4f4d\u7f6e\u548c\u70b9\u51fb\u610f\u56fe\uff0c\u4f46\u8ba9\u4e0b\u4e00\u5e27\u4e3b\u52a8\u63a2\u7d22\u4e0d\u540c\u60c5\u7eea\u4e0e\u98ce\u683c\u3002",
                {
                    dominant_signal.axis: -0.85,
                    "overall_alignment": -0.25,
                },
            )
        )
    else:
        options.append(
            _feedback_option(
                "change_mood",
                "\u6211\u60f3\u6362\u4e00\u79cd\u6c1b\u56f4",
                "\u4fdd\u7559\u5f53\u524d\u4f4d\u7f6e\u548c\u70b9\u51fb\u610f\u56fe\uff0c\u4f46\u8ba9\u4e0b\u4e00\u5e27\u4e3b\u52a8\u63a2\u7d22\u4e0d\u540c\u60c5\u7eea\u4e0e\u98ce\u683c\u3002",
                {
                    "affective_alignment": -0.55,
                    "novelty_surprise": 0.45,
                    "visual_alignment": -0.25,
                    "emotional_intensity": 0.35,
                },
            )
        )

    options.append(
        _feedback_option(
            "missed_expectation",
            "这张图偏离了我的期待",
            "保留用户点击轨迹，但降低当前叙事/视觉方向权重，下一帧重新校准。",
            {
                "overall_alignment": -0.75,
                "narrative_alignment": -0.55,
                "visual_alignment": -0.35,
                "continuity": -0.2,
                "closure_need": 0.25,
            },
        )
    )

    return {
        "feedback_prompt": prompt,
        "feedback_options": options[:4],
    }

def summarize_render_token_usage(state: Any) -> dict[str, Any]:
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    render_pages_with_usage = 0

    for page in state.pages.values():
        metadata = getattr(page, "render_metadata", {}) or {}
        usage = metadata.get("usage") or {}
        page_input = usage.get("input_tokens")
        page_output = usage.get("output_tokens")
        page_total = usage.get("total_tokens")
        if page_input is None and page_output is None and page_total is None:
            continue
        render_pages_with_usage += 1
        total_input_tokens += int(page_input or 0)
        total_output_tokens += int(page_output or 0)
        total_tokens += int(page_total or 0)

    return {
        "render_pages_with_usage": render_pages_with_usage,
        "render_input_tokens": total_input_tokens,
        "render_output_tokens": total_output_tokens,
        "render_total_tokens": total_tokens,
    }


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
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simplified OpenAI-image web demo for the WorldWeaver agent.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--frontend-title", default="WorldWeaver OpenAI")
    parser.add_argument("--planning-backend", choices=("rule", "llm"), default="llm")
    parser.add_argument("--llm-endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--llm-model", default="Qwen3_8B")
    parser.add_argument("--llm-api-key", default="EMPTY")
    parser.add_argument("--llm-timeout-seconds", type=int, default=60)
    parser.add_argument("--llm-max-completion-tokens", type=int, default=2048)
    parser.add_argument("--llm-json-retry-count", type=int, default=1)
    parser.add_argument("--perception-backend", choices=("rule", "llm"), default="llm")
    parser.add_argument("--perception-llm-endpoint", default="http://127.0.0.1:8001")
    parser.add_argument("--perception-llm-model", default="Qwen3_VL_8B")
    parser.add_argument("--perception-llm-api-key", default="EMPTY")
    parser.add_argument("--perception-llm-timeout-seconds", type=int, default=60)
    parser.add_argument("--perception-llm-max-completion-tokens", type=int, default=2048)
    parser.add_argument("--perception-llm-json-retry-count", type=int, default=1)
    parser.add_argument("--narration-backend", choices=("rule", "llm"), default="llm")
    parser.add_argument("--narration-llm-endpoint", default=None)
    parser.add_argument("--narration-llm-model", default=None)
    parser.add_argument("--narration-llm-api-key", default=None)
    parser.add_argument("--narration-llm-timeout-seconds", type=int, default=60)
    parser.add_argument("--rendering-backend", choices=("mock", "openai_image"), default="openai_image")
    parser.add_argument("--openai-image-api-key", default=None)
    parser.add_argument("--openai-image-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--openai-image-model", default="gpt-image-2")
    parser.add_argument("--openai-image-output-dir", default="output_openai")
    parser.add_argument("--openai-image-size", default="1024x1024")
    parser.add_argument("--openai-image-quality", default="high")
    parser.add_argument("--openai-image-background", default="auto")
    parser.add_argument("--openai-image-input-fidelity", default="high")
    parser.add_argument("--openai-image-timeout-seconds", type=int, default=180)
    parser.add_argument("--openai-image-connect-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--openai-image-read-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--openai-image-max-retries", type=int, default=2)
    parser.add_argument("--openai-image-max-reference-images", type=int, default=4)
    return parser.parse_args()


def build_planning_agent(args: argparse.Namespace) -> PlanningAgent:
    if args.planning_backend == "rule":
        return PlanningAgent(RuleBasedPlanningBackend())
    planning_backend = OpenAICompatibleJsonBackend(
        endpoint=args.llm_endpoint,
        model=args.llm_model,
        api_key=args.llm_api_key,
        timeout_seconds=args.llm_timeout_seconds,
        max_completion_tokens=args.llm_max_completion_tokens,
        json_retry_count=args.llm_json_retry_count,
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
        max_completion_tokens=args.perception_llm_max_completion_tokens,
        json_retry_count=args.perception_llm_json_retry_count,
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
        max_completion_tokens=args.llm_max_completion_tokens,
        json_retry_count=args.llm_json_retry_count,
    )
    return NarrationAgent(LlmNarrationBackend(reasoning_backend=narration_backend))


def build_rendering_agent(args: argparse.Namespace) -> RenderingAgent:
    if args.rendering_backend == "mock":
        return RenderingAgent(MockRenderingBackend())
    api_key = args.openai_image_api_key
    if not api_key:
        raise ValueError("openai_image rendering backend requires --openai-image-api-key.")
    return RenderingAgent(
        OpenAIImageRenderingBackend(
            api_key=api_key,
            base_url=args.openai_image_base_url,
            model=args.openai_image_model,
            output_dir=args.openai_image_output_dir,
            size=args.openai_image_size,
            quality=args.openai_image_quality,
            background=args.openai_image_background,
            input_fidelity=args.openai_image_input_fidelity,
            timeout_seconds=args.openai_image_timeout_seconds,
            connect_timeout_seconds=args.openai_image_connect_timeout_seconds,
            read_timeout_seconds=args.openai_image_read_timeout_seconds,
            max_retries=args.openai_image_max_retries,
            max_reference_images=args.openai_image_max_reference_images,
        )
    )


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

    if args.rendering_backend == "openai_image":
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
            render_metadata=root_result.metadata,
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
        trajectory_log_dir=create_trajectory_log_dir(memory.state.session_id),
        checkpoints=[
            SessionCheckpoint(
                label="root_page",
                memory_state=copy.deepcopy(memory.state),
                image=copy.deepcopy(root_image),
            )
        ],
    )


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="WorldWeaver OpenAI Web Demo")
    sessions: dict[str, SessionRuntime] = {}
    frontend_path = PROJECT_ROOT / "web" / "index_openai.html"
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
            "planning_backend": args.planning_backend,
            "perception_backend": args.perception_backend,
            "narration_backend": args.narration_backend,
            "rendering_backend": args.rendering_backend,
            "rendering_model": args.openai_image_model if args.rendering_backend == "openai_image" else "mock",
            "rendering_base_url": args.openai_image_base_url if args.rendering_backend == "openai_image" else None,
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
        append_trajectory_event(
            runtime,
            "session_created",
            {
                "topic": request.topic,
                "style_guide": request.style_guide,
                "story_outline_source": runtime.memory.state.story_outline_source,
                "story_outline": runtime.memory.state.story_outline,
                "root_page_id": runtime.current_image.page_id,
                "root_image_uri": runtime.current_image.image_uri,
                "root_prompt": runtime.current_image.prompt,
                "root_summary": runtime.current_image.summary,
                "log_dir": str(runtime.trajectory_log_dir),
            },
        )
        write_profile_snapshot(runtime)
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
        click = ClickEvent(x=request.x, y=request.y, user_hint=request.user_hint)
        perception, plan = runtime.explorer.analyze_intent(image=source_image, click=click)
        runtime.pending_confirmation = PendingIntentConfirmation(
            source_image=source_image,
            click=click,
            perception=perception,
            plan=plan,
        )
        append_trajectory_event(
            runtime,
            "click_analyzed",
            {
                "source_page_id": source_image.page_id,
                "click_x": click.x,
                "click_y": click.y,
                "user_hint": click.user_hint,
                "target_label": perception.target_label,
                "region_caption": perception.region_caption,
                "interaction_reason": perception.interaction_reason,
                "suggested_story_direction": perception.suggested_story_direction,
                "intent_options": [
                    {
                        "index": index,
                        "action": option.action.value,
                        "label": option.label,
                        "description": option.description,
                        "confidence": option.confidence,
                        "target_label": option.target_label,
                    }
                    for index, option in enumerate(perception.intent_options)
                ],
                "requires_user_confirmation": perception.requires_user_confirmation,
            },
        )
        payload = serialize_session(runtime)
        payload["latest_turn"] = serialize_intent_turn(
            image=source_image,
            perception=perception,
            plan=plan,
        )
        payload["pending_intent_confirmation"] = True
        return payload

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
        payload = render_confirmed_turn(
            runtime,
            source_image=pending.source_image,
            click=pending.click,
            perception=perception,
            plan=plan,
            background_tasks=background_tasks,
        )
        append_trajectory_event(
            runtime,
            "intent_confirmed_and_rendered",
            {
                "source_page_id": pending.source_image.page_id,
                "confirmed_option_index": request.option_index,
                "confirmed_user_hint": request.user_hint,
                "accepted_intent": perception.interaction_intent,
                "target_label": perception.target_label,
                "plan_action": plan.action.value,
                "next_story_beat": plan.next_story_beat,
                "story_purpose": plan.story_purpose,
                "rendered_page_id": runtime.current_image.page_id,
                "image_uri": runtime.current_image.image_uri,
                "page_summary": runtime.current_image.summary,
                "render_prompt": runtime.current_image.prompt,
                "feedback_prompt": payload.get("feedback_prompt"),
                "feedback_options": payload.get("feedback_options"),
                "user_profile_summary": runtime.memory.state.user_profile_summary,
            },
        )
        write_profile_snapshot(runtime)
        return payload

    @app.post("/api/sessions/{session_id}/feedback")
    def submit_feedback(session_id: str, request: FeedbackRequest) -> dict[str, Any]:
        runtime = sessions.get(session_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="Unknown session id.")
        page_id = request.page_id or runtime.current_image.page_id
        if page_id not in runtime.memory.state.pages:
            raise HTTPException(status_code=404, detail="Unknown page id.")
        record = runtime.memory.record_user_feedback(
            page_id=page_id,
            feedback_type=request.feedback_type,
            label=request.label,
            axes=request.axes,
            note=request.note,
        )
        payload = serialize_session(runtime)
        append_trajectory_event(
            runtime,
            "feedback_recorded",
            {
                "page_id": record.page_id,
                "feedback_type": record.feedback_type,
                "label": record.label,
                "axes": record.axes,
                "note": record.note,
                "plan_action": record.plan_action,
                "target_label": record.target_label,
                "updated_user_profile_summary": runtime.memory.state.user_profile_summary,
                "updated_user_analysis": payload.get("user_analysis"),
                "user_preference_signals": payload.get("user_preference_signals"),
                "next_feedback_prompt": payload.get("feedback_prompt"),
                "next_feedback_options": payload.get("feedback_options"),
            },
        )
        write_profile_snapshot(runtime)
        payload["feedback"] = {
            "page_id": record.page_id,
            "feedback_type": record.feedback_type,
            "label": record.label,
            "axes": record.axes,
            "note": record.note,
        }
        return payload

    @app.post("/api/sessions/{session_id}/next-round")
    def set_next_round_prompt(session_id: str, request: NextRoundRequest) -> dict[str, Any]:
        runtime = sessions.get(session_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="Unknown session id.")
        prompt = request.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
        state = runtime.memory.state
        previous_outline = state.story_outline.strip()
        addition = f"Next round user direction: {prompt}"
        state.story_outline = f"{previous_outline}\n\n{addition}".strip() if previous_outline else addition
        state.story_outline_source = "user_next_round"
        state.story_outline_revision += 1
        append_trajectory_event(
            runtime,
            "next_round_prompt_set",
            {
                "prompt": prompt,
                "story_outline_revision": state.story_outline_revision,
                "story_outline": state.story_outline,
                "stage_report": build_stage_report(runtime),
            },
        )
        write_profile_snapshot(runtime)
        payload = serialize_session(runtime)
        payload["next_round"] = {
            "prompt": prompt,
            "message": "Next round prompt saved. Continue by clicking the current image.",
        }
        return payload

    @app.post("/api/sessions/{session_id}/restore")
    def restore_session(session_id: str, request: RestoreRequest) -> dict[str, Any]:
        runtime = sessions.get(session_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="Unknown session id.")
        if not runtime.checkpoints:
            raise HTTPException(status_code=400, detail="No previous world state is available.")
        runtime.pending_confirmation = None

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
        runtime.pending_confirmation = None
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
        return Response(content=build_mock_svg(runtime), media_type="image/svg+xml")

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
    usage = summarize_render_token_usage(state)
    personalized_feedback = build_personalized_feedback(runtime)
    user_analysis = build_user_analysis(runtime)
    stage_report = build_stage_report(runtime)
    payload = {
        "session_id": runtime.session_id,
        "topic": state.root_topic,
        "style_guide": state.style_guide,
        "world_summary": state.world_summary,
        "user_profile_summary": user_analysis["summary"],
        "user_analysis": user_analysis,
        "stage_report": stage_report,
        "user_preference_signals": [
            {
                "axis": signal.axis,
                "label": signal.label,
                "score": round(signal.score, 3),
                "evidence": list(signal.evidence),
            }
            for signal in sorted(
                state.user_preference_signals.values(),
                key=lambda item: abs(item.score),
                reverse=True,
            )
            if abs(signal.score) >= 0.05
        ][:12],
        "user_feedback_history": [
            {
                "page_id": record.page_id,
                "feedback_type": record.feedback_type,
                "label": record.label,
                "created_at": record.created_at,
                "axes": dict(record.axes),
                "note": record.note,
                "plan_action": record.plan_action,
                "target_label": record.target_label,
            }
            for record in state.user_feedback_history[-12:]
        ],
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
        "trajectory_log_dir": str(runtime.trajectory_log_dir) if runtime.trajectory_log_dir else None,
        "feedback_prompt": personalized_feedback["feedback_prompt"],
        "feedback_options": personalized_feedback["feedback_options"],
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
        "usage": usage,
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
  <text x="112" y="628" font-size="18" font-family="IBM Plex Sans, Segoe UI, sans-serif" fill="#d7e7ec">Mock image mode: the webpage can still capture clicks, but real image generation requires the openai_image backend.</text>
</svg>"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
