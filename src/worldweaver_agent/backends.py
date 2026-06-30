from __future__ import annotations

import base64
import gc
import json
import logging
import mimetypes
import re
import time
from contextlib import ExitStack
from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol
from urllib import error, request

logger = logging.getLogger(__name__)

DEFAULT_INTENT_CONFIDENCE_THRESHOLD = 0.7
STORY_OUTLINE_ALIGNMENT_VALUES = {"aligned", "minor_divergence", "major_divergence"}

from worldweaver_agent.schemas import (
    CandidateEntity,
    ClickEvent,
    EntityConditioning,
    ExplorationAction,
    ImageFrame,
    IntentOption,
    NarrationResult,
    NormalizedBox,
    PerceptionResult,
    PlanningDecision,
    RenderRequest,
    RenderResult,
    RetrievedFact,
    WorldState,
)

ACTION_ALIASES = {
    "approach": ExplorationAction.ZOOM_IN,
    "close_up": ExplorationAction.ZOOM_IN,
    "detail": ExplorationAction.ZOOM_IN,
    "examine": ExplorationAction.ZOOM_IN,
    "inspect": ExplorationAction.ZOOM_IN,
    "observe": ExplorationAction.ZOOM_IN,
    "zoom": ExplorationAction.ZOOM_IN,
    "branch": ExplorationAction.BRANCH_OUT,
    "enter": ExplorationAction.BRANCH_OUT,
    "enter_scene": ExplorationAction.BRANCH_OUT,
    "explore": ExplorationAction.BRANCH_OUT,
    "follow": ExplorationAction.BRANCH_OUT,
    "go": ExplorationAction.BRANCH_OUT,
    "move": ExplorationAction.BRANCH_OUT,
    "open_path": ExplorationAction.BRANCH_OUT,
    "travel": ExplorationAction.BRANCH_OUT,
    "discover": ExplorationAction.REVEAL,
    "open": ExplorationAction.REVEAL,
    "uncover": ExplorationAction.REVEAL,
    "unlock": ExplorationAction.REVEAL,
    "change_view": ExplorationAction.REFRAME,
    "continue": ExplorationAction.REFRAME,
    "pan": ExplorationAction.REFRAME,
    "shift": ExplorationAction.REFRAME,
}


class PerceptionBackend(Protocol):
    def analyze(self, image: ImageFrame, click: ClickEvent, state: WorldState) -> PerceptionResult: ...


class PlanningBackend(Protocol):
    def plan(
        self,
        *,
        image: ImageFrame,
        click: ClickEvent,
        state: WorldState,
        perception: PerceptionResult,
    ) -> PlanningDecision: ...


class RetrievalBackend(Protocol):
    def search(self, queries: list[str], state: WorldState) -> list[RetrievedFact]: ...


class RenderingBackend(Protocol):
    def render(self, render_request: RenderRequest, state: WorldState) -> RenderResult: ...


class NarrationBackend(Protocol):
    def narrate(
        self,
        *,
        image: ImageFrame,
        click: ClickEvent,
        state: WorldState,
        perception: PerceptionResult,
        plan: PlanningDecision,
        render_result: RenderResult,
    ) -> NarrationResult: ...


class JsonReasoningBackend(Protocol):
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[str] | None = None,
    ) -> dict[str, Any]: ...

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[str] | None = None,
    ) -> str: ...


def _coerce_json_response_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    text_parts.append(str(item["text"]))
                elif item.get("text"):
                    text_parts.append(str(item["text"]))
        return "\n".join(part.strip() for part in text_parts if part).strip()
    return str(content).strip()


def _strip_reasoning_blocks(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return text

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    if text:
        return text

    lowered = raw_text.lower()
    end_tag = "</think>"
    if end_tag in lowered:
        end_index = lowered.rfind(end_tag) + len(end_tag)
        return raw_text[end_index:].strip()
    return raw_text.strip()


def _find_balanced_json_object(text: str) -> str:
    start_positions = [index for index, char in enumerate(text) if char == "{"]
    for start in start_positions:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
    return ""


def _extract_json_object_text(raw_text: str) -> str:
    text = _strip_reasoning_blocks(raw_text)
    if not text:
        return ""

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    balanced = _find_balanced_json_object(text)
    if balanced:
        return balanced

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _clean_visual_fragment(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).replace("\n", " ").split()).strip(" ,.;:-")
    return text or None


def _normalize_entity_key(value: Any) -> str:
    text = _clean_visual_fragment(value)
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _dedupe_strings(values: list[Any]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, list):
            for nested in _dedupe_strings(value):
                key = _normalize_entity_key(nested)
                if key and key not in seen:
                    deduped.append(nested)
                    seen.add(key)
            continue
        text = _clean_visual_fragment(value)
        key = _normalize_entity_key(text)
        if text and key and key not in seen:
            deduped.append(text)
            seen.add(key)
    return deduped


def _derive_protected_entities(*values: Any) -> list[str]:
    generic_keys = {
        "area",
        "character",
        "clicked_area",
        "clicked_region",
        "item",
        "location",
        "main_character",
        "object",
        "person",
        "place",
        "protagonist",
        "region",
        "scene",
        "target",
        "thing",
        "world",
    }
    return [
        item
        for item in _dedupe_strings(list(values))
        if _normalize_entity_key(item) not in generic_keys
    ]


def _compose_visual_story_prompt(
    *,
    root_topic: str,
    target_label: str,
    focus_subject: str | None,
    action: ExplorationAction,
    prompt_subject: Any = None,
    prompt_action: Any = None,
    prompt_environment_change: Any = None,
    prompt_new_element: Any = None,
    prompt_background_continuity: Any = None,
    fallback_render_prompt: Any = None,
) -> str:
    subject = (
        _clean_visual_fragment(prompt_subject)
        or _clean_visual_fragment(focus_subject)
        or _clean_visual_fragment(target_label)
        or "clicked area"
    )

    default_action_map = {
        ExplorationAction.ZOOM_IN: f"show a more specific development inside {subject}",
        ExplorationAction.BRANCH_OUT: f"show a connected nearby thread that grows out from {subject}",
        ExplorationAction.REVEAL: f"show a hidden detail, clue, opening, or consequence emerging from {subject}",
        ExplorationAction.REFRAME: f"show the next visible moment continuing from {subject}",
    }
    visible_action = _clean_visual_fragment(prompt_action) or default_action_map[action]

    default_change_map = {
        ExplorationAction.ZOOM_IN: "new local details become visible in the same area",
        ExplorationAction.BRANCH_OUT: "the surrounding area expands with a linked place, object, or event",
        ExplorationAction.REVEAL: "something previously hidden becomes visible and changes the local scene",
        ExplorationAction.REFRAME: "the same area clearly changes into a new next moment instead of repeating the previous image",
    }
    environment_change = _clean_visual_fragment(prompt_environment_change) or default_change_map[action]

    new_element = _clean_visual_fragment(prompt_new_element)
    continuity = _clean_visual_fragment(prompt_background_continuity) or (
        f"same world of {root_topic}, preserve nearby environment continuity"
    )
    fallback = _clean_visual_fragment(fallback_render_prompt)

    fragments = [
        f"{subject}",
        f"{visible_action}",
        f"{environment_change}",
    ]
    if new_element:
        fragments.append(new_element)
    fragments.append(continuity)
    fragments.append("clearly new next story moment, not a duplicate image")
    if fallback:
        fragments.append(fallback)
    return ". ".join(fragment.capitalize() for fragment in fragments if fragment) + "."


def _infer_rule_continuity_mode(action: ExplorationAction, click: ClickEvent, perception: PerceptionResult) -> str:
    hint = (click.user_hint or "").lower()
    scene_shift_tokens = [
        "new scene",
        "next scene",
        "another room",
        "another place",
        "go outside",
        "go inside",
        "enter",
        "leave",
        "move to",
        "travel",
        "follow path",
        "next location",
    ]
    if any(token in hint for token in scene_shift_tokens):
        return "scene_shift"
    if action == ExplorationAction.BRANCH_OUT:
        return "scene_shift"
    if action == ExplorationAction.REFRAME and (click.x < 0.18 or click.x > 0.82 or click.y < 0.18 or click.y > 0.82):
        return "scene_shift"
    if perception.story_role in {"destination", "pathway", "exit", "portal"}:
        return "scene_shift"
    return "local_continuation"


def _normalize_interaction_type(click: ClickEvent) -> str:
    interaction_type = (click.interaction_type or "click").strip().lower()
    if interaction_type in {"box", "bbox", "box_select", "drag"}:
        return "box"
    if interaction_type in {"long_press", "longpress", "hold", "press"}:
        return "long_press"
    return "click"


def _default_interaction_region(click: ClickEvent) -> NormalizedBox:
    if click.selection_box is not None:
        return click.selection_box
    return NormalizedBox(
        left=max(0.0, click.x - 0.1),
        top=max(0.0, click.y - 0.1),
        width=0.2,
        height=0.2,
    )


def _describe_interaction_payload(click: ClickEvent) -> dict[str, Any]:
    interaction_type = _normalize_interaction_type(click)
    payload: dict[str, Any] = {
        "interaction_type": interaction_type,
        "x": click.x,
        "y": click.y,
        "user_hint": click.user_hint,
        "press_duration_ms": click.press_duration_ms,
    }
    if click.selection_box is not None:
        payload["selection_box"] = {
            "left": click.selection_box.left,
            "top": click.selection_box.top,
            "width": click.selection_box.width,
            "height": click.selection_box.height,
        }
    return payload


def _intent_option_label(action: ExplorationAction, target_label: str) -> str:
    action_labels = {
        ExplorationAction.ZOOM_IN: "Observe more closely",
        ExplorationAction.BRANCH_OUT: "Follow a connected path",
        ExplorationAction.REVEAL: "Reveal what is hidden",
        ExplorationAction.REFRAME: "Continue from a new view",
    }
    return f"{action_labels[action]}: {target_label}"


def _intent_option_description(action: ExplorationAction, target_label: str) -> str:
    descriptions = {
        ExplorationAction.ZOOM_IN: f"Inspect {target_label} in more detail and keep the next panel locally grounded.",
        ExplorationAction.BRANCH_OUT: f"Use {target_label} as a bridge into a connected place, thread, or consequence.",
        ExplorationAction.REVEAL: f"Uncover a hidden clue, mechanism, backstory, or surprise around {target_label}.",
        ExplorationAction.REFRAME: f"Keep {target_label} in continuity while shifting viewpoint or moving to the next moment.",
    }
    return descriptions[action]


def _build_intent_options(
    *,
    primary_action: ExplorationAction,
    target_label: str,
    top_k: int = 4,
) -> list[IntentOption]:
    ordered_actions = [
        primary_action,
        ExplorationAction.ZOOM_IN,
        ExplorationAction.REVEAL,
        ExplorationAction.BRANCH_OUT,
        ExplorationAction.REFRAME,
    ]
    deduped: list[ExplorationAction] = []
    for action in ordered_actions:
        if action not in deduped:
            deduped.append(action)

    confidence_by_rank = [0.78, 0.56, 0.42, 0.30]
    options: list[IntentOption] = []
    for index, action in enumerate(deduped[:top_k]):
        options.append(
            IntentOption(
                action=action,
                label=_intent_option_label(action, target_label),
                description=_intent_option_description(action, target_label),
                confidence=confidence_by_rank[min(index, len(confidence_by_rank) - 1)],
                target_label=target_label,
            )
        )
    return options


def _clamp_confidence(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _prepare_intent_decision(
    options: list[IntentOption],
    *,
    threshold: float = DEFAULT_INTENT_CONFIDENCE_THRESHOLD,
) -> tuple[list[IntentOption], bool, IntentOption | None]:
    normalized_options = [
        IntentOption(
            action=option.action,
            label=option.label,
            description=option.description,
            confidence=_clamp_confidence(option.confidence),
            target_label=option.target_label,
        )
        for option in options
    ]
    normalized_options.sort(key=lambda option: option.confidence, reverse=True)
    high_confidence_options = [
        option for option in normalized_options if option.confidence > threshold
    ]
    accepted_option = high_confidence_options[0] if len(high_confidence_options) == 1 else None
    requires_user_confirmation = accepted_option is None
    return normalized_options[:4], requires_user_confirmation, accepted_option


def _coerce_exploration_action(value: Any, fallback: ExplorationAction = ExplorationAction.ZOOM_IN) -> ExplorationAction:
    if isinstance(value, ExplorationAction):
        return value
    if value is None:
        return fallback
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return ExplorationAction(normalized)
    except ValueError:
        pass
    return ACTION_ALIASES.get(normalized, fallback)


def _infer_rule_story_outline_alignment(state: WorldState, perception: PerceptionResult) -> tuple[str | None, str | None, str | None]:
    outline = state.story_outline.strip()
    if not outline:
        return None, None, None
    intent_terms = [
        perception.target_label,
        perception.focus_subject,
        perception.interaction_intent,
        perception.intent_hint,
        perception.action.value,
    ]
    outline_lower = outline.lower()
    overlap = [
        str(term).lower()
        for term in intent_terms
        if term and str(term).lower() in outline_lower
    ]
    if overlap:
        return (
            "aligned",
            "The clicked intent overlaps the current global story outline, so the planner can advance the intended plot path.",
            outline,
        )
    return (
        "minor_divergence",
        "The clicked intent is not explicit in the global outline, so the planner keeps the outline but adds a small adaptive allowance.",
        f"{outline}\n- Adaptive note: allow a local detour around {perception.target_label} if it can still return to the main outline.",
    )


def _normalize_story_outline_alignment(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized in STORY_OUTLINE_ALIGNMENT_VALUES else None


def _serialize_user_preference_signals_for_prompt(state: WorldState) -> list[dict[str, Any]]:
    signals = getattr(state, "user_preference_signals", {}) or {}
    return [
        {
            "axis": signal.axis,
            "label": signal.label,
            "score": round(signal.score, 3),
            "evidence": list(signal.evidence[-3:]),
        }
        for signal in sorted(signals.values(), key=lambda item: abs(item.score), reverse=True)
        if abs(signal.score) >= 0.05
    ][:10]


@dataclass(frozen=True)
class OpenAICompatibleJsonBackend:
    endpoint: str
    model: str
    api_key: str | None = None
    timeout_seconds: int = 60
    max_completion_tokens: int = 2048
    json_retry_count: int = 1

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        if images:
            content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
            for data_url in images:
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_prompt})

        last_error: Exception | None = None
        for attempt in range(self.json_retry_count + 1):
            payload = {
                "model": self.model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "max_tokens": self.max_completion_tokens,
            }
            if attempt > 0:
                payload["messages"] = [
                    *messages,
                    {
                        "role": "system",
                        "content": (
                            "Your previous response was truncated or malformed JSON. "
                            "Return only one complete valid JSON object. "
                            "Do not add explanations, markdown fences, or trailing text."
                        ),
                    },
                ]
            data = self._request_chat_completion(payload)
            content = data["choices"][0]["message"]["content"]
            raw_text = _coerce_json_response_text(content)
            json_text = _extract_json_object_text(raw_text)
            if not json_text:
                last_error = RuntimeError(
                    f"LLM backend at {self.endpoint.rstrip('/')} returned empty content instead of JSON. "
                    f"Raw content: {content!r}"
                )
                continue
            try:
                return json.loads(json_text)
            except json.JSONDecodeError as exc:
                last_error = RuntimeError(
                    f"LLM backend at {self.endpoint.rstrip('/')} returned non-JSON content. "
                    f"Could not parse message content as JSON. "
                    f"Attempt {attempt + 1}/{self.json_retry_count + 1}. "
                    f"Snippet: {json_text[:500]!r}"
                )
                continue

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"LLM backend at {self.endpoint.rstrip('/')} failed to produce JSON.")

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[str] | None = None,
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        if images:
            content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
            for data_url in images:
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
        }
        data = self._request_chat_completion(payload)
        content = data["choices"][0]["message"]["content"]
        raw_text = _coerce_json_response_text(content)
        text = _strip_reasoning_blocks(raw_text)
        if not text:
            raise RuntimeError(
                f"LLM backend at {self.endpoint.rstrip('/')} returned empty content instead of text. "
                f"Raw content: {content!r}"
            )
        return text

    def _request_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(
            url=self.endpoint.rstrip("/") + "/v1/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LLM request failed with status {exc.code} at {self.endpoint.rstrip('/')}/v1/chat/completions: {message}"
            ) from exc


class RuleBasedPerceptionBackend:
    def analyze(self, image: ImageFrame, click: ClickEvent, state: WorldState) -> PerceptionResult:
        interaction_type = _normalize_interaction_type(click)
        x_bucket = "left" if click.x < 0.33 else "right" if click.x > 0.67 else "center"
        y_bucket = "top" if click.y < 0.33 else "bottom" if click.y > 0.67 else "middle"
        target_label = click.user_hint or f"{y_bucket}-{x_bucket} region"
        clicked_region = _default_interaction_region(click)
        if interaction_type == "box":
            action = ExplorationAction.ZOOM_IN
            action_description = (
                f"Zoom into the specifically selected target {target_label} and continue the comic by revealing a tighter, more precise local development."
            )
            user_profile_summary = (
                f"用户像一个精确的视觉观察者，偏好围绕“{target_label}”展开具体、扎实的细节。"
            )
            region_caption = (
                f"User box-selected a precise target in the {y_bucket} {x_bucket} area while exploring {state.root_topic}. "
                "Treat the selection as a specific object, figure, or bounded detail that should be interpreted precisely."
            )
            intent_hint = "inspect_precise_object"
            focus_type = "object"
            story_role = "clue_source"
            interaction_intent = "inspect_object"
            interaction_reason = "The box selection suggests the user wants precise grounding on a specific target."
            suggested_story_direction = (
                f"Resolve what exactly {target_label} is and let the next beat interact with that exact target."
            )
            next_panel_expectation = f"Clarify or inspect the exact selected target: {target_label}."
            emotion_hint = "curious"
            confidence = 0.72
            tags = [y_bucket, x_bucket, "precise_selection"]
        elif interaction_type == "long_press":
            action = ExplorationAction.REVEAL
            action_description = (
                f"Reveal a deeper hidden layer, backstory, clue, or concealed consequence around {target_label} rather than simply moving the scene forward."
            )
            user_profile_summary = (
                f"用户被隐藏层次和背景故事吸引，愿意耐心等待“{target_label}”附近更深的揭示。"
            )
            region_caption = (
                f"User long-pressed the {y_bucket} {x_bucket} area while exploring {state.root_topic}. "
                "Treat this as a request to dwell, deepen, or unfold hidden story layers in that region."
            )
            intent_hint = "deepen_story_layer"
            focus_type = "event"
            story_role = "story_anchor"
            interaction_intent = "discover_backstory"
            interaction_reason = "The long press suggests the user wants a deeper reveal rather than a quick local continuation."
            suggested_story_direction = (
                f"Use {target_label} as an anchor for a deeper reveal, hidden cause, or expanded narrative layer."
            )
            next_panel_expectation = f"Reveal a deeper layer, memory, hidden mechanism, or consequence around {target_label}."
            emotion_hint = "suspenseful"
            confidence = 0.68
            tags = [y_bucket, x_bucket, "deepening"]
        else:
            if click.user_hint and any(token in click.user_hint.lower() for token in ["detail", "close", "inside"]):
                action = ExplorationAction.ZOOM_IN
            elif click.user_hint and any(token in click.user_hint.lower() for token in ["story", "history", "why", "hidden", "secret"]):
                action = ExplorationAction.REVEAL
            elif click.x < 0.2 or click.x > 0.8:
                action = ExplorationAction.REFRAME
            elif click.y < 0.2:
                action = ExplorationAction.BRANCH_OUT
            else:
                action = ExplorationAction.ZOOM_IN
            if action == ExplorationAction.ZOOM_IN:
                action_description = (
                    f"Continue deeper into {target_label} with a more specific local comic-panel development."
                )
                user_profile_summary = (
                    f"用户当前偏好聚焦式局部探索，喜欢深入“{target_label}”这样的具体细节。"
                )
            elif action == ExplorationAction.BRANCH_OUT:
                action_description = (
                    f"Follow a nearby branch, adjacent place, or linked consequence growing out from {target_label}."
                )
                user_profile_summary = (
                    f"用户似乎享受分支式探索，愿意从“{target_label}”这样的目标走向旁支路径和相邻地点。"
                )
            elif action == ExplorationAction.REVEAL:
                action_description = (
                    f"Expose something newly visible, hidden, or narratively important around {target_label}."
                )
                user_profile_summary = (
                    f"用户受到发现欲驱动，常常在“{target_label}”这样的目标周围寻找隐藏或新显露的意义。"
                )
            else:
                action_description = (
                    f"Keep the story around {target_label} moving, but reframe it through a fresh nearby viewpoint or short next moment."
                )
                user_profile_summary = (
                    f"用户倾向保留场景连续性，同时希望围绕“{target_label}”从新的视角推进下一刻。"
                )
            region_caption = (
                f"User clicked the {y_bucket} {x_bucket} area of the current page while exploring {state.root_topic}. "
                "Treat this area as the place where the next story beat should continue."
            )
            intent_hint = "continue_story_from_clicked_area"
            focus_type = "region"
            story_role = "story_anchor"
            interaction_intent = "explore_clicked_area"
            interaction_reason = "The click suggests the user wants the story to continue from this part of the scene."
            suggested_story_direction = (
                f"Let the protagonist engage with or move toward {target_label} so the story can continue there."
            )
            next_panel_expectation = f"Continue the story from the {target_label} area with a visible local change or event."
            emotion_hint = "developing"
            confidence = 0.55
            tags = [y_bucket, x_bucket]
        entity = CandidateEntity(
            name=target_label,
            description=region_caption,
            region=clicked_region,
            confidence=confidence,
            tags=tags,
        )
        intent_options, requires_user_confirmation, accepted_intent_option = _prepare_intent_decision(
            _build_intent_options(
                primary_action=action,
                target_label=target_label,
            )
        )
        perception_confidence = (
            accepted_intent_option.confidence if accepted_intent_option is not None else _clamp_confidence(confidence)
        )
        return PerceptionResult(
            target_label=target_label,
            region_caption=region_caption,
            clicked_region=clicked_region,
            action=action,
            action_description=action_description,
            user_profile_summary=user_profile_summary,
            intent_hint=intent_hint,
            confidence=perception_confidence,
            focus_type=focus_type,
            focus_subject=target_label,
            story_role=story_role,
            emotion_hint=emotion_hint,
            next_panel_expectation=next_panel_expectation,
            interaction_intent=interaction_intent,
            interaction_reason=interaction_reason,
            suggested_story_direction=suggested_story_direction,
            intent_options=intent_options,
            intent_confidence_threshold=DEFAULT_INTENT_CONFIDENCE_THRESHOLD,
            requires_user_confirmation=requires_user_confirmation,
            accepted_intent_option=accepted_intent_option,
            candidate_entities=[entity],
            notes=[f"Rule-based fallback perception backend was used for interaction_type={interaction_type}."],
        )


@dataclass(frozen=True)
class LlmPerceptionBackend:
    reasoning_backend: JsonReasoningBackend

    @staticmethod
    def _encode_image(image_uri: str | None) -> list[str] | None:
        if not image_uri:
            return None
        try:
            image_path = Path(image_uri)
            if not image_path.is_file():
                return None
            raw_bytes = image_path.read_bytes()
            encoded = base64.b64encode(raw_bytes).decode("ascii")
            mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
            return [f"data:{mime_type};base64,{encoded}"]
        except Exception:
            logger.warning("Failed to encode image from %s", image_uri, exc_info=True)
            return None

    def analyze(self, image: ImageFrame, click: ClickEvent, state: WorldState) -> PerceptionResult:
        interaction_type = _normalize_interaction_type(click)
        system_prompt = (
            "You are the recognition agent for an interactive narrative image system. "
            "Your job is to infer the user's narrative intent from the interaction by looking at both the whole image and the selected local region. "
            "This system is currently focused on continuous comic and illustrated story generation. "
            "Do not write the next full scene. Do not plan the whole next panel. "
            "Instead, decide what kind of interaction the user likely wants with the clicked target and choose the exploration action directly. "
            "Examples: if the click is on a place, the user may want the protagonist to enter, approach, inspect, or travel there; "
            "if the click is on a person, the user may want dialogue, confrontation, meeting, following, or background introduction; "
            "if the click is on an object, the user may want the protagonist to pick it up, use it, unlock something with it, or discover its meaning. "
            "The interaction is not always a simple click. "
            "interaction_type=click means broad exploration of a region. "
            "interaction_type=box means the user drew a precise box and wants exact grounding on a specific object, figure, sign, clue, or localized target. "
            "interaction_type=long_press means the user wants to dwell on the region and deepen or unfold its hidden narrative layer. "
            "Always combine the spatial signal with user_hint if one is provided. "
            "Return JSON with keys: "
            "target_label, region_caption, action, action_description, user_profile_summary, intent_hint, confidence, clicked_region, "
            "focus_type, focus_subject, story_role, emotion_hint, next_panel_expectation, "
            "interaction_intent, interaction_reason, suggested_story_direction, intent_options, candidate_entities, notes. "
            "action must be exactly one of [zoom_in, branch_out, reveal, reframe]. "
            "intent_options must be a top-k list of 2 to 4 possible user-facing intent choices for the clicked target. "
            "Each intent option must contain action, label, description, confidence, and target_label. "
            "Do not put natural verbs like enter, open, follow, inspect, or discover in action; put those in interaction_intent, label, or description instead. "
            "For each intent option, confidence must be a calibrated float from 0 to 1 that estimates how likely this option matches the user's intent compared with the other plausible options. "
            "Use high confidence only when the visual target, interaction type, and user_hint clearly support that option; use closer scores when several interpretations are plausible. "
            f"The current auto-accept threshold is {DEFAULT_INTENT_CONFIDENCE_THRESHOLD:.2f}; only one option above that threshold should be treated as unambiguous. "
            "Use short labels that a user can click, such as 'Observe the door more closely' or 'Open the door'. "
            "Use zoom_in when the user wants a more specific local continuation, detail, or interior focus. "
            "Use branch_out when the click should follow a nearby linked thread, side path, adjacent place, or consequence. "
            "Use reveal when the user likely expects hidden information, a secret layer, a clue, an opening, or a narrative uncovering. "
            "Use reframe when the same local story should continue with a changed composition, nearby viewpoint, or short time-step shift rather than a deeper reveal. "
            "action_description should be a short free-form explanation of what this action means in this exact scene, written as a handoff to the planning stage. "
            "You also receive a historical one-sentence user profile summary and structured user preference signals from memory. "
            "Use them to make the 2 to 4 intent options personally relevant: preserve the user's favored mood, narrative focus, and visual taste while still respecting the clicked region. "
            "user_profile_summary should be exactly one sentence that updates that profile using the current action and click, describing the user's current exploration preference or style. "
            "focus_type should usually be one of [region, location, object, event, character, clue, mood]. "
            "Prefer region / location / object / event unless the clicked area clearly centers on a character. "
            "intent_hint should describe the broad user intent, for example inspect_place, talk_to_character, follow_character, use_object, reveal_hidden_detail, or move_into_new_scene. "
            "interaction_intent should be a direct interaction guess such as enter_scene, start_dialogue, inspect_object, use_item, follow_person, discover_backstory, or trigger_event. "
            "interaction_reason should explain why this interpretation fits the click. "
            "suggested_story_direction should be a short handoff note for the planning agent, describing what kind of story move should happen next. "
            "story_role should describe the narrative role of the clicked target, for example story_anchor, clue_source, obstacle, destination, active_event, companion, stranger, tool, or portal. "
            "next_panel_expectation should describe what kind of immediate narrative response the user likely expects next. "
            "clicked_region must contain left, top, width, height in normalized [0,1] coordinates."
        )
        user_prompt = json.dumps(
            {
                "root_topic": state.root_topic,
                "world_summary": state.world_summary,
                "user_profile_summary": state.user_profile_summary,
                "user_preference_signals": _serialize_user_preference_signals_for_prompt(state),
                "image": {
                    "page_id": image.page_id,
                    "prompt": image.prompt,
                    "summary": image.summary,
                    "width": image.width,
                    "height": image.height,
                },
                "interaction": _describe_interaction_payload(click),
            },
            ensure_ascii=False,
        )

        images = self._encode_image(image.image_uri)
        result = self.reasoning_backend.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
        )
        clicked_region = self._parse_clicked_region(result.get("clicked_region"), click)
        candidates = self._parse_candidate_entities(
            result.get("candidate_entities", []),
            fallback_region=clicked_region,
        )
        action = _coerce_exploration_action(result.get("action"), ExplorationAction.ZOOM_IN)
        target_label = str(result.get("target_label") or click.user_hint or "clicked region")
        intent_options, requires_user_confirmation, accepted_intent_option = _prepare_intent_decision(
            self._parse_intent_options(
                result.get("intent_options", []),
                fallback_action=action,
                fallback_target=target_label,
            )
        )
        if accepted_intent_option is not None:
            action = accepted_intent_option.action
        action_description = str(
            result.get("action_description")
            or f"Continue the story from {result.get('target_label') or click.user_hint or 'the clicked region'} with a {action.value} move."
        )
        if accepted_intent_option is not None:
            action_description = accepted_intent_option.description
        perception_confidence = (
            accepted_intent_option.confidence
            if accepted_intent_option is not None
            else _clamp_confidence(result.get("confidence", 0.0))
        )
        return PerceptionResult(
            target_label=target_label,
            region_caption=str(
                result.get("region_caption")
                or f"User clicked an area related to {result.get('target_label') or state.root_topic}."
            ),
            clicked_region=clicked_region,
            action=action,
            action_description=action_description,
            user_profile_summary=str(result.get("user_profile_summary") or state.user_profile_summary),
            intent_hint=str(result.get("intent_hint", "inspect_or_expand")),
            confidence=perception_confidence,
            focus_type=str(result.get("focus_type", "region")),
            focus_subject=result.get("focus_subject"),
            story_role=result.get("story_role"),
            emotion_hint=result.get("emotion_hint"),
            next_panel_expectation=result.get("next_panel_expectation"),
            interaction_intent=result.get("interaction_intent"),
            interaction_reason=result.get("interaction_reason"),
            suggested_story_direction=result.get("suggested_story_direction"),
            intent_options=intent_options,
            intent_confidence_threshold=DEFAULT_INTENT_CONFIDENCE_THRESHOLD,
            requires_user_confirmation=requires_user_confirmation,
            accepted_intent_option=accepted_intent_option,
            candidate_entities=candidates,
            notes=self._parse_notes(result.get("notes", []))
            + [f"interaction_type={interaction_type}"],
        )

    @staticmethod
    def _parse_clicked_region(raw_region: Any, click: ClickEvent) -> NormalizedBox:
        if isinstance(raw_region, dict):
            try:
                return NormalizedBox(**raw_region)
            except TypeError:
                logger.warning("Invalid clicked_region from perception model: %s", raw_region)
        return _default_interaction_region(click)

    @staticmethod
    def _parse_candidate_entities(raw_candidates: Any, fallback_region: NormalizedBox) -> list[CandidateEntity]:
        if not isinstance(raw_candidates, list):
            return []

        parsed: list[CandidateEntity] = []
        for item in raw_candidates:
            if isinstance(item, dict):
                parsed.append(
                    CandidateEntity(
                        name=str(item.get("name") or item.get("label") or "unnamed entity"),
                        description=str(item.get("description") or item.get("summary") or ""),
                        region=LlmPerceptionBackend._parse_optional_region(item.get("region"), fallback_region),
                        confidence=float(item.get("confidence", 0.0)),
                        tags=LlmPerceptionBackend._coerce_string_list(item.get("tags", [])),
                    )
                )
            elif isinstance(item, str):
                parsed.append(
                    CandidateEntity(
                        name=item,
                        description=item,
                        region=fallback_region,
                        confidence=0.3,
                        tags=[],
                    )
                )
            else:
                logger.warning("Skipping unsupported candidate entity payload: %r", item)
        return parsed

    @staticmethod
    def _parse_intent_options(
        raw_options: Any,
        *,
        fallback_action: ExplorationAction,
        fallback_target: str,
    ) -> list[IntentOption]:
        parsed: list[IntentOption] = []
        if isinstance(raw_options, list):
            for item in raw_options:
                if not isinstance(item, dict):
                    continue
                action = _coerce_exploration_action(item.get("action"), fallback_action)
                target_label = str(item.get("target_label") or fallback_target)
                parsed.append(
                    IntentOption(
                        action=action,
                        label=str(item.get("label") or _intent_option_label(action, target_label)),
                        description=str(
                            item.get("description") or _intent_option_description(action, target_label)
                        ),
                        confidence=_clamp_confidence(item.get("confidence", 0.0)),
                        target_label=target_label,
                    )
                )
        if not parsed:
            return _build_intent_options(
                primary_action=fallback_action,
                target_label=fallback_target,
            )
        if len(parsed) < 3:
            seen_actions = {option.action for option in parsed}
            for fallback in _build_intent_options(
                primary_action=fallback_action,
                target_label=fallback_target,
            ):
                if fallback.action in seen_actions:
                    continue
                parsed.append(fallback)
                seen_actions.add(fallback.action)
                if len(parsed) >= 4:
                    break
        return parsed[:4]

    @staticmethod
    def _parse_optional_region(raw_region: Any, fallback_region: NormalizedBox) -> NormalizedBox | None:
        if raw_region is None:
            return fallback_region
        if isinstance(raw_region, dict):
            try:
                return NormalizedBox(**raw_region)
            except TypeError:
                logger.warning("Invalid candidate region from perception model: %s", raw_region)
        return fallback_region

    @staticmethod
    def _coerce_string_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if value is None:
            return []
        return [str(value)]

    @staticmethod
    def _parse_notes(raw_notes: Any) -> list[str]:
        if isinstance(raw_notes, list):
            return [str(item) for item in raw_notes]
        if raw_notes is None:
            return []
        return [str(raw_notes)]


class RuleBasedPlanningBackend:
    def plan(
        self,
        *,
        image: ImageFrame,
        click: ClickEvent,
        state: WorldState,
        perception: PerceptionResult,
    ) -> PlanningDecision:
        interaction_type = _normalize_interaction_type(click)
        action = perception.action

        continuity_mode = _infer_rule_continuity_mode(action, click, perception)
        branch_label = f"{perception.target_label} branch"
        branch_id = branch_label.lower().replace(" ", "_").replace("-", "_")
        primary_actor = perception.focus_subject if perception.focus_type == "character" else "protagonist"
        supporting_subject = perception.focus_subject or perception.target_label
        protected_entities = _derive_protected_entities(
            primary_actor,
            supporting_subject,
            [candidate.name for candidate in perception.candidate_entities[:4]],
        )
        base_render_prompt = (
            f"Continue the same world about {state.root_topic}. "
            f"The next image should continue the story from the clicked area: {perception.target_label}. "
            f"Show what changes, appears, unfolds, or becomes newly visible in that area after the click. "
        )
        render_prompt = _compose_visual_story_prompt(
            root_topic=state.root_topic,
            target_label=perception.target_label,
            focus_subject=perception.focus_subject,
            action=action,
            prompt_subject=perception.focus_subject or perception.target_label,
            prompt_action=(
                f"show a hidden detail, clue, opening, or consequence around {perception.target_label}"
                if action == ExplorationAction.REVEAL
                else (
                    f"show movement from {perception.target_label} into a connected nearby place"
                    if continuity_mode == "scene_shift"
                    else f"show the next visible development around {perception.target_label}"
                )
            ),
            prompt_environment_change=(
                "the scene shifts into a connected new place while preserving story continuity"
                if continuity_mode == "scene_shift"
                else (
                    "a connected nearby place or thread opens up from the clicked area"
                    if action == ExplorationAction.BRANCH_OUT
                    else "the clicked area changes with a clear new local event"
                )
            ),
            prompt_background_continuity=f"same world of {state.root_topic}, preserve nearby environment and style continuity",
            fallback_render_prompt=base_render_prompt,
        )
        retrieval_queries = [f"{state.root_topic} {perception.target_label}"]
        if action == ExplorationAction.REVEAL:
            retrieval_queries.append(f"{state.root_topic} hidden facts about {perception.target_label}")
        if action == ExplorationAction.BRANCH_OUT:
            retrieval_queries.append(f"related topics connected to {perception.target_label}")
        global_intent_alignment, global_intent_rationale, story_outline_update = _infer_rule_story_outline_alignment(
            state,
            perception,
        )

        return PlanningDecision(
            action=action,
            rationale=f"Rule-based planner expanded the perceived {action.value} interaction into the next comic-panel beat.",
            branch_id=branch_id,
            branch_label=branch_label,
            world_update=(
                f"Advance the local story around {perception.target_label} with a clear new event, discovery, "
                "or consequence that emerges from the clicked area."
            ),
            render_prompt=render_prompt,
            negative_prompt="random unrelated objects, style drift, isolated composition",
            continuity_mode=continuity_mode,
            scene_location=perception.target_label,
            primary_actor=primary_actor,
            primary_action=(
                "inspects the precisely selected target"
                if interaction_type == "box"
                else (
                    "dwells on the selected area to uncover a deeper layer"
                    if interaction_type == "long_press"
                    else "enters or approaches the clicked place"
                )
                if perception.interaction_intent in {"enter_scene", "move_into_new_scene", "explore_clicked_area"}
                else "interacts with the clicked target"
            ),
            supporting_subject=supporting_subject,
            next_story_beat=perception.suggested_story_direction or f"The protagonist continues the story around {perception.target_label}.",
            story_purpose="advance the story from the user's selected target",
            shot_type="local_detail" if action == ExplorationAction.ZOOM_IN else "story_continuation",
            narrative_function="continue_from_clicked_area",
            transition_type="story_progression",
            prompt_subject=perception.focus_subject or perception.target_label,
            prompt_action=(
                f"show a precise close interaction with {perception.target_label}"
                if interaction_type == "box"
                else (
                    f"show a hidden detail, clue, opening, or consequence around {perception.target_label}"
                    if action == ExplorationAction.REVEAL
                    else (
                        f"show movement from {perception.target_label} into a connected nearby place"
                        if continuity_mode == "scene_shift"
                        else f"show the next visible development around {perception.target_label}"
                    )
                )
            ),
            prompt_environment_change=(
                "the scene shifts into a connected new place while preserving story continuity"
                if continuity_mode == "scene_shift"
                else (
                    "a connected nearby place or thread opens up from the clicked area"
                    if action == ExplorationAction.BRANCH_OUT
                    else "the clicked area changes with a clear new local event"
                )
            ),
            prompt_new_element=None,
            prompt_background_continuity=f"same world of {state.root_topic}, preserve nearby environment and style continuity",
            protected_entities=protected_entities,
            continuity_notes=[
                "Preserve the same world and nearby environment.",
                "The next image should visibly progress the clicked area's local story rather than only changing the camera framing.",
                "When the clicked area implies movement, exit, entry, or a linked place, allow the next image to change into a connected new scene.",
            ],
            retrieval_queries=retrieval_queries,
            style_directives=["preserve visual continuity", "keep a sense of navigable world structure"],
            user_profile_summary=(
                f"用户当前偏好围绕“{perception.target_label}”进行“{action.value.replace('_', ' ')}”式推进，并保持连环画式探索节奏。"
            ),
            global_intent_alignment=global_intent_alignment,
            global_intent_rationale=global_intent_rationale,
            story_outline_update=story_outline_update,
            narrative_beat=(
                f"A fresh development rises from {perception.target_label}, carrying the story deeper into the world of {state.root_topic}."
            ),
            narration_brief="Write one concise literary paragraph with concrete imagery, atmosphere, and forward narrative motion.",
            narration_style="elegant, atmospheric, image-rich, and restrained",
        )


@dataclass(frozen=True)
class LlmPlanningBackend:
    reasoning_backend: JsonReasoningBackend

    def draft_story_outline(self, *, root_topic: str, style_guide: str, world_summary: str) -> str:
        system_prompt = (
            "You draft concise global story outlines for interactive illustrated stories. "
            "Return JSON with exactly one key: story_outline. "
            "The outline should be 5 to 7 short beats, flexible enough for user-driven exploration, "
            "but specific enough to guide future planning decisions."
        )
        user_prompt = json.dumps(
            {
                "root_topic": root_topic,
                "style_guide": style_guide,
                "world_summary": world_summary,
            },
            ensure_ascii=False,
        )
        result = self.reasoning_backend.generate_json(system_prompt=system_prompt, user_prompt=user_prompt)
        return str(result.get("story_outline") or "").strip()

    def plan(
        self,
        *,
        image: ImageFrame,
        click: ClickEvent,
        state: WorldState,
        perception: PerceptionResult,
    ) -> PlanningDecision:
        system_prompt = (
            "You are the planning agent for an interactive narrative image system. "
            "The perception agent has already interpreted the user's click intent and selected a high-level exploration action. "
            "Your job is to turn that recognized interaction intent into the next story beat. "
            "You must take the interaction type seriously. "
            "interaction_type=click means broad regional exploration. "
            "interaction_type=box means the user wants a precise object-centered continuation. "
            "interaction_type=long_press means the user wants a deeper unfolding, hidden layer, or slower reveal from the same area. "
            "If user_hint is present, combine it with the spatial interaction instead of ignoring either signal. "
            "Plan the next panel as if you were outlining the next sentence of a visual novel or illustrated story. "
            "Focus on who is where doing what, what changes in the scene, and why this moment matters for the story. "
            "The result should feel like the next narrative development, not just a camera adjustment. "
            "You may receive a global story_outline from memory. Treat it as the story's global intent, not as an unbreakable railroad. "
            "On every turn, compare the user's current interaction intent with the global story_outline. "
            "If they are close, directly advance the outlined plot. "
            "If there is a small difference, gradually adapt the next beat and write a concise revised story_outline that preserves the main arc while incorporating the user's direction. "
            "If there is a major conflict, prioritize the user's explicit interaction for the immediate next panel, but keep the outline update conservative and explain the conflict. "
            "Keep the perceived exploration action stable unless there is an overwhelming reason to reinterpret it. "
            "The action meanings are: "
            "zoom_in = continue deeper into the clicked area and show a more specific local development, "
            "branch_out = follow a nearby thread, connected place, or consequence linked to the clicked area, "
            "reveal = expose hidden information, a new layer, or a consequence that was not visible before, "
            "reframe = keep the same local story but advance it with a changed composition, time step, or nearby viewpoint. "
            "Return JSON with keys: "
            "action, rationale, branch_id, branch_label, world_update, render_prompt, negative_prompt, "
            "continuity_mode, scene_location, primary_actor, primary_action, supporting_subject, next_story_beat, story_purpose, "
            "shot_type, narrative_function, transition_type, continuity_notes, retrieval_queries, style_directives, user_profile_summary, "
            "global_intent_alignment, global_intent_rationale, story_outline_update, "
            "prompt_subject, prompt_action, prompt_environment_change, prompt_new_element, prompt_background_continuity, protected_entities. "
            "The action field must be exactly one of [zoom_in, branch_out, reveal, reframe]; put natural verbs like enter, open, follow, inspect, or discover in primary_action, transition_type, or narrative_function instead. "
            "continuity_mode must be one of [local_continuation, scene_shift]. "
            "global_intent_alignment must be one of [aligned, minor_divergence, major_divergence] when story_outline is present, otherwise null. "
            "global_intent_rationale should briefly explain whether the current user intent advances or bends the outline. "
            "story_outline_update should be null when no update is needed; otherwise return the full revised concise outline to save back into memory. "
            "Use scene_shift when the clicked place should lead into a connected new room, street, building interior, hidden space, next location, or adjacent environment. "
            "Use local_continuation only when the next image should stay mainly in the same scene. "
            "scene_location should say where the next beat happens. "
            "primary_actor should usually be the protagonist or the main active character in this beat. "
            "primary_action should describe what that actor is doing next. "
            "supporting_subject should name the other person, object, or place involved if there is one. "
            "next_story_beat should read like a short narrative planning note with a slight novel-like feeling. "
            "story_purpose should explain the dramatic function of the beat, for example introduce stranger, reveal clue, move into new room, use object, escalate conflict, or explain backstory. "
            "render_prompt is still required, but it should be a plain visual scene description rather than abstract direction. "
            "Also fill the helper prompt fields so they can be converted into a final visual prompt template. "
            "Use simple, concrete language that image models can draw. "
            "Prefer object + visible action + environment change. "
            "Examples: "
            "'old wooden door, slowly opening inward, dark stairway becoming visible behind it'; "
            "'library bookshelf, one glowing book pulled halfway out, hidden passage opening in the wall'; "
            "'market table, fruit crates knocked over, nearby crowd turning toward the noise'. "
            "Avoid abstract phrases like emotional subtext, cinematic framing, atmosphere deepens, symbolic tension, or close-up of the protagonist. "
            "A good render_prompt should help an image model generate a clearly new next scene, not just a different crop of the old one. "
            "shot_type can still be filled, but keep it secondary. "
            "narrative_function should describe story progression, for example continue_local_event, reveal_clue, follow_consequence, open_new_path, or deepen_location. "
            "transition_type should describe story movement, for example direct_continuation, local_reveal, causal_followup, adjacent_shift, or time_step_forward. "
            "protected_entities should list the recurring characters, creatures, props, or named objects that must stay visually recognizable if they appear again. "
            "Do not include generic placeholders like protagonist, person, place, or object. "
            "narrative_beat should describe the same next panel as a short prose-oriented story beat. "
            "narration_brief should describe how the literary narration should feel in one sentence. "
            "narration_style should name a concise prose style direction. "
            "You also receive a historical one-sentence user profile summary and structured user preference signals from memory. "
            "Treat those signals as personalization constraints: lean into positive axes, avoid or soften negative axes, and make the chosen intent feel specific to this user's taste. "
            "Return user_profile_summary as exactly one updated sentence that reflects both the historical profile and the current chosen action. "
            "Keep continuity stable, but make sure the next image visibly advances the clicked area's story."
        )
        user_prompt = json.dumps(
            {
                "root_topic": state.root_topic,
                "style_guide": state.style_guide,
                "world_summary": state.world_summary,
                "user_profile_summary": state.user_profile_summary,
                "user_preference_signals": _serialize_user_preference_signals_for_prompt(state),
                "story_outline": state.story_outline,
                "story_outline_source": state.story_outline_source,
                "story_outline_revision": state.story_outline_revision,
                "current_branch_id": state.current_branch_id,
                "image": {
                    "page_id": image.page_id,
                    "prompt": image.prompt,
                    "summary": image.summary,
                },
                "interaction": _describe_interaction_payload(click),
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
                },
                "known_entities": [
                    {
                        "name": entity.name,
                        "visual_signature": entity.visual_signature,
                        "mentions": entity.mentions,
                    }
                    for entity in list(state.entities.values())[:12]
                ],
            },
            ensure_ascii=False,
        )
        result = self.reasoning_backend.generate_json(system_prompt=system_prompt, user_prompt=user_prompt)
        action = _coerce_exploration_action(result.get("action"), perception.action)
        story_outline_update = _clean_visual_fragment(result.get("story_outline_update"))
        if story_outline_update == state.story_outline:
            story_outline_update = None
        final_render_prompt = _compose_visual_story_prompt(
            root_topic=state.root_topic,
            target_label=perception.target_label,
            focus_subject=perception.focus_subject,
            action=action,
            prompt_subject=result.get("prompt_subject"),
            prompt_action=result.get("prompt_action"),
            prompt_environment_change=result.get("prompt_environment_change"),
            prompt_new_element=result.get("prompt_new_element"),
            prompt_background_continuity=result.get("prompt_background_continuity"),
            fallback_render_prompt=result.get("render_prompt"),
        )
        return PlanningDecision(
            action=action,
            rationale=result["rationale"],
            branch_id=result["branch_id"],
            branch_label=result["branch_label"],
            world_update=result["world_update"],
            render_prompt=final_render_prompt,
            negative_prompt=result.get("negative_prompt", ""),
            continuity_mode=result.get("continuity_mode"),
            scene_location=_clean_visual_fragment(result.get("scene_location")),
            primary_actor=_clean_visual_fragment(result.get("primary_actor")),
            primary_action=_clean_visual_fragment(result.get("primary_action")),
            supporting_subject=_clean_visual_fragment(result.get("supporting_subject")),
            next_story_beat=_clean_visual_fragment(result.get("next_story_beat")),
            story_purpose=_clean_visual_fragment(result.get("story_purpose")),
            shot_type=result.get("shot_type"),
            narrative_function=result.get("narrative_function"),
            transition_type=result.get("transition_type"),
            prompt_subject=_clean_visual_fragment(result.get("prompt_subject")),
            prompt_action=_clean_visual_fragment(result.get("prompt_action")),
            prompt_environment_change=_clean_visual_fragment(result.get("prompt_environment_change")),
            prompt_new_element=_clean_visual_fragment(result.get("prompt_new_element")),
            prompt_background_continuity=_clean_visual_fragment(result.get("prompt_background_continuity")),
            protected_entities=_derive_protected_entities(
                result.get("protected_entities", []),
                result.get("primary_actor"),
                result.get("supporting_subject"),
                perception.focus_subject,
                [candidate.name for candidate in perception.candidate_entities[:4]],
            ),
            continuity_notes=[str(item) for item in result.get("continuity_notes", [])],
            retrieval_queries=list(result.get("retrieval_queries", [])),
            style_directives=list(result.get("style_directives", [])),
            user_profile_summary=_clean_visual_fragment(result.get("user_profile_summary")) or perception.user_profile_summary,
            global_intent_alignment=_normalize_story_outline_alignment(result.get("global_intent_alignment")),
            global_intent_rationale=_clean_visual_fragment(result.get("global_intent_rationale")),
            story_outline_update=story_outline_update,
            narrative_beat=_clean_visual_fragment(result.get("narrative_beat")),
            narration_brief=_clean_visual_fragment(result.get("narration_brief")),
            narration_style=_clean_visual_fragment(result.get("narration_style")),
        )


@dataclass
class InMemoryRetrievalBackend:
    knowledge_base: list[RetrievedFact]

    def search(self, queries: list[str], state: WorldState) -> list[RetrievedFact]:
        lowered_queries = " ".join(query.lower() for query in queries)
        hits = [
            fact
            for fact in self.knowledge_base
            if any(token in fact.snippet.lower() or token in fact.source_title.lower() for token in lowered_queries.split())
        ]
        return hits[:5]


class RuleBasedNarrationBackend:
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
        del image, click
        target = perception.focus_subject or perception.target_label or state.root_topic
        beat = plan.narrative_beat or plan.next_story_beat or render_result.page_summary
        story_text = (
            f"Around {target}, the world of {state.root_topic} seems to lean forward. "
            f"{beat} "
            "The moment does not break from what came before so much as unfold from it, "
            "as though the page had been waiting for this quieter turn all along."
        )
        return NarrationResult(
            story_text=story_text,
            summary_text=render_result.page_summary,
            caption_text=f"{target} · {plan.action.value.replace('_', ' ')}",
            metadata={
                "mode": "rule_based",
                "narration_style": plan.narration_style or "elegant atmospheric prose",
            },
        )


@dataclass(frozen=True)
class LlmNarrationBackend:
    reasoning_backend: JsonReasoningBackend

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
        system_prompt = (
            "You are the literary narration agent for an interactive illustrated story system. "
            "Write polished story prose for the newly generated panel. "
            "Keep the prose concise, vivid, tasteful, and concrete rather than abstract or melodramatic. "
            "Return a single short paragraph only, with no JSON, no markdown, and no explanation."
        )
        user_prompt = json.dumps(
            {
                "root_topic": state.root_topic,
                "style_guide": state.style_guide,
                "world_summary": state.world_summary,
                "image": {
                    "page_id": image.page_id,
                    "prompt": image.prompt,
                    "summary": image.summary,
                },
                "click": {"x": click.x, "y": click.y, "user_hint": click.user_hint},
                "perception": {
                    "target_label": perception.target_label,
                    "focus_subject": perception.focus_subject,
                    "story_role": perception.story_role,
                    "emotion_hint": perception.emotion_hint,
                    "suggested_story_direction": perception.suggested_story_direction,
                },
                "plan": {
                    "action": plan.action.value,
                    "branch_label": plan.branch_label,
                    "next_story_beat": plan.next_story_beat,
                    "narrative_beat": plan.narrative_beat,
                    "narration_brief": plan.narration_brief,
                    "narration_style": plan.narration_style,
                    "story_purpose": plan.story_purpose,
                },
                "render_result": {
                    "page_id": render_result.page_id,
                    "page_summary": render_result.page_summary,
                    "revised_prompt": render_result.revised_prompt,
                    "world_facts": render_result.world_facts,
                },
            },
            ensure_ascii=False,
        )
        story_text = self.reasoning_backend.generate_text(system_prompt=system_prompt, user_prompt=user_prompt).strip()
        return NarrationResult(
            story_text=story_text or render_result.page_summary,
            summary_text=render_result.page_summary,
            caption_text=f"{(perception.focus_subject or perception.target_label or state.root_topic)} · {plan.action.value.replace('_', ' ')}",
            metadata={
                "mode": "llm_text",
                "narration_style": plan.narration_style or "",
            },
        )


class MockRenderingBackend:
    def __init__(self) -> None:
        self._counter = 0

    def render(self, render_request: RenderRequest, state: WorldState) -> RenderResult:
        self._counter += 1
        page_id = f"page_{self._counter:04d}"
        summary = (
            f"{render_request.action.value} on {render_request.target_label} inside the world of {state.root_topic}."
        )
        world_facts = [fact.snippet for fact in render_request.retrieved_facts[:3]]
        metadata = {
            "conditioning": render_request.conditioning,
            "style_guide": state.style_guide,
            "retrieval_count": len(render_request.retrieved_facts),
        }
        return RenderResult(
            page_id=page_id,
            image_uri=f"mock://generated/{page_id}",
            revised_prompt=render_request.render_prompt,
            page_summary=summary,
            world_facts=world_facts,
            metadata=metadata,
        )


class OpenAIImageRenderingBackend:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str = "gpt-image-2",
        output_dir: str = "output_openai",
        size: str = "1024x1024",
        quality: str = "high",
        background: str = "auto",
        input_fidelity: str = "high",
        timeout_seconds: int = 180,
        connect_timeout_seconds: float = 30.0,
        read_timeout_seconds: float | None = None,
        max_retries: int = 2,
        max_reference_images: int = 4,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.output_dir = Path(output_dir)
        self.size = size
        self.quality = quality
        self.background = background
        self.input_fidelity = input_fidelity
        self.timeout_seconds = timeout_seconds
        self.connect_timeout_seconds = connect_timeout_seconds
        self.read_timeout_seconds = read_timeout_seconds or float(timeout_seconds)
        self.max_retries = max_retries
        self.max_reference_images = max_reference_images
        self._counter = 0
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            import httpx

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    timeout=None,
                    connect=self.connect_timeout_seconds,
                    read=self.read_timeout_seconds,
                    write=self.timeout_seconds,
                    pool=self.timeout_seconds,
                ),
                max_retries=self.max_retries,
            )
        return self._client

    def render(self, render_request: RenderRequest, state: WorldState) -> RenderResult:
        self._counter += 1
        page_id = f"page_{self._counter:04d}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{page_id}.png"

        final_prompt = self._build_prompt(render_request=render_request, state=state)
        reference_paths = self._collect_reference_paths(render_request)
        start_time = time.perf_counter()

        logger.info("Rendering %s with %s via %s", page_id, self.model, self.base_url)

        response = self._invoke_image_api(
            prompt=final_prompt,
            reference_paths=reference_paths,
        )
        elapsed_seconds = round(time.perf_counter() - start_time, 3)
        image_bytes, revised_prompt = self._extract_image_payload(response=response, fallback_prompt=final_prompt)
        output_path.write_bytes(image_bytes)

        summary = (
            f"{render_request.action.value} on {render_request.target_label} "
            f"inside the world of {state.root_topic}."
        )
        world_facts = [fact.snippet for fact in render_request.retrieved_facts[:3]]
        usage = self._serialize_usage(getattr(response, "usage", None))
        metadata = {
            "conditioning": render_request.conditioning,
            "style_guide": state.style_guide,
            "retrieval_count": len(render_request.retrieved_facts),
            "model": self.model,
            "base_url": self.base_url,
            "provider": "openai-compatible-images",
            "output_path": str(output_path),
            "render_mode": "edit" if reference_paths else "generate",
            "reference_image_count": len(reference_paths),
            "reference_images": reference_paths,
            "size": self.size,
            "quality": self.quality,
            "background": self.background,
            "input_fidelity": self.input_fidelity if reference_paths else None,
            "elapsed_seconds": elapsed_seconds,
            "usage": usage,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "protected_entities": render_request.conditioning.get("protected_entities", []),
            "selected_entity_references": render_request.conditioning.get("selected_entity_references", []),
        }

        logger.info(
            "Rendered %s saved to %s. total_tokens=%s elapsed=%.3fs",
            page_id,
            output_path,
            usage.get("total_tokens"),
            elapsed_seconds,
        )

        return RenderResult(
            page_id=page_id,
            image_uri=str(output_path),
            revised_prompt=revised_prompt,
            page_summary=summary,
            world_facts=world_facts,
            metadata=metadata,
        )

    def _invoke_image_api(self, *, prompt: str, reference_paths: list[str]):
        client = self._get_client()
        try:
            if reference_paths:
                with ExitStack() as stack:
                    images = [stack.enter_context(Path(path).open("rb")) for path in reference_paths]
                    return client.images.edit(
                        model=self.model,
                        image=images,
                        prompt=prompt,
                        size=self.size,
                        quality=self.quality,
                        background=self.background,
                        input_fidelity=self.input_fidelity,
                        response_format="b64_json",
                    )
            return client.images.generate(
                model=self.model,
                prompt=prompt,
                size=self.size,
                quality=self.quality,
                background=self.background,
                response_format="b64_json",
            )
        except Exception as exc:
            raise RuntimeError(
                "OpenAI-compatible image request failed. "
                f"base_url={self.base_url}, model={self.model}, "
                f"mode={'edit' if reference_paths else 'generate'}, "
                f"connect_timeout={self.connect_timeout_seconds}s, "
                f"read_timeout={self.read_timeout_seconds}s, "
                f"max_retries={self.max_retries}. "
                f"Original error: {exc}"
            ) from exc

    def _build_prompt(self, *, render_request: RenderRequest, state: WorldState) -> str:
        sections = [
            f"Create the next panel for an interactive visual story about {state.root_topic}.",
            f"Style guide: {state.style_guide}",
            "Dual objective: make this panel advance a coherent, eventually complete comic story arc, and adapt the scene to the user's emerging personality/emotional profile without breaking continuity.",
            f"Primary instruction: {render_request.render_prompt}",
        ]
        if render_request.negative_prompt:
            sections.append(f"Avoid: {render_request.negative_prompt}")
        if render_request.branch_summary:
            sections.append(f"Current branch context: {render_request.branch_summary}")
        if render_request.world_summary:
            sections.append(f"World summary: {render_request.world_summary}")
        personalization_guidance = render_request.conditioning.get("personalization_guidance")
        if personalization_guidance:
            sections.append(f"Personalization guidance: {personalization_guidance}")
        user_preference_signals = render_request.conditioning.get("user_preference_signals") or []
        if user_preference_signals:
            signals = "; ".join(
                f"{item.get('label', item.get('axis'))}: {item.get('score')}"
                for item in user_preference_signals[:8]
            )
            sections.append(f"Learned user preference signals: {signals}")
            sections.append(
                "Use these signals as soft constraints: positive signals should shape subject choice, emotional tone, pacing, and reveal depth; negative signals should be softened. Do not sacrifice plot clarity or panel-to-panel continuity."
            )
        if render_request.retrieved_facts:
            facts = " | ".join(fact.snippet for fact in render_request.retrieved_facts[:3])
            sections.append(f"Useful world facts: {facts}")
        protected_entities = render_request.conditioning.get("protected_entities") or []
        if protected_entities:
            sections.append(
                "Preserve the visual identity of these recurring entities when relevant: "
                + ", ".join(str(item) for item in protected_entities)
            )
        return "\n".join(section for section in sections if section).strip()

    def _collect_reference_paths(self, render_request: RenderRequest) -> list[str]:
        candidates: list[str] = []
        if render_request.reference_image_uri and Path(render_request.reference_image_uri).is_file():
            candidates.append(render_request.reference_image_uri)

        for entity in render_request.entity_conditioning:
            for reference in entity.reference_images:
                if reference.image_uri and Path(reference.image_uri).is_file():
                    candidates.append(reference.image_uri)

        deduped: list[str] = []
        seen: set[str] = set()
        for path in candidates:
            normalized = str(Path(path))
            if normalized in seen:
                continue
            deduped.append(normalized)
            seen.add(normalized)
            if len(deduped) >= self.max_reference_images:
                break
        return deduped

    @staticmethod
    def _extract_image_payload(*, response, fallback_prompt: str) -> tuple[bytes, str]:
        data = getattr(response, "data", None) or []
        if not data:
            raise RuntimeError("Image API returned no image data.")
        first = data[0]
        image_base64 = getattr(first, "b64_json", None)
        if not image_base64:
            raise RuntimeError("Image API response did not include b64_json image content.")
        revised_prompt = getattr(first, "revised_prompt", None) or fallback_prompt
        return base64.b64decode(image_base64), revised_prompt

    @staticmethod
    def _serialize_usage(usage: Any) -> dict[str, Any]:
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            return usage.model_dump(exclude_none=True)
        if isinstance(usage, dict):
            return {key: value for key, value in usage.items() if value is not None}
        payload: dict[str, Any] = {}
        for name in (
            "input_tokens",
            "input_tokens_details",
            "output_tokens",
            "output_tokens_details",
            "total_tokens",
        ):
            value = getattr(usage, name, None)
            if value is None:
                continue
            if hasattr(value, "model_dump"):
                payload[name] = value.model_dump(exclude_none=True)
            else:
                payload[name] = value
        return payload


class FluxRenderingBackend:
    def __init__(
        self,
        model_path: str = "/c20250509/ZhongzhengWang/model/FLUX.1-dev",
        output_dir: str = "output",
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        height: int = 1024,
        width: int = 1024,
        image_conditioning_mode: str = "off",
        img2img_strength: float = 0.35,
        ip_adapter_model_path: str | None = None,
        ip_adapter_weight_name: str | None = None,
        ip_adapter_subfolder: str = "",
        ip_adapter_image_encoder_path: str | None = None,
        ip_adapter_image_encoder_subfolder: str = "",
        ip_adapter_scale: float = 0.6,
        ip_adapter_image_size: int = 512,
        enable_cpu_offload: bool = True,
        enable_vae_tiling: bool = True,
        enable_vae_slicing: bool = True,
        retry_without_cudnn: bool = True,
        force_disable_cudnn: bool = False,
    ) -> None:
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.height = height
        self.width = width
        self.image_conditioning_mode = image_conditioning_mode
        self.img2img_strength = img2img_strength
        self.ip_adapter_model_path = ip_adapter_model_path
        self.ip_adapter_weight_name = ip_adapter_weight_name
        self.ip_adapter_subfolder = ip_adapter_subfolder
        self.ip_adapter_image_encoder_path = ip_adapter_image_encoder_path
        self.ip_adapter_image_encoder_subfolder = ip_adapter_image_encoder_subfolder
        self.ip_adapter_scale = ip_adapter_scale
        self.ip_adapter_image_size = ip_adapter_image_size
        self.enable_cpu_offload = enable_cpu_offload
        self.enable_vae_tiling = enable_vae_tiling
        self.enable_vae_slicing = enable_vae_slicing
        self.retry_without_cudnn = retry_without_cudnn
        self.force_disable_cudnn = force_disable_cudnn
        self._counter = 0
        self._text2img_pipe = None
        self._ip_adapter_pipe = None
        self._img2img_pipe = None
        self._ip_adapter_loaded = False
        self._ip_adapter_failed = False
        self._ip_adapter_failure_reason: str | None = None

    def preload_models(self) -> None:
        logger.info("Preloading FLUX rendering models at startup...")
        self._get_text2img_pipe()
        if self.image_conditioning_mode == "img2img":
            self._get_img2img_pipe()
        elif self.image_conditioning_mode == "ip_adapter":
            self._get_ip_adapter_pipe()
        logger.info("FLUX rendering models are ready.")

    def _configure_pipeline(self, pipe):
        if self.enable_vae_tiling and hasattr(pipe, "vae"):
            pipe.vae.enable_tiling()
        if self.enable_vae_slicing and hasattr(pipe, "vae"):
            pipe.vae.enable_slicing()

        return pipe

    def _place_pipeline(self, pipe):
        if getattr(pipe, "_worldweaver_runtime_placed", False):
            return pipe
        if self.device == "cuda":
            if self.enable_cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(self.device)
        else:
            pipe.to(self.device)
        setattr(pipe, "_worldweaver_runtime_placed", True)
        return pipe

    def _build_flux_pipeline(self):
        logger.info("Loading FluxPipeline from %s ...", self.model_path)
        import torch
        from diffusers import FluxPipeline

        torch_dtype = getattr(torch, self.dtype, torch.bfloat16)
        pipe = FluxPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
        )
        pipe = self._configure_pipeline(pipe)
        logger.info("FluxPipeline loaded successfully.")
        return pipe

    def _load_text2img_pipeline(self):
        pipe = self._build_flux_pipeline()
        return self._place_pipeline(pipe)

    def _load_ip_adapter_pipeline(self):
        pipe = self._build_flux_pipeline()
        if not self._load_ip_adapter_into_pipe(pipe):
            raise RuntimeError(self._ip_adapter_failure_reason or "Failed to load IP-Adapter pipeline.")
        return self._place_pipeline(pipe)

    def _load_img2img_pipeline(self):
        logger.info("Loading FluxImg2ImgPipeline from %s ...", self.model_path)
        import torch
        from diffusers import FluxImg2ImgPipeline

        torch_dtype = getattr(torch, self.dtype, torch.bfloat16)
        pipe = FluxImg2ImgPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
        )
        pipe = self._configure_pipeline(pipe)
        logger.info("FluxImg2ImgPipeline loaded successfully.")
        return pipe

    def _get_text2img_pipe(self):
        if self._text2img_pipe is None:
            self._text2img_pipe = self._load_text2img_pipeline()
        return self._text2img_pipe

    def _get_ip_adapter_pipe(self):
        if self._ip_adapter_pipe is None:
            self._ip_adapter_pipe = self._load_ip_adapter_pipeline()
        return self._ip_adapter_pipe

    def _get_img2img_pipe(self):
        if self._img2img_pipe is None:
            self._img2img_pipe = self._load_img2img_pipeline()
        return self._img2img_pipe

    def render(self, render_request: RenderRequest, state: WorldState) -> RenderResult:
        import torch

        self._counter += 1
        page_id = f"page_{self._counter:04d}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{page_id}.png"

        logger.info("Rendering %s with prompt: %s", page_id, render_request.render_prompt)
        resolved_conditioning_mode = self._resolve_conditioning_mode(render_request)
        render_request.conditioning["resolved_conditioning_mode"] = resolved_conditioning_mode

        try:
            image = self._run_pipeline(
                render_request,
                disable_cudnn=self.force_disable_cudnn,
                conditioning_mode=resolved_conditioning_mode,
            )
        except RuntimeError as exc:
            error_text = str(exc)
            is_cudnn_init_error = "CUDNN_STATUS_NOT_INITIALIZED" in error_text
            is_oom_error = "out of memory" in error_text.lower()
            if is_cudnn_init_error and self.retry_without_cudnn and not self.force_disable_cudnn:
                logger.warning(
                    "FLUX hit cuDNN initialization failure. Clearing CUDA cache and retrying once with cuDNN disabled."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                image = self._run_pipeline(
                    render_request,
                    disable_cudnn=True,
                    conditioning_mode=resolved_conditioning_mode,
                )
            elif is_oom_error and resolved_conditioning_mode in {"ip_adapter", "img2img"}:
                logger.warning(
                    "FLUX ran out of memory with conditioning enabled. Retrying once with text-only rendering for stability."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                image = self._run_pipeline(
                    render_request,
                    disable_cudnn=self.force_disable_cudnn,
                    conditioning_mode="off",
                )
                resolved_conditioning_mode = "off"
                render_request.conditioning["resolved_conditioning_mode"] = resolved_conditioning_mode
            elif is_cudnn_init_error or is_oom_error:
                raise RuntimeError(
                    "FLUX rendering failed during CUDA execution. "
                    "This is usually a GPU memory or CUDA initialization issue. "
                    "Try smaller settings first, for example "
                    "--flux-height 768 --flux-width 768 --flux-steps 16, "
                    "or temporarily switch to --rendering-backend mock to verify the rest of the pipeline."
                ) from exc
            else:
                raise

        try:
            image.save(str(output_path))
            logger.info("Saved rendered image to %s", output_path)
        finally:
            # Be aggressive about releasing per-step tensors and PIL images.
            try:
                del image
            except UnboundLocalError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        summary = (
            f"{render_request.action.value} on {render_request.target_label} "
            f"inside the world of {state.root_topic}."
        )
        world_facts = [fact.snippet for fact in render_request.retrieved_facts[:3]]
        metadata = {
            "conditioning": render_request.conditioning,
            "style_guide": state.style_guide,
            "retrieval_count": len(render_request.retrieved_facts),
            "model": "FLUX.1-dev",
            "steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "output_path": str(output_path),
            "force_disable_cudnn": self.force_disable_cudnn,
            "image_conditioning_mode": self.image_conditioning_mode,
            "resolved_conditioning_mode": resolved_conditioning_mode,
            "used_reference_image": resolved_conditioning_mode == "img2img",
            "used_ip_adapter": resolved_conditioning_mode == "ip_adapter",
            "used_entity_reference_count": len(render_request.conditioning.get("ip_adapter_references", [])),
            "protected_entities": render_request.conditioning.get("protected_entities", []),
            "selected_entity_references": render_request.conditioning.get("selected_entity_references", []),
            "img2img_strength": self.img2img_strength if resolved_conditioning_mode == "img2img" else None,
            "resolved_img2img_strength": render_request.conditioning.get("resolved_img2img_strength"),
            "reference_crop_applied": render_request.conditioning.get("reference_crop_applied"),
            "reference_crop_box": render_request.conditioning.get("reference_crop_box"),
            "ip_adapter_scale": self.ip_adapter_scale if resolved_conditioning_mode == "ip_adapter" else None,
            "ip_adapter_model_configured": self._is_ip_adapter_configured(),
            "ip_adapter_loaded": self._ip_adapter_loaded,
            "ip_adapter_failure_reason": self._ip_adapter_failure_reason,
        }

        return RenderResult(
            page_id=page_id,
            image_uri=str(output_path),
            revised_prompt=render_request.render_prompt,
            page_summary=summary,
            world_facts=world_facts,
            metadata=metadata,
        )

    def _load_reference_image(self, image_uri: str):
        from PIL import Image

        image = Image.open(image_uri).convert("RGB")
        if image.width != self.width or image.height != self.height:
            image = image.resize((self.width, self.height))
        return image

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _resolve_reference_region(self, render_request: RenderRequest) -> tuple[float, float, float, float] | None:
        raw_region = render_request.conditioning.get("clicked_region")
        if not isinstance(raw_region, dict):
            return None
        try:
            left = self._clamp01(float(raw_region.get("left", 0.0)))
            top = self._clamp01(float(raw_region.get("top", 0.0)))
            width = self._clamp01(float(raw_region.get("width", 0.0)))
            height = self._clamp01(float(raw_region.get("height", 0.0)))
        except (TypeError, ValueError):
            logger.warning("Invalid clicked_region in render conditioning: %s", raw_region)
            return None
        if width <= 0.0 or height <= 0.0:
            return None
        right = self._clamp01(left + width)
        bottom = self._clamp01(top + height)
        if right <= left or bottom <= top:
            return None
        return left, top, right, bottom

    def _resolve_conditioning_mode(self, render_request: RenderRequest) -> str:
        if self.image_conditioning_mode == "ip_adapter":
            if self._has_entity_reference_images(render_request) and self._can_use_ip_adapter():
                return "ip_adapter"
            if self._should_use_reference_image(render_request):
                return "img2img"
            return "off"
        if self.image_conditioning_mode == "img2img" and self._should_use_reference_image(render_request):
            return "img2img"
        return "off"

    def _should_use_reference_image(self, render_request: RenderRequest) -> bool:
        if render_request.reference_image_uri is None or not Path(render_request.reference_image_uri).is_file():
            return False
        if render_request.conditioning.get("continuity_mode") == "scene_shift":
            return False
        return True

    def _should_crop_reference(self, render_request: RenderRequest) -> bool:
        if render_request.conditioning.get("continuity_mode") == "scene_shift":
            return False
        return True

    @staticmethod
    def _has_entity_reference_images(render_request: RenderRequest) -> bool:
        return any(
            reference.image_uri and Path(reference.image_uri).is_file()
            for entity in render_request.entity_conditioning
            for reference in entity.reference_images
        )

    def _is_ip_adapter_configured(self) -> bool:
        return bool(self.ip_adapter_model_path and self.ip_adapter_weight_name)

    def _can_use_ip_adapter(self) -> bool:
        if self._ip_adapter_loaded:
            return True
        if self._ip_adapter_failed:
            return False
        try:
            self._get_ip_adapter_pipe()
            return True
        except Exception as exc:
            self._ip_adapter_failed = True
            self._ip_adapter_failure_reason = str(exc)
            logger.warning("Failed to initialize IP-Adapter pipeline: %s", exc, exc_info=True)
            return False

    def _resolve_ip_adapter_image_encoder_path(self) -> str | None:
        if self.ip_adapter_image_encoder_path:
            return self.ip_adapter_image_encoder_path
        if not self.ip_adapter_model_path:
            return None
        candidate = Path(self.ip_adapter_model_path) / "image_encoder"
        if candidate.is_dir():
            return str(candidate)
        return None

    def _load_ip_adapter_into_pipe(self, pipe) -> bool:
        if not self._is_ip_adapter_configured():
            logger.warning(
                "IP-Adapter conditioning was requested but no adapter weights were configured. "
                "Set --flux-ip-adapter-model-path and --flux-ip-adapter-weight-name to enable it."
            )
            return False
        try:
            image_encoder_path = self._resolve_ip_adapter_image_encoder_path()
            pipe.load_ip_adapter(
                self.ip_adapter_model_path,
                subfolder=self.ip_adapter_subfolder or "",
                weight_name=self.ip_adapter_weight_name,
                image_encoder_pretrained_model_name_or_path=image_encoder_path,
                image_encoder_subfolder=self.ip_adapter_image_encoder_subfolder or "",
            )
            pipe.set_ip_adapter_scale(self.ip_adapter_scale)
            self._ip_adapter_loaded = True
            logger.info("Loaded IP-Adapter weights from %s", self.ip_adapter_model_path)
            if image_encoder_path:
                logger.info("Loaded IP-Adapter image encoder from %s", image_encoder_path)
            return True
        except Exception as exc:
            self._ip_adapter_failed = True
            self._ip_adapter_failure_reason = str(exc)
            logger.warning("Failed to load IP-Adapter weights: %s", exc, exc_info=True)
            return False

    def _expand_region(
        self,
        region: tuple[float, float, float, float],
        *,
        min_size: float,
        expand_factor: float,
    ) -> tuple[float, float, float, float]:
        left, top, right, bottom = region
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        width = max(right - left, min_size) * expand_factor
        height = max(bottom - top, min_size) * expand_factor
        half_w = min(width / 2.0, 0.5)
        half_h = min(height / 2.0, 0.5)
        new_left = self._clamp01(cx - half_w)
        new_top = self._clamp01(cy - half_h)
        new_right = self._clamp01(cx + half_w)
        new_bottom = self._clamp01(cy + half_h)
        return new_left, new_top, new_right, new_bottom

    def _prepare_reference_image(self, render_request: RenderRequest):
        from PIL import Image

        if render_request.reference_image_uri is None:
            return None, {"reference_crop_applied": False}

        with Image.open(render_request.reference_image_uri) as opened:
            original = opened.convert("RGB")
        crop_applied = False
        crop_box = None
        region = self._resolve_reference_region(render_request)

        if region is not None and self._should_crop_reference(render_request):
            expanded = self._expand_region(region, min_size=0.12, expand_factor=1.25)
            left, top, right, bottom = expanded
            pixel_box = (
                int(left * original.width),
                int(top * original.height),
                int(right * original.width),
                int(bottom * original.height),
            )
            if pixel_box[2] > pixel_box[0] and pixel_box[3] > pixel_box[1]:
                original = original.crop(pixel_box)
                crop_applied = True
                crop_box = {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                }

        if original.width != self.width or original.height != self.height:
            original = original.resize((self.width, self.height))

        return original, {
            "reference_crop_applied": crop_applied,
            "reference_crop_box": crop_box,
        }

    def _resolve_img2img_strength(self, render_request: RenderRequest, *, use_reference_image: bool) -> float:
        if not use_reference_image:
            return self.img2img_strength

        action_strength_map = {
            ExplorationAction.ZOOM_IN: 0.72,
            ExplorationAction.REVEAL: 0.74,
            ExplorationAction.REFRAME: 0.78,
            ExplorationAction.BRANCH_OUT: 0.82,
        }
        resolved = max(self.img2img_strength, action_strength_map.get(render_request.action, self.img2img_strength))

        if self._resolve_reference_region(render_request) is None:
            resolved = max(resolved, 0.80)

        return min(resolved, 0.88)

    def _build_identity_guidance(self, entity_conditioning: list[EntityConditioning]) -> str:
        if not entity_conditioning:
            return ""

        guidance_fragments: list[str] = []
        for entity in entity_conditioning[:3]:
            signature = _clean_visual_fragment(entity.visual_signature)
            reference_caption = None
            for reference in reversed(entity.reference_images):
                reference_caption = _clean_visual_fragment(reference.caption)
                if reference_caption:
                    break
            detail = signature or reference_caption
            if detail:
                guidance_fragments.append(
                    f"Keep {entity.entity_name} visually consistent with earlier appearances: {detail}."
                )
            else:
                guidance_fragments.append(
                    f"Keep {entity.entity_name} visually consistent with earlier appearances."
                )
        return " ".join(guidance_fragments)

    def _build_conditioned_prompt(self, render_request: RenderRequest, *, conditioning_mode: str) -> str:
        conditioning = render_request.conditioning or {}
        focus_subject = conditioning.get("focus_subject") or render_request.target_label
        focus_type = conditioning.get("focus_type") or "region"
        shot_type = conditioning.get("shot_type")
        narrative_function = conditioning.get("narrative_function")
        emotion_hint = conditioning.get("emotion_hint")
        next_panel_expectation = conditioning.get("next_panel_expectation")
        continuity_notes = conditioning.get("continuity_notes") or []
        continuity_text = " ".join(str(item) for item in continuity_notes if item)
        continuity_mode = conditioning.get("continuity_mode") or "local_continuation"
        identity_guidance = self._build_identity_guidance(render_request.entity_conditioning)

        if conditioning_mode == "off":
            return (
                f"{render_request.render_prompt} "
                f"The clicked area to continue from is {focus_subject} as a {focus_type}. "
                f"{f'Identity continuity: {identity_guidance} ' if identity_guidance else ''}"
                f"{f'Secondary shot hint: {shot_type}. ' if shot_type else ''}"
                f"{f'Narrative function: {narrative_function}. ' if narrative_function else ''}"
                f"{f'Emotion: {emotion_hint}. ' if emotion_hint else ''}"
                f"{f'Expected next panel beat: {next_panel_expectation}. ' if next_panel_expectation else ''}"
                f"{'Allow the next image to move into a connected new room, street, interior, or adjacent place while preserving story identity. ' if continuity_mode == 'scene_shift' else ''}"
                "Prefer concrete visible objects, actions, and environment changes over abstract cinematic wording. "
                "Show a real next story event or local change in that clicked area. "
                "Make this panel clearly different from the previous panel rather than a near-duplicate."
            ).strip()

        if conditioning_mode == "ip_adapter":
            return (
                "Use the supplied reference images as identity anchors for recurring characters, props, or creatures only. "
                "Keep those entities clearly recognizable while allowing the surrounding scene, camera, pose, and environment to change freely for the next story beat. "
                "Do not recreate the previous frame composition. Do not lock the old background. "
                f"{f'Identity continuity: {identity_guidance} ' if identity_guidance else ''}"
                f"The clicked area to continue from is {focus_subject} as a {focus_type}. "
                f"{f'Secondary shot hint: {shot_type}. ' if shot_type else ''}"
                f"{f'Narrative function: {narrative_function}. ' if narrative_function else ''}"
                f"{f'Emotion: {emotion_hint}. ' if emotion_hint else ''}"
                f"{f'Expected next panel beat: {next_panel_expectation}. ' if next_panel_expectation else ''}"
                f"{f'Continuity notes: {continuity_text}. ' if continuity_text else ''}"
                "Prefer concrete visible objects, actions, and environment changes over abstract cinematic wording. "
                f"{render_request.render_prompt}"
            )

        return (
            "Use the reference image only as a weak hint for keeping the clicked local object, character identity, or key prop loosely recognizable. "
            "Do not preserve the whole original scene layout. Do not simply recreate the same frame. "
            "Narrative progression is the primary goal: allow the surrounding environment, composition, and broader scene structure to change whenever the next story beat benefits from it. "
            "Preserve only small local identity cues from the clicked area; allow the rest of the image to evolve freely into the next moment. "
            f"{f'Identity continuity: {identity_guidance} ' if identity_guidance else ''}"
            f"The clicked area to continue from is {focus_subject} as a {focus_type}. "
            f"{f'Secondary shot hint: {shot_type}. ' if shot_type else ''}"
            f"{f'Narrative function: {narrative_function}. ' if narrative_function else ''}"
            f"{f'Emotion: {emotion_hint}. ' if emotion_hint else ''}"
            f"{f'Expected next panel beat: {next_panel_expectation}. ' if next_panel_expectation else ''}"
            f"{f'Continuity notes: {continuity_text}. ' if continuity_text else ''}"
            "Prefer concrete visible objects, actions, and environment changes over abstract cinematic wording. "
            f"{render_request.render_prompt}"
        )

    def _prepare_ip_adapter_image(self, render_request: RenderRequest):
        from PIL import Image, ImageOps

        prepared_tiles = []
        prepared_metadata: list[dict[str, Any]] = []
        resampling = getattr(Image, "Resampling", Image)
        tile_size = self.ip_adapter_image_size

        for entity in render_request.entity_conditioning[:3]:
            for reference in entity.reference_images[-2:]:
                if not reference.image_uri or not Path(reference.image_uri).is_file():
                    continue
                with Image.open(reference.image_uri) as opened:
                    reference_image = opened.convert("RGB")
                prepared = ImageOps.pad(
                    reference_image,
                    (tile_size, tile_size),
                    method=resampling.BICUBIC,
                    color=(245, 245, 245),
                )
                prepared_tiles.append(prepared)
                prepared_metadata.append(
                    {
                        "entity_id": entity.entity_id,
                        "entity_name": entity.entity_name,
                        "image_uri": reference.image_uri,
                        "source_page_id": reference.source_page_id,
                        "caption": reference.caption,
                        "reference_type": reference.reference_type,
                    }
                )
                if len(prepared_tiles) >= 4:
                    break
            if len(prepared_tiles) >= 4:
                break

        if not prepared_tiles:
            return None, prepared_metadata

        if len(prepared_tiles) == 1:
            return prepared_tiles[0], prepared_metadata

        columns = 2 if len(prepared_tiles) > 1 else 1
        rows = (len(prepared_tiles) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), color=(245, 245, 245))

        for index, tile in enumerate(prepared_tiles):
            col = index % columns
            row = index // columns
            sheet.paste(tile, (col * tile_size, row * tile_size))
            tile.close()

        return sheet, prepared_metadata

    def _run_pipeline(self, render_request: RenderRequest, *, disable_cudnn: bool, conditioning_mode: str):
        import torch

        cudnn_context = (
            torch.backends.cudnn.flags(enabled=False)
            if disable_cudnn and self.device.startswith("cuda")
            else nullcontext()
        )
        with cudnn_context:
            prompt = self._build_conditioned_prompt(
                render_request,
                conditioning_mode=conditioning_mode,
            )
            if conditioning_mode == "ip_adapter":
                pipe = self._get_ip_adapter_pipe()
                pipe.set_ip_adapter_scale(self.ip_adapter_scale)
                ip_adapter_image, ip_adapter_metadata = self._prepare_ip_adapter_image(render_request)
                render_request.conditioning["ip_adapter_references"] = ip_adapter_metadata
                render_request.conditioning["ip_adapter_reference_layout"] = {
                    "reference_count": len(ip_adapter_metadata),
                    "composited": len(ip_adapter_metadata) > 1,
                }
                if ip_adapter_image is None:
                    logger.warning(
                        "IP-Adapter mode was selected but no usable reference images were available after preparation. "
                        "Falling back to text-only generation for this turn."
                    )
                    pipe = self._get_text2img_pipe()
                    return pipe(
                        prompt=prompt,
                        negative_prompt=render_request.negative_prompt or None,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.num_inference_steps,
                        height=self.height,
                        width=self.width,
                    ).images[0]
                try:
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=render_request.negative_prompt or None,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.num_inference_steps,
                        height=self.height,
                        width=self.width,
                        ip_adapter_image=ip_adapter_image,
                    ).images[0]
                finally:
                    try:
                        ip_adapter_image.close()
                    except Exception:
                        pass
                return result

            if conditioning_mode == "img2img":
                pipe = self._get_img2img_pipe()
                reference_image, reference_metadata = self._prepare_reference_image(render_request)
                strength = self._resolve_img2img_strength(
                    render_request,
                    use_reference_image=True,
                )
                render_request.conditioning["resolved_img2img_strength"] = strength
                render_request.conditioning.update(reference_metadata)
                try:
                    result = pipe(
                        prompt=prompt,
                        image=reference_image,
                        strength=strength,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.num_inference_steps,
                        height=self.height,
                        width=self.width,
                    ).images[0]
                finally:
                    try:
                        del reference_image
                    except UnboundLocalError:
                        pass
                return result

            pipe = self._get_text2img_pipe()
            return pipe(
                prompt=prompt,
                negative_prompt=render_request.negative_prompt or None,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                height=self.height,
                width=self.width,
            ).images[0]


def load_retrieval_facts(path: str | Path) -> list[RetrievedFact]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [RetrievedFact(**record) for record in records]
