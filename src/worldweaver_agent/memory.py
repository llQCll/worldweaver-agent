from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

from worldweaver_agent.schemas import (
    ClickEvent,
    EntityConditioning,
    EntityReference,
    ExplorationAction,
    ImageFrame,
    NarrationResult,
    NormalizedBox,
    PageRecord,
    PerceptionResult,
    PlanningDecision,
    SceneMemory,
    SceneReference,
    RenderRequest,
    RenderResult,
    RetrievedFact,
    UserFeedbackRecord,
    UserPreferenceSignal,
    WorldEntity,
    WorldState,
)


class MemoryAgent:
    _GENERIC_ENTITY_KEYS = {
        "area",
        "character",
        "clicked_area",
        "clicked_region",
        "destination",
        "item",
        "location",
        "main_character",
        "object",
        "person",
        "place",
        "portal",
        "protagonist",
        "region",
        "scene",
        "story_anchor",
        "target",
        "thing",
        "world",
    }

    def __init__(self, state: WorldState):
        self.state = state
        self._reference_dir = Path("memory_refs")
        self._reference_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        *,
        root_topic: str,
        style_guide: str,
        story_outline: str | None = None,
        story_outline_source: str = "none",
    ) -> "MemoryAgent":
        state = WorldState.create(root_topic=root_topic, style_guide=style_guide)
        if story_outline:
            state.story_outline = story_outline.strip()
            state.story_outline_source = story_outline_source
            state.story_outline_revision = 1
        return cls(state=state)

    def build_render_request(
        self,
        *,
        image_page_id: str,
        image_uri: str | None,
        perception: PerceptionResult,
        plan: PlanningDecision,
        retrieved_facts: list[RetrievedFact],
    ) -> RenderRequest:
        branch = self.state.ensure_branch(plan.branch_id, plan.branch_label)
        entity_conditioning = self._select_entity_conditioning(plan=plan, perception=perception)
        return RenderRequest(
            session_id=self.state.session_id,
            branch_id=branch.branch_id,
            source_page_id=image_page_id,
            target_label=perception.target_label,
            action=plan.action,
            render_prompt=plan.render_prompt,
            negative_prompt=plan.negative_prompt,
            world_summary=self.state.world_summary,
            branch_summary=branch.summary,
            retrieved_facts=retrieved_facts,
            entity_conditioning=entity_conditioning,
            reference_image_uri=image_uri,
            conditioning={
                "source_page_id": image_page_id,
                "target_label": perception.target_label,
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
                "action": plan.action.value,
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
                "continuity_notes": list(plan.continuity_notes),
                "style_directives": plan.style_directives,
                "world_update": plan.world_update,
                "current_branch": branch.label,
                "protected_entities": list(plan.protected_entities),
                "selected_entity_references": self._serialize_entity_conditioning(entity_conditioning),
                "user_preference_signals": self._serialize_user_preference_signals(),
                "personalization_guidance": self._build_personalization_guidance(),
            },
        )

    def commit_turn(
        self,
        *,
        source_page_id: str | None,
        click: ClickEvent,
        perception: PerceptionResult,
        plan: PlanningDecision,
        retrieved_facts: list[RetrievedFact],
        render_result: RenderResult,
        narration: NarrationResult,
    ) -> None:
        self.commit_turn_without_narration(
            source_page_id=source_page_id,
            click=click,
            perception=perception,
            plan=plan,
            retrieved_facts=retrieved_facts,
            render_result=render_result,
        )
        self.attach_narration(page_id=render_result.page_id, narration=narration)

    def commit_turn_without_narration(
        self,
        *,
        source_page_id: str | None,
        click: ClickEvent,
        perception: PerceptionResult,
        plan: PlanningDecision,
        retrieved_facts: list[RetrievedFact],
        render_result: RenderResult,
    ) -> None:
        branch = self.state.ensure_branch(plan.branch_id, plan.branch_label)
        page = PageRecord(
            page_id=render_result.page_id,
            branch_id=branch.branch_id,
            scene_id=None,
            source_page_id=source_page_id,
            image_uri=render_result.image_uri,
            prompt=render_result.revised_prompt,
            page_summary=render_result.page_summary,
            action=plan.action,
            target_label=perception.target_label,
            click=click,
            perception=perception,
            plan=plan,
            retrieved_facts=retrieved_facts,
            world_facts=render_result.world_facts,
            narration=None,
            narration_status="pending",
            narration_error=None,
            render_metadata=dict(render_result.metadata),
        )
        self.state.pages[page.page_id] = page
        branch.page_ids.append(page.page_id)
        branch.summary = self._summarize_branch(branch_id=branch.branch_id)
        self.state.current_page_id = page.page_id
        self.state.current_branch_id = branch.branch_id
        self.state.history.append(page.page_id)
        self.state.user_profile_summary = plan.user_profile_summary or perception.user_profile_summary
        self.update_story_outline_from_plan(plan)
        self.record_user_interaction(
            click=click,
            perception=perception,
            plan=plan,
            page_id=page.page_id,
        )
        image_frame = ImageFrame(
            page_id=page.page_id,
            image_uri=page.image_uri,
            prompt=page.prompt,
            summary=page.page_summary,
        )
        scene = self._update_scene_memory(page=page, plan=plan, perception=perception, image=image_frame)
        page.scene_id = scene.scene_id
        self._merge_entities(perception, image=image_frame)
        self.state.world_summary = self._summarize_world()

    def record_user_interaction(
        self,
        *,
        click: ClickEvent,
        perception: PerceptionResult,
        plan: PlanningDecision,
        page_id: str,
    ) -> None:
        axes: dict[str, float] = {
            "agency_preference": 0.16,
            "narrative_alignment": 0.08,
        }
        if plan.action == ExplorationAction.REVEAL:
            axes.update(
                {
                    "mystery_tension": 0.18,
                    "comfort_with_ambiguity": 0.12,
                    "exploration_drive": 0.1,
                }
            )
        elif plan.action == ExplorationAction.BRANCH_OUT:
            axes.update(
                {
                    "world_focus": 0.16,
                    "exploration_drive": 0.18,
                    "novelty_surprise": 0.1,
                }
            )
        elif plan.action == ExplorationAction.ZOOM_IN:
            axes.update(
                {
                    "detail_orientation": 0.16,
                    "closure_need": 0.08,
                    "continuity": 0.08,
                }
            )
        elif plan.action == ExplorationAction.REFRAME:
            axes.update(
                {
                    "continuity": 0.14,
                    "pacing_preference": 0.08,
                }
            )

        if perception.focus_type == "character":
            axes["character_focus"] = axes.get("character_focus", 0.0) + 0.14
            axes["social_focus"] = axes.get("social_focus", 0.0) + 0.1
        elif perception.focus_type in {"place", "location", "environment", "world"}:
            axes["world_focus"] = axes.get("world_focus", 0.0) + 0.12

        if perception.emotion_hint:
            axes["emotional_intensity"] = axes.get("emotional_intensity", 0.0) + 0.1
            if any(term in perception.emotion_hint.lower() for term in ("warm", "safe", "tender", "cozy")):
                axes["warmth_safety"] = axes.get("warmth_safety", 0.0) + 0.12
            if any(term in perception.emotion_hint.lower() for term in ("suspense", "tense", "mystery", "dark")):
                axes["mystery_tension"] = axes.get("mystery_tension", 0.0) + 0.12

        if click.user_hint:
            axes["agency_preference"] = axes.get("agency_preference", 0.0) + 0.08

        evidence = (
            f"clicked {perception.target_label} on {page_id}; action={plan.action.value}; "
            f"intent={perception.interaction_intent or perception.intent_hint}"
        )
        self._apply_user_signal_axes(axes, evidence=evidence, learning_rate=0.06)
        self.state.user_profile_summary = self._summarize_user_profile()

    def record_user_feedback(
        self,
        *,
        page_id: str,
        feedback_type: str,
        label: str,
        axes: dict[str, float],
        note: str | None = None,
    ) -> UserFeedbackRecord:
        page = self.state.pages.get(page_id)
        cleaned_axes = {
            str(axis): max(-1.0, min(1.0, float(score)))
            for axis, score in axes.items()
            if axis
        }
        record = UserFeedbackRecord(
            page_id=page_id,
            feedback_type=feedback_type,
            label=label,
            created_at=time.time(),
            axes=cleaned_axes,
            note=note,
            plan_action=page.plan.action.value if page and page.plan else None,
            target_label=page.target_label if page else None,
        )
        self.state.user_feedback_history.append(record)
        self.state.user_feedback_history = self.state.user_feedback_history[-40:]
        evidence = self._feedback_evidence(record, page)
        self._apply_user_signal_axes(cleaned_axes, evidence=evidence, learning_rate=0.18)
        self.state.user_profile_summary = self._summarize_user_profile()
        return record

    def _apply_user_signal_axes(
        self,
        axes: dict[str, float],
        *,
        evidence: str,
        learning_rate: float,
    ) -> None:
        for axis, score in axes.items():
            signal = self.state.user_preference_signals.get(axis)
            if signal is None:
                signal = UserPreferenceSignal(axis=axis, label=self._axis_label(axis))
                self.state.user_preference_signals[axis] = signal
            clipped_score = max(-1.0, min(1.0, float(score)))
            signal.score = max(
                -1.0,
                min(1.0, signal.score * (1.0 - learning_rate) + clipped_score * learning_rate),
            )
            signal.evidence.append(evidence)
            signal.evidence = signal.evidence[-6:]

    def update_story_outline_from_plan(self, plan: PlanningDecision) -> None:
        outline_update = (plan.story_outline_update or "").strip()
        if not outline_update:
            return
        if outline_update == self.state.story_outline:
            return
        self.state.story_outline = outline_update
        self.state.story_outline_source = "planning_update"
        self.state.story_outline_revision += 1

    def _serialize_user_preference_signals(self) -> list[dict[str, Any]]:
        return [
            {
                "axis": signal.axis,
                "label": signal.label,
                "score": round(signal.score, 3),
                "evidence": list(signal.evidence),
            }
            for signal in sorted(
                self.state.user_preference_signals.values(),
                key=lambda item: abs(item.score),
                reverse=True,
            )
            if abs(signal.score) >= 0.05
        ][:10]

    def _build_personalization_guidance(self) -> str:
        signals = self._serialize_user_preference_signals()
        if not signals:
            return self.state.user_profile_summary
        positive = [
            f"{item['label']} ({item['score']:+.2f})"
            for item in signals
            if item["score"] > 0
        ][:5]
        negative = [
            f"{item['label']} ({item['score']:+.2f})"
            for item in signals
            if item["score"] < 0
        ][:3]
        fragments = [self.state.user_profile_summary]
        if positive:
            fragments.append("倾向强化：" + "，".join(positive) + "。")
        if negative:
            fragments.append("需要减弱：" + "，".join(negative) + "。")
        fragments.append(
            "同时平衡两个目标：保持连环画故事弧连贯完整，并保留可用于阶段性性格与情绪分析的交互信号。"
        )
        return " ".join(fragment for fragment in fragments if fragment).strip()

    def _summarize_user_profile(self) -> str:
        signals = self._serialize_user_preference_signals()
        if not signals:
            return self.state.user_profile_summary
        positive = [item for item in signals if item["score"] > 0.08]
        negative = [item for item in signals if item["score"] < -0.08]
        if not positive and not negative:
            return self.state.user_profile_summary
        exploration = [
            item["label"]
            for item in positive
            if item["axis"] in {"exploration_drive", "novelty_surprise", "mystery_tension", "detail_orientation"}
        ][:3]
        emotional = [
            item["label"]
            for item in positive
            if item["axis"] in {"warmth_safety", "emotional_intensity", "comfort_with_ambiguity", "closure_need"}
        ][:3]
        narrative = [
            item["label"]
            for item in positive
            if item["axis"] in {"narrative_alignment", "continuity", "pacing_preference", "character_focus", "world_focus", "social_focus"}
        ][:4]
        parts = []
        if exploration:
            parts.append("在探索方式上偏向" + "、".join(exploration))
        if narrative:
            parts.append("在叙事反应上偏向" + "、".join(narrative))
        if emotional:
            parts.append("在情绪体验上偏向" + "、".join(emotional))
        if negative:
            parts.append("对" + "、".join(item["label"] for item in negative[:3]) + "耐受较低")
        return "用户" + "；".join(parts) + "。"

    @staticmethod
    def _axis_label(axis: str) -> str:
        labels = {
            "overall_alignment": "整体符合期待",
            "narrative_alignment": "清晰的故事方向",
            "affective_alignment": "情绪与氛围控制",
            "visual_alignment": "视觉风格一致性",
            "character_focus": "角色中心的发展",
            "world_focus": "世界与地点探索",
            "mystery_tension": "谜团、秘密与张力",
            "warmth_safety": "温暖、安全、亲密的时刻",
            "novelty_surprise": "惊喜变化与发现",
            "continuity": "与前序画面的连续性",
            "exploration_drive": "主动探索与好奇心",
            "detail_orientation": "对细节的细致关注",
            "agency_preference": "主动掌控故事路径",
            "emotional_intensity": "情绪浓度较高的时刻",
            "comfort_with_ambiguity": "对暧昧和未解线索的开放度",
            "closure_need": "对清晰解释和收束的需求",
            "pacing_preference": "稳定的故事节奏",
            "social_focus": "对关系和社交线索的兴趣",
        }
        return labels.get(axis, axis.replace("_", " "))

    @staticmethod
    def _feedback_evidence(record: UserFeedbackRecord, page: PageRecord | None) -> str:
        target = record.target_label or (page.target_label if page else "current panel")
        action = record.plan_action or "unknown_action"
        note = f"; note={record.note}" if record.note else ""
        return f"{record.label} on {target} after {action}{note}"

    def attach_narration(self, *, page_id: str, narration: NarrationResult) -> None:
        page = self.state.pages.get(page_id)
        if page is None:
            return
        page.narration = narration
        page.narration_status = "ready"
        page.narration_error = None

    def fail_narration(self, *, page_id: str, error: str) -> None:
        page = self.state.pages.get(page_id)
        if page is None:
            return
        page.narration_status = "failed"
        page.narration_error = error

    def mark_page_narration_pending(self, *, page_id: str) -> None:
        page = self.state.pages.get(page_id)
        if page is None:
            return
        page.narration = None
        page.narration_status = "pending"
        page.narration_error = None

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state.to_dict(), indent=2), encoding="utf-8")

    def _merge_entities(self, perception: PerceptionResult, *, image: ImageFrame) -> None:
        for candidate in self._iter_entity_candidates(perception):
            entity_id = candidate.name.lower().replace(" ", "_")
            existing = self.state.entities.get(entity_id)
            if existing is None:
                self.state.entities[entity_id] = WorldEntity(
                    entity_id=entity_id,
                    name=candidate.name,
                    description=candidate.description,
                    tags=list(candidate.tags),
                    mentions=1,
                    visual_signature=self._build_visual_signature(candidate),
                    first_seen_page_id=image.page_id,
                    last_seen_page_id=image.page_id,
                )
                existing = self.state.entities[entity_id]
            else:
                existing.mentions += 1
                merged_tags = sorted(set(existing.tags) | set(candidate.tags))
                existing.tags = merged_tags
                existing.last_seen_page_id = image.page_id
                if not existing.visual_signature:
                    existing.visual_signature = self._build_visual_signature(candidate)

            reference = self._build_entity_reference(image=image, candidate_region=candidate.region, caption=candidate.description)
            if reference is not None:
                self._append_reference(existing, reference)

    @staticmethod
    def _iter_entity_candidates(perception: PerceptionResult) -> Iterable:
        if perception.candidate_entities:
            return perception.candidate_entities
        fallback_name = perception.focus_subject or perception.target_label
        if not fallback_name:
            return []
        return [
            type("FallbackCandidate", (), {
                "name": fallback_name,
                "description": perception.region_caption,
                "region": perception.clicked_region,
                "confidence": perception.confidence,
                "tags": [perception.focus_type] if perception.focus_type else [],
            })()
        ]

    @staticmethod
    def _build_visual_signature(candidate) -> str | None:
        parts: list[str] = []
        if getattr(candidate, "description", None):
            parts.append(str(candidate.description))
        if getattr(candidate, "tags", None):
            parts.extend(str(tag) for tag in candidate.tags if tag)
        if not parts:
            return None
        return "; ".join(parts[:6])

    def _build_entity_reference(
        self,
        *,
        image: ImageFrame,
        candidate_region: NormalizedBox | None,
        caption: str | None,
    ) -> EntityReference | None:
        if not image.image_uri:
            return None
        image_path = Path(image.image_uri)
        if not image_path.is_file():
            return None
        if candidate_region is None:
            return None

        crop_path = self._save_reference_crop(
            image_path=image_path,
            page_id=image.page_id,
            region=candidate_region,
        )
        if crop_path is None:
            return None

        return EntityReference(
            image_uri=str(crop_path),
            source_page_id=image.page_id,
            caption=caption,
            region=candidate_region,
            reference_type="crop",
        )

    def _select_entity_conditioning(
        self,
        *,
        plan: PlanningDecision,
        perception: PerceptionResult,
        max_entities: int = 3,
        max_references_per_entity: int = 2,
    ) -> list[EntityConditioning]:
        selected: list[EntityConditioning] = []
        seen_entity_ids: set[str] = set()
        candidate_names = self._gather_conditioning_candidates(plan=plan, perception=perception)

        for candidate_name in candidate_names:
            entity = self._resolve_known_entity(candidate_name)
            if entity is None or entity.entity_id in seen_entity_ids:
                continue

            valid_references = [
                reference
                for reference in entity.reference_bank
                if reference.image_uri and Path(reference.image_uri).is_file()
            ]
            selected_references = valid_references[-max_references_per_entity:]
            if not selected_references and not entity.visual_signature:
                continue

            selected.append(
                EntityConditioning(
                    entity_id=entity.entity_id,
                    entity_name=entity.name,
                    visual_signature=entity.visual_signature,
                    reference_images=selected_references,
                )
            )
            seen_entity_ids.add(entity.entity_id)
            if len(selected) >= max_entities:
                break

        return selected

    def _gather_conditioning_candidates(
        self,
        *,
        plan: PlanningDecision,
        perception: PerceptionResult,
    ) -> list[str]:
        ordered_names = [
            *(plan.protected_entities or []),
            perception.focus_subject or "",
            plan.supporting_subject or "",
            perception.target_label or "",
            *(candidate.name for candidate in perception.candidate_entities),
        ]
        deduped: list[str] = []
        seen: set[str] = set()
        for name in ordered_names:
            normalized = self._normalize_entity_name(name)
            if not normalized or normalized in self._GENERIC_ENTITY_KEYS or normalized in seen:
                continue
            deduped.append(str(name).strip())
            seen.add(normalized)
        return deduped

    def _resolve_known_entity(self, candidate_name: str) -> WorldEntity | None:
        normalized = self._normalize_entity_name(candidate_name)
        if not normalized:
            return None

        direct = self.state.entities.get(normalized)
        if direct is not None:
            return direct

        best_entity: WorldEntity | None = None
        best_score = 0
        candidate_tokens = self._tokenize_entity_name(candidate_name)

        for entity in self.state.entities.values():
            entity_key = self._normalize_entity_name(entity.name or entity.entity_id)
            if normalized == entity_key:
                return entity
            if normalized in entity_key or entity_key in normalized:
                score = min(len(normalized), len(entity_key))
                if score > best_score:
                    best_entity = entity
                    best_score = score
                continue

            entity_tokens = self._tokenize_entity_name(entity.name)
            overlap = len(candidate_tokens & entity_tokens)
            if overlap > best_score:
                best_entity = entity
                best_score = overlap

        return best_entity

    @staticmethod
    def _normalize_entity_name(value: str | None) -> str:
        if not value:
            return ""
        normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        return normalized

    @classmethod
    def _tokenize_entity_name(cls, value: str | None) -> set[str]:
        normalized = cls._normalize_entity_name(value)
        if not normalized:
            return set()
        return {token for token in normalized.split("_") if token}

    @staticmethod
    def _serialize_entity_conditioning(entity_conditioning: list[EntityConditioning]) -> list[dict[str, object]]:
        return [
            {
                "entity_id": item.entity_id,
                "entity_name": item.entity_name,
                "visual_signature": item.visual_signature,
                "reference_count": len(item.reference_images),
                "references": [
                    {
                        "image_uri": reference.image_uri,
                        "source_page_id": reference.source_page_id,
                        "caption": reference.caption,
                        "reference_type": reference.reference_type,
                    }
                    for reference in item.reference_images
                ],
            }
            for item in entity_conditioning
        ]

    def _save_reference_crop(
        self,
        *,
        image_path: Path,
        page_id: str,
        region: NormalizedBox,
    ) -> Path | None:
        try:
            from PIL import Image
        except ImportError:
            return None

        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
        except Exception:
            return None

        left = max(0, int(region.left * image.width))
        top = max(0, int(region.top * image.height))
        right = min(image.width, int((region.left + region.width) * image.width))
        bottom = min(image.height, int((region.top + region.height) * image.height))
        if right <= left or bottom <= top:
            return None

        crop = image.crop((left, top, right, bottom))
        crop_name = f"{page_id}_{left}_{top}_{right}_{bottom}.png"
        crop_path = self._reference_dir / crop_name
        crop.save(crop_path)
        return crop_path

    @staticmethod
    def _append_reference(entity: WorldEntity, reference: EntityReference, max_references: int = 6) -> None:
        if any(existing.image_uri == reference.image_uri for existing in entity.reference_bank):
            return
        entity.reference_bank.append(reference)
        if len(entity.reference_bank) > max_references:
            entity.reference_bank[:] = entity.reference_bank[-max_references:]

    def _summarize_branch(self, branch_id: str) -> str:
        branch = self.state.branches[branch_id]
        latest_pages = [self.state.pages[page_id] for page_id in branch.page_ids[-3:] if page_id in self.state.pages]
        if not latest_pages:
            return branch.summary
        summaries = ", ".join(page.page_summary for page in latest_pages)
        return f"{branch.label}: {summaries}"

    def _summarize_world(self) -> str:
        branch = self.state.branches[self.state.current_branch_id]
        entity_names = ", ".join(entity.name for entity in list(self.state.entities.values())[:8]) or "no entities yet"
        current_scene = self.state.scenes.get(self.state.current_scene_id or "")
        scene_fragment = (
            f"Current scene is {current_scene.scene_name}. "
            if current_scene is not None
            else ""
        )
        return (
            f"World about {self.state.root_topic}. "
            f"Current branch is {branch.label}. "
            f"{scene_fragment}"
            f"Known entities include {entity_names}."
        )

    def _update_scene_memory(
        self,
        *,
        page: PageRecord,
        plan: PlanningDecision,
        perception: PerceptionResult,
        image: ImageFrame,
    ) -> SceneMemory:
        is_new_scene = plan.continuity_mode == "scene_shift" or self.state.current_scene_id is None
        if is_new_scene:
            scene_id = self._build_scene_id(plan=plan, page=page)
            scene = self.state.scenes.get(scene_id)
            if scene is None:
                scene = SceneMemory(
                    scene_id=scene_id,
                    scene_name=self._derive_scene_name(plan=plan, perception=perception),
                    description=self._derive_scene_description(plan=plan, page=page, perception=perception),
                    branch_id=page.branch_id,
                    first_seen_page_id=page.page_id,
                    last_seen_page_id=page.page_id,
                )
                self.state.scenes[scene_id] = scene
            else:
                scene.last_seen_page_id = page.page_id
                scene.description = self._derive_scene_description(plan=plan, page=page, perception=perception)
        else:
            current_scene_id = self.state.current_scene_id or self._build_scene_id(plan=plan, page=page)
            scene = self.state.scenes.get(current_scene_id)
            if scene is None:
                scene = SceneMemory(
                    scene_id=current_scene_id,
                    scene_name=self._derive_scene_name(plan=plan, perception=perception),
                    description=self._derive_scene_description(plan=plan, page=page, perception=perception),
                    branch_id=page.branch_id,
                    first_seen_page_id=page.page_id,
                    last_seen_page_id=page.page_id,
                )
                self.state.scenes[current_scene_id] = scene
            else:
                scene.last_seen_page_id = page.page_id
                scene.description = self._merge_scene_description(scene.description, page.page_summary, plan.next_story_beat)

        if page.page_id not in scene.page_ids:
            scene.page_ids.append(page.page_id)
        self._append_scene_reference(
            scene,
            SceneReference(
                image_uri=image.image_uri or page.image_uri,
                page_id=page.page_id,
                summary=page.page_summary,
                reference_type="full_frame",
            ),
            replace_existing=False,
        )
        scene.update_notes.append(plan.world_update)
        if len(scene.update_notes) > 8:
            scene.update_notes[:] = scene.update_notes[-8:]
        self.state.current_scene_id = scene.scene_id
        return scene

    @staticmethod
    def _build_scene_id(*, plan: PlanningDecision, page: PageRecord) -> str:
        raw = plan.scene_location or plan.branch_label or page.target_label or page.page_id
        normalized = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
        return normalized or page.page_id

    @staticmethod
    def _derive_scene_name(*, plan: PlanningDecision, perception: PerceptionResult) -> str:
        return (
            plan.scene_location
            or perception.focus_subject
            or perception.target_label
            or "Unnamed Scene"
        )

    @staticmethod
    def _derive_scene_description(
        *,
        plan: PlanningDecision,
        page: PageRecord,
        perception: PerceptionResult,
    ) -> str:
        fragments = [
            plan.scene_location,
            page.page_summary,
            plan.next_story_beat,
            perception.action_description,
        ]
        cleaned = [fragment.strip() for fragment in fragments if isinstance(fragment, str) and fragment.strip()]
        if not cleaned:
            return "Scene memory placeholder."
        return " ".join(cleaned[:3])

    @staticmethod
    def _merge_scene_description(existing: str, page_summary: str | None, next_story_beat: str | None) -> str:
        fragments = [existing, page_summary, next_story_beat]
        cleaned = [fragment.strip() for fragment in fragments if isinstance(fragment, str) and fragment.strip()]
        deduped: list[str] = []
        seen: set[str] = set()
        for fragment in cleaned:
            key = fragment.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fragment)
        return " ".join(deduped[:3])

    @staticmethod
    def _append_scene_reference(scene: SceneMemory, reference: SceneReference, *, replace_existing: bool) -> None:
        if replace_existing:
            scene.references[:] = [item for item in scene.references if item.page_id != reference.page_id]
        if any(existing.image_uri == reference.image_uri for existing in scene.references):
            return
        scene.references.append(reference)
        if len(scene.references) > 6:
            scene.references[:] = scene.references[-6:]

    def seed_root_page(
        self,
        *,
        page_id: str,
        image_uri: str,
        prompt: str,
        summary: str,
        render_metadata: dict | None = None,
    ) -> None:
        empty_perception = PerceptionResult(
            target_label=self.state.root_topic,
            region_caption=summary,
            clicked_region=None,
            action=ExplorationAction.REFRAME,
            action_description=f"Establish the opening page for {self.state.root_topic} with a reframed introductory comic panel.",
            user_profile_summary=self.state.user_profile_summary,
            intent_hint="root_page",
            confidence=1.0,
        )
        root_click = ClickEvent(x=0.5, y=0.5, user_hint="root")
        root_page = PageRecord(
            page_id=page_id,
            branch_id="main",
            scene_id="root_scene",
            source_page_id=None,
            image_uri=image_uri,
            prompt=prompt,
            page_summary=summary,
            action=ExplorationAction.REFRAME,
            target_label=self.state.root_topic,
            click=root_click,
            perception=empty_perception,
            plan=None,
            narration=None,
            narration_status="pending",
            narration_error=None,
            render_metadata=dict(render_metadata or {}),
        )
        self.state.pages[page_id] = root_page
        self.state.branches["main"].page_ids.append(page_id)
        self.state.current_page_id = page_id
        self.state.current_scene_id = "root_scene"
        self.state.history.append(page_id)
        self.state.scenes["root_scene"] = SceneMemory(
            scene_id="root_scene",
            scene_name=self.state.root_topic,
            description=summary,
            branch_id="main",
            first_seen_page_id=page_id,
            last_seen_page_id=page_id,
            page_ids=[page_id],
            references=[
                SceneReference(
                    image_uri=image_uri,
                    page_id=page_id,
                    summary=summary,
                    reference_type="full_frame",
                )
            ],
            update_notes=["Initialized root scene."],
        )
