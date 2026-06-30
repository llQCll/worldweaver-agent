# WorldWeaver Agent Architecture Plan

## Goal

Refactor the current WorldWeaver agent into clearer functional stages so future work on:

- user intent recognition
- session-level memory and global intent modeling
- narrative control
- frame generation control

can be optimized independently.

This refactor is intentionally behavior-preserving where possible. The immediate goal is not to improve model quality, but to make responsibilities explicit.

## New Functional Split

### 1. Interaction Understanding

Responsibility:

- understand what the user interacted with
- interpret click / box / long press / text hint
- infer local interaction semantics in context

Primary module:

- `worldweaver_agent.interaction.InteractionUnderstandingAgent`

Compatibility alias:

- `worldweaver_agent.perception.PerceptionAgent`

Current output:

- `PerceptionResult`

### 2. Narrative Planning

Responsibility:

- convert interaction understanding into a story-level next-turn decision
- decide how the story should respond
- keep branch, continuity, and story-purpose decisions together

Primary module:

- `worldweaver_agent.narrative_planning.NarrativePlanningAgent`

Compatibility alias:

- `worldweaver_agent.planning.PlanningAgent`

Current output:

- `PlanningDecision`

### 3. Frame Planning

Responsibility:

- translate story-level decisions into a renderable next-frame request
- assemble render conditioning from memory, entities, continuity, and retrieved facts

Primary module:

- `worldweaver_agent.frame_planning.FramePlanningAgent`

Current output:

- `RenderRequest`

### 4. Rendering

Responsibility:

- execute image generation from the frame plan

Primary module:

- `worldweaver_agent.rendering.RenderingAgent`

### 5. Memory

Responsibility today:

- store world state
- pages, branches, entities, summaries
- construct render requests
- commit finished turns

Responsibility in next phase:

- expand from world memory to joint world/user/narrative memory
- support short-term intent tracking
- support long-term user goal tracking
- support explicit global narrative intent

Primary module:

- `worldweaver_agent.memory.MemoryAgent`

## Current Orchestration

The explorer now maps conceptually to:

1. Interaction understanding
2. Narrative planning
3. Retrieval
4. Frame planning
5. Rendering
6. Memory commit

Implemented by:

- `worldweaver_agent.orchestrator.InteractiveWorldExplorer`

## Why This Refactor

Previously:

- `PerceptionAgent` mixed visual grounding and interaction interpretation
- `PlanningAgent` mixed story response and render-oriented prompt organization
- `MemoryAgent` stored state but was not clearly positioned in the control stack

After this refactor:

- interaction semantics are a separate stage
- story decisions are a separate stage
- frame construction is a separate stage

This gives a cleaner path for future research upgrades.

## Planned Next Refactor

The next planned architecture step is to strengthen memory into three explicit layers:

- world memory
- user intent memory
- narrative/global-intent memory

At that point, planning can be further split into:

- user-goal tracking
- narrative policy / story controller
- frame planner

## Compatibility Notes

For now, existing external code can still use:

- `PerceptionAgent`
- `PlanningAgent`
- `InteractiveWorldExplorer`

The new modules provide clearer internal structure without forcing immediate downstream rewrites.
