# WorldWeaver Agent

WorldWeaver is a research prototype for interactive narrative image generation. It turns image generation into a closed-loop experience: the system generates a visual story panel, the user clicks or gives feedback, the agent interprets the intent, continues the narrative, and gradually builds a preference profile from the interaction trajectory.

The project is designed around one central research idea:

> Interactive image generation should not only produce the next image. It should also preserve narrative continuity, adapt to user preference, and actively collect evidence that refines a user model over time.

## Research Framing

This repository explores three connected layers.

1. **Interactive image generation**

   The user does not write a full prompt every turn. Instead, they interact with the current image by clicking a region and optionally providing a short hint. The system interprets the click as an interaction intent, proposes candidate intents when confidence is low, and uses the confirmed intent to generate the next story panel.

2. **Preference-aware narrative consistency**

   The system maintains world memory, story outline, branch state, entities, prior pages, and user preference signals. Each new image is planned as a continuation of the current narrative rather than an isolated prompt. User feedback changes future generation by increasing or decreasing preference dimensions such as narrative alignment, visual alignment, affective alignment, continuity, mystery, novelty, and agency.

3. **Image generation as active user profiling**

   Generated images are also probes. By observing what the user clicks, confirms, likes, rejects, or redirects, the system updates a lightweight user profile. Over multiple panels, the system can raise confidence in specific preference dimensions and produce stage reports that summarize the emerging profile and suggest the next exploration direction.

In paper terms, the project can be described as a **closed-loop framework for interactive narrative image generation and progressive user modeling**.

## Core Features

- Clickable web interface for interactive visual storytelling.
- OpenAI-compatible text planning backend.
- OpenAI-compatible vision/perception backend.
- OpenAI-compatible image generation backend, including `gpt-image-2` style APIs.
- Intent confirmation flow when the system is uncertain about a click.
- Story outline generation and revision across turns.
- Post-panel feedback options for personalization.
- User profile signals updated from both weak implicit behavior and stronger explicit feedback.
- Stage report after every 5 generated panels.
- Next-round prompt input for steering the following story arc.
- Preview, rollback, and resume controls for previous story states.
- User trajectory logs with `trajectory.md` and `profile_snapshot.json` snapshots.

## System Architecture

The current system is organized as a multi-agent pipeline:

| Component | Responsibility |
| --- | --- |
| `InteractionUnderstandingAgent` / `PerceptionAgent` | Understand the clicked image region and infer local user intent. |
| `NarrativePlanningAgent` / `PlanningAgent` | Convert interaction intent into a story-level next action. |
| `FramePlanningAgent` | Translate narrative decisions into renderable image prompts and continuity constraints. |
| `RenderingAgent` | Generate the next image through mock, local, or OpenAI-compatible backends. |
| `MemoryAgent` | Store pages, branches, entities, world state, user preference signals, and feedback history. |
| Web demo backend | Provides FastAPI endpoints for sessions, clicks, intent confirmation, feedback, rollback, narration, and stage reports. |

The typical loop is:

```text
current image
  -> user click / hint
  -> perception and intent options
  -> intent confirmation
  -> narrative planning
  -> frame planning
  -> image rendering
  -> memory update
  -> optional feedback
  -> user profile update
```

## Repository Layout

```text
src/worldweaver_agent/
  backends.py              # LLM/VLM/image backend adapters and prompt logic
  memory.py                # world memory, user preference memory, checkpoints
  schemas.py               # shared data models
  interaction.py           # interaction understanding agent
  narrative_planning.py    # story-level planning agent
  frame_planning.py        # render-request planning agent
  rendering.py             # rendering agent
  orchestrator.py          # end-to-end explorer loop
scripts/
  run_web_demo_openai.py   # main OpenAI-compatible web demo
  run_web_demo.py          # earlier web demo
  run_demo.py              # CLI demo
server/
  qwen_planning_server.py  # lightweight OpenAI-compatible planning server
  qwen_perception_server.py# lightweight OpenAI-compatible perception server
web/
  index_openai.html        # current interactive frontend
  index.html               # earlier frontend
docs/
  worldweaver_agent_paper_architecture.svg
  story_outlines/
```

## Installation

Python 3.10 or newer is recommended.

```bash
pip install -e .
```

The base package installs FastAPI, Uvicorn, OpenAI SDK, and Pydantic. If you use local model servers, install their model/runtime dependencies separately.

## Quick Start

### 1. Mock rendering mode

Use this mode to test the web interaction flow without spending image generation tokens.

```bash
python scripts/run_web_demo_openai.py \
  --host 127.0.0.1 \
  --port 7860 \
  --planning-backend rule \
  --perception-backend rule \
  --narration-backend rule \
  --rendering-backend mock
```

Open:

```text
http://127.0.0.1:7860
```

### 2. OpenAI-compatible image generation

Set your image API key in the environment first:

```bash
set OPENAI_API_KEY=your_api_key_here
```

Then start the web demo:

```bash
python scripts/run_web_demo_openai.py \
  --host 127.0.0.1 \
  --port 7860 \
  --planning-backend llm \
  --llm-endpoint http://127.0.0.1:8000 \
  --llm-model Qwen3_8B \
  --llm-api-key EMPTY \
  --perception-backend llm \
  --perception-llm-endpoint http://127.0.0.1:8001 \
  --perception-llm-model Qwen3_VL_8B \
  --perception-llm-api-key EMPTY \
  --narration-backend llm \
  --rendering-backend openai_image \
  --openai-image-api-key %OPENAI_API_KEY% \
  --openai-image-model gpt-image-2 \
  --openai-image-output-dir output_openai \
  --openai-image-size 1024x1024 \
  --openai-image-quality high
```

If your image provider is OpenAI-compatible but not the default OpenAI endpoint, pass:

```bash
--openai-image-base-url https://your-provider.example/v1
```

### 3. Local OpenAI-compatible planning/perception servers

Planning server:

```bash
python server/qwen_planning_server.py \
  --model-path /path/to/Qwen3_8B \
  --served-model-name Qwen3_8B \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key EMPTY
```

Perception server:

```bash
python server/qwen_perception_server.py \
  --model-path /path/to/Qwen3_VL_8B \
  --served-model-name Qwen3_VL_8B \
  --host 127.0.0.1 \
  --port 8001 \
  --api-key EMPTY
```

## Web Interaction Flow

1. Start a new session from a topic, style guide, optional root image, and optional story outline.
2. The system creates an opening panel.
3. The user clicks a region in the image.
4. The system parses the click and may ask the user to confirm one of several textual intent options.
5. After confirmation, the planning and rendering pipeline generates the next panel.
6. The user can give feedback such as liking the direction, marking drift, or asking to change mood.
7. The preference model updates and future panels adapt.
8. After enough panels, the stage report summarizes the emerging preference profile and suggests a next-round direction.

## User Modeling

The user profile is updated from two signal types:

- **Implicit signals:** click location, clicked subject, focus type, inferred intent, action type, story role, and emotion hint.
- **Explicit signals:** feedback buttons with weighted axes such as narrative alignment, visual alignment, affective alignment, continuity, novelty, mystery, and agency.

The frontend currently displays:

- a natural-language profile summary,
- top preference dimensions,
- stage-level dimension bars,
- suggestions for the next exploration round.

Trajectory logs are written under:

```text
user_trajectory_logs/<session_id>/
```

These logs are ignored by Git because they can contain private user behavior and generated content.

## Paper-Oriented Contribution Statement

A concise contribution statement for the project:

> We present WorldWeaver, an interactive narrative image generation framework that unifies visual generation, user interaction, and progressive user modeling. Unlike one-shot text-to-image systems, WorldWeaver treats each generated image as both a narrative continuation and an interaction probe. The framework maintains story-level continuity, adapts future panels to user preference signals, and incrementally refines a user profile from clicks and feedback.

Possible evaluation directions:

- Narrative continuity across generated panels.
- Alignment between generated panels and user-confirmed intent.
- Preference adaptation after explicit feedback.
- Profile stability and confidence growth over multiple interaction rounds.
- User study comparing click-based interaction against prompt-only generation.

## Development Notes

Run a syntax check after code changes:

```bash
python -m compileall src scripts
```

Generated images, local state, trajectory logs, virtual environments, and ad-hoc secret-bearing test files should not be committed.

## Security Notes

Do not hard-code API keys in source files. Use environment variables or local config files that are excluded by `.gitignore`.

If an API key was ever committed or shared accidentally, revoke and rotate it before publishing the repository.
