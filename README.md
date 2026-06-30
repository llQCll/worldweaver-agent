# WorldWeaver Agent

WorldWeaver Agent is a research prototype for interactive narrative image generation. It explores a simple idea: image generation can become a continuous conversation between a user and a visual story world, rather than a sequence of isolated prompts.

In WorldWeaver, the user enters a visual scene, clicks on what they care about, confirms or redirects the system's interpretation, and gradually shapes the next panels of the story. Each generated image is both a continuation of the narrative and a small probe into the user's preferences.

## Core Idea

The project connects three layers:

- **Interactive image generation:** users guide the story by interacting with images instead of repeatedly writing full prompts.
- **Preference-aware narrative continuity:** the system tries to keep the story coherent while adapting tone, focus, pacing, and visual direction to user preference.
- **Progressive user modeling:** clicks, choices, and feedback gradually refine a user profile, allowing the system to become more confident about what kind of narrative experience the user wants.

In paper terms, WorldWeaver can be framed as a closed-loop framework for interactive visual storytelling and incremental user preference modeling.

## Experience

A typical session begins with an opening image. The user clicks on a character, object, place, or visual detail that interests them. The system interprets the click as a possible narrative intent, asks for confirmation when needed, and then generates the next image as a story continuation.

After each generated panel, the user can give lightweight feedback. Over time, the system builds an increasingly specific profile of the user's preferences, such as whether they favor mystery, emotional intimacy, visual spectacle, slower exploration, stronger agency, or tighter story continuity.

The interaction is meant to feel less like controlling a tool and more like wandering through a living illustrated story that learns what kind of journey the user wants.

## Research Motivation

Most image generation systems treat the user prompt as the main input and the generated image as the final output. WorldWeaver instead treats generation as an ongoing loop:

```text
image -> interaction -> interpretation -> story continuation -> feedback -> user model -> next image
```

This loop makes the image serve two purposes at once. It advances the narrative, and it creates an opportunity to observe user preference. A panel can reveal what the user chooses to follow, what they ignore, what they correct, and what they want more or less of.

## Contributions

- A closed-loop interaction model for visual narrative generation.
- A narrative continuation mechanism that balances local user intent with broader story coherence.
- A user preference model that evolves from both implicit interaction and explicit feedback.
- A stage-based reflection mechanism that summarizes the emerging user profile and suggests future story directions.
- A research framing where generated images are not only outputs, but also active probes for user understanding.

## Possible Evaluation Directions

- Whether generated panels remain narratively coherent across multiple turns.
- Whether the system correctly follows user-confirmed visual intent.
- Whether feedback improves alignment with user preference in later panels.
- Whether the inferred user profile becomes more stable and specific over time.
- Whether users feel more agency and continuity compared with prompt-only image generation.

## Paper-Framing Statement

WorldWeaver Agent presents interactive image generation as a joint process of storytelling and user modeling. By turning each generated panel into both a narrative step and a preference probe, the system aims to produce visual stories that are continuous, adaptive, and increasingly personalized over time.
