from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRecommendation:
    agent_name: str
    responsibility: str
    recommended_models: list[str]
    serving_hint: str
    notes: str


def default_open_source_stack() -> list[ModelRecommendation]:
    return [
        ModelRecommendation(
            agent_name="Perception Agent",
            responsibility="Ground the click inside the current image, caption the local region, and identify objects.",
            recommended_models=[
                "Qwen2.5-VL-7B-Instruct",
                "Qwen2.5-VL-32B-Instruct",
                "InternVL3-8B",
                "Molmo-7B-D",
            ],
            serving_hint="Serve a VLM through vLLM or SGLang with an OpenAI-compatible endpoint.",
            notes=(
                "Use a vision-language model first. The perception model should accept the current image, click "
                "coordinates, and short memory context."
            ),
        ),
        ModelRecommendation(
            agent_name="Memory Agent",
            responsibility="Maintain world state, branches, entity summaries, and long-horizon consistency.",
            recommended_models=[
                "Qwen3-14B",
                "Llama-3.1-8B-Instruct",
                "Mistral-Small-3.1-Instruct",
            ],
            serving_hint="The memory agent can stay mostly symbolic. Use an LLM only for periodic summarization.",
            notes=(
                "Do not rely on an LLM alone for memory. Keep explicit state in JSON or a database, and use the LLM "
                "only to compress branch summaries."
            ),
        ),
        ModelRecommendation(
            agent_name="Planning Agent",
            responsibility="Choose between zoom_in, branch_out, reveal, and reframe, and produce the next-world plan.",
            recommended_models=[
                "Qwen3-14B",
                "Qwen3-32B",
                "DeepSeek-R1-Distill-Qwen-14B",
                "Llama-3.3-70B-Instruct",
            ],
            serving_hint="Use a text-only reasoning model with structured JSON output.",
            notes=(
                "This is the most important LLM in the stack. It needs good reasoning, reliable JSON, and enough "
                "capacity to use memory plus click semantics."
            ),
        ),
        ModelRecommendation(
            agent_name="Retrieval Agent",
            responsibility="Fetch external knowledge when the world needs factual grounding or richer expansion.",
            recommended_models=[
                "bge-m3",
                "Qwen2.5-7B-Instruct",
                "jina-embeddings-v3",
            ],
            serving_hint="Pair a text embedding model with a lightweight planner or reranker.",
            notes=(
                "Retrieval usually does not need a large model on every step. Use embeddings for recall and a small "
                "LLM or reranker to filter evidence."
            ),
        ),
        ModelRecommendation(
            agent_name="Rendering Agent",
            responsibility="Convert the planned next world state into the next conditioned image.",
            recommended_models=[
                "FLUX.1 Kontext-dev",
                "Stable Diffusion 3.5 Large",
                "PixArt-Sigma",
                "Qwen-Image",
            ],
            serving_hint="Use an image generator that supports strong prompt following and, ideally, image conditioning.",
            notes=(
                "This component is often not an LLM. Prioritize image conditioning, subject consistency, and the "
                "ability to preserve world continuity from the previous page."
            ),
        ),
    ]
