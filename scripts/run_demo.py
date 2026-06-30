import argparse
import json
import logging
import sys
from pathlib import Path

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
    LlmPerceptionBackend,
    LlmPlanningBackend,
    MemoryAgent,
    MockRenderingBackend,
    OpenAICompatibleJsonBackend,
    PerceptionAgent,
    PlanningAgent,
    RenderRequest,
    RenderingAgent,
    RetrievalAgent,
    RetrievedFact,
    RuleBasedPerceptionBackend,
    RuleBasedPlanningBackend,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small interactive world-agent demo.")
    parser.add_argument("--topic", required=True, help="Root topic of the world.")
    parser.add_argument("--style-guide", default="Editorial infographic world with clear spatial storytelling.")
    parser.add_argument(
        "--root-image-path",
        default=None,
        help="Optional existing root image path. If provided, perception VLM can inspect this image directly.",
    )
    parser.add_argument("--click-x", type=float, default=0.5)
    parser.add_argument("--click-y", type=float, default=0.5)
    parser.add_argument("--user-hint", default=None)
    parser.add_argument(
        "--planning-backend",
        choices=("rule", "llm"),
        default="rule",
        help="Choose whether PlanningAgent uses the rule backend or a real LLM backend.",
    )
    parser.add_argument(
        "--llm-endpoint",
        default="http://127.0.0.1:8000",
        help="OpenAI-compatible local endpoint for the planning LLM backend.",
    )
    parser.add_argument(
        "--llm-model",
        default="/c20250509/ZhongzhengWang/model/Qwen3_14B",
        help="Model name or served model id for the planning backend.",
    )
    parser.add_argument(
        "--llm-api-key",
        default="EMPTY",
        help="API key for the OpenAI-compatible endpoint. Use EMPTY for local services if not required.",
    )
    parser.add_argument(
        "--llm-timeout-seconds",
        type=int,
        default=60,
        help="Timeout for the planning LLM request.",
    )
    parser.add_argument(
        "--perception-backend",
        choices=("rule", "llm"),
        default="rule",
        help="Choose whether PerceptionAgent uses the rule backend or a VLM backend.",
    )
    parser.add_argument(
        "--perception-llm-endpoint",
        default="http://127.0.0.1:8001",
        help="OpenAI-compatible local endpoint for the perception VLM backend.",
    )
    parser.add_argument(
        "--perception-llm-model",
        default="/c20250509/ZhongzhengWang/model/Qwen3_VL_8B",
        help="Model name for the perception VLM backend.",
    )
    parser.add_argument(
        "--perception-llm-api-key",
        default="EMPTY",
        help="API key for the perception VLM endpoint.",
    )
    parser.add_argument(
        "--perception-llm-timeout-seconds",
        type=int,
        default=60,
        help="Timeout for the perception VLM request.",
    )
    parser.add_argument(
        "--rendering-backend",
        choices=("mock", "flux"),
        default="mock",
        help="Choose whether RenderingAgent uses the mock backend or real FLUX pipeline.",
    )
    parser.add_argument(
        "--flux-model-path",
        default="/c20250509/ZhongzhengWang/model/FLUX.1-dev",
        help="Path to the FLUX.1-dev model directory.",
    )
    parser.add_argument(
        "--flux-device",
        default="cuda",
        help="Device for FLUX rendering, for example 'cuda', 'cuda:2', or 'cpu'.",
    )
    parser.add_argument(
        "--flux-output-dir",
        default="output",
        help="Directory to save FLUX-generated images.",
    )
    parser.add_argument(
        "--flux-steps",
        type=int,
        default=28,
        help="Number of FLUX inference steps.",
    )
    parser.add_argument(
        "--flux-guidance-scale",
        type=float,
        default=3.5,
        help="FLUX guidance scale.",
    )
    parser.add_argument(
        "--flux-height",
        type=int,
        default=1024,
        help="FLUX output image height.",
    )
    parser.add_argument(
        "--flux-width",
        type=int,
        default=1024,
        help="FLUX output image width.",
    )
    parser.add_argument(
        "--flux-image-conditioning-mode",
        choices=("off", "img2img", "ip_adapter"),
        default="off",
        help="Choose conditioning strategy. 'ip_adapter' prioritizes entity references for identity consistency while keeping the scene free.",
    )
    parser.add_argument(
        "--flux-img2img-strength",
        type=float,
        default=0.35,
        help="Strength for FLUX img2img mode. Lower keeps more consistency, higher allows larger changes.",
    )
    parser.add_argument(
        "--flux-ip-adapter-model-path",
        default=None,
        help="Path or repo id for the FLUX IP-Adapter weights.",
    )
    parser.add_argument(
        "--flux-ip-adapter-weight-name",
        default=None,
        help="Weight filename for the FLUX IP-Adapter.",
    )
    parser.add_argument(
        "--flux-ip-adapter-subfolder",
        default="",
        help="Optional subfolder containing the IP-Adapter weights.",
    )
    parser.add_argument(
        "--flux-ip-adapter-image-encoder-path",
        default=None,
        help="Optional local path or model id for the IP-Adapter image encoder.",
    )
    parser.add_argument(
        "--flux-ip-adapter-image-encoder-subfolder",
        default="",
        help="Optional subfolder for the IP-Adapter image encoder.",
    )
    parser.add_argument(
        "--flux-ip-adapter-scale",
        type=float,
        default=0.6,
        help="Adapter scale for identity consistency. Higher is stronger identity lock.",
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
        help="Disable cuDNN during FLUX inference. Useful on some servers when VAE decode crashes.",
    )
    parser.add_argument(
        "--preload-models",
        dest="preload_models",
        action="store_true",
        help="Preload rendering models during startup. This is now the default behavior.",
    )
    parser.add_argument(
        "--lazy-load-models",
        dest="preload_models",
        action="store_false",
        help="Delay rendering model loading until first use. This reduces startup time but makes the first render slower.",
    )
    parser.set_defaults(preload_models=True, flux_disable_cpu_offload=True)
    parser.add_argument("--output-json", default=None, help="Optional path to save the final world state.")
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


def build_root_image_frame(
    args: argparse.Namespace,
    *,
    memory: MemoryAgent,
    rendering_agent: RenderingAgent,
) -> ImageFrame:
    prompt = f"Opening page for {args.topic}"
    summary = f"Entry page introducing the world of {args.topic}."

    if args.root_image_path:
        root_path = Path(args.root_image_path)
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
            target_label=args.topic,
            action=ExplorationAction.REFRAME,
            render_prompt=(
                f"Create an opening WorldWeaver-style world page for {args.topic}. "
                f"{args.style_guide}. Show several visually distinct and semantically meaningful regions "
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    memory = MemoryAgent.create(root_topic=args.topic, style_guide=args.style_guide)

    retrieval_corpus = [
        RetrievedFact(
            query=args.topic,
            snippet=f"{args.topic} often benefits from branching into subtopics and hidden context.",
            source_title="Seed Knowledge",
            source_url=None,
            confidence=0.7,
        ),
        RetrievedFact(
            query=args.topic,
            snippet=f"A coherent world about {args.topic} should preserve entities, style, and exploration history.",
            source_title="World Design Note",
            source_url=None,
            confidence=0.75,
        ),
    ]
    planning_agent = build_planning_agent(args)
    perception_agent = build_perception_agent(args)
    rendering_agent = build_rendering_agent(args)
    image = build_root_image_frame(
        args,
        memory=memory,
        rendering_agent=rendering_agent,
    )
    explorer = InteractiveWorldExplorer(
        perception_agent=perception_agent,
        memory_agent=memory,
        planning_agent=planning_agent,
        retrieval_agent=RetrievalAgent(InMemoryRetrievalBackend(retrieval_corpus)),
        rendering_agent=rendering_agent,
    )

    if args.perception_backend == "llm" and (not image.image_uri or image.image_uri.startswith("mock://")):
        logging.warning(
            "Perception backend is 'llm' but the root image is still mock. "
            "Use --root-image-path or --rendering-backend flux to provide a real image to the VLM.",
        )

    click = ClickEvent(x=args.click_x, y=args.click_y, user_hint=args.user_hint)
    turn = explorer.step(image=image, click=click)

    logging.info("Perception target: %s", turn.perception.target_label)
    logging.info("Planned action: %s", turn.plan.action.value)
    logging.info("Next prompt: %s", turn.render_result.revised_prompt)
    logging.info("Generated page URI: %s", turn.render_result.image_uri)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(turn.world_state, indent=2), encoding="utf-8")
        logging.info("Saved world state to %s", output_path)


if __name__ == "__main__":
    main()
