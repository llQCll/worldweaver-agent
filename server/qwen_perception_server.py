from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import time
import traceback
from contextlib import nullcontext
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoProcessor

logger = logging.getLogger("qwen_perception_server")


class ChatMessage(BaseModel):
    role: str
    content: Any


class ResponseFormat(BaseModel):
    type: str = "json_object"


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    response_format: ResponseFormat | None = None
    temperature: float = 0.2
    max_tokens: int = 1024


class ServerConfig(BaseModel):
    model_path: str
    served_model_name: str
    api_key: str
    device: str
    cuda_visible_devices: str | None = None
    host: str
    port: int
    torch_dtype: str
    retry_without_cudnn: bool = True
    force_disable_cudnn: bool = False


class QwenPerceptionServer:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.processor = AutoProcessor.from_pretrained(
            config.model_path,
            trust_remote_code=True,
        )
        dtype = getattr(torch, config.torch_dtype, torch.bfloat16)
        model_class, class_name = _resolve_multimodal_model_class()
        logger.info("Using model loader: %s", class_name)
        try:
            self.model = model_class.from_pretrained(
                config.model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map=config.device,
            )
        except ValueError as exc:
            if "qwen3_vl" in str(exc).lower():
                raise RuntimeError(
                    "Your installed transformers build still does not support Qwen3-VL. "
                    "Please upgrade on the server with: "
                    "pip install -U 'transformers>=4.57.3' accelerate safetensors"
                ) from exc
            raise
        self.model.eval()
        self.model_device = _infer_model_device(self.model)

    def create_completion(self, request: ChatCompletionRequest) -> dict[str, Any]:
        rendered_messages = self._prepare_messages(request.messages)
        model_inputs = self.processor.apply_chat_template(
            rendered_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs.pop("token_type_ids", None)
        model_inputs = model_inputs.to(self.model_device)
        try:
            generated = self._generate_with_recovery(
                model_inputs=model_inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        except RuntimeError as exc:
            error_text = str(exc)
            is_cudnn_init_error = "CUDNN_STATUS_NOT_INITIALIZED" in error_text
            is_oom_error = "out of memory" in error_text.lower()
            if is_cudnn_init_error or is_oom_error:
                raise RuntimeError(
                    "Qwen perception inference failed during CUDA execution. "
                    "Try restarting the perception server with --force-disable-cudnn, "
                    "or use a separate GPU with more free memory."
                ) from exc
            raise
        prompt_len = model_inputs["input_ids"].shape[1]
        new_tokens = generated[:, prompt_len:]
        output_text = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        content = self._extract_json_text(output_text)
        now = int(time.time())
        return {
            "id": f"chatcmpl-{now}",
            "object": "chat.completion",
            "created": now,
            "model": self.config.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(prompt_len),
                "completion_tokens": int(new_tokens.shape[1]),
                "total_tokens": int(prompt_len + new_tokens.shape[1]),
            },
        }

    def _generate_with_recovery(
        self,
        *,
        model_inputs: Any,
        max_new_tokens: int,
        temperature: float,
    ):
        try:
            return self._generate(
                model_inputs=model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                disable_cudnn=self.config.force_disable_cudnn,
            )
        except RuntimeError as exc:
            if (
                "CUDNN_STATUS_NOT_INITIALIZED" in str(exc)
                and self.config.retry_without_cudnn
                and not self.config.force_disable_cudnn
            ):
                logger.warning(
                    "Perception model hit cuDNN initialization failure. Clearing CUDA cache and retrying once with cuDNN disabled."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                return self._generate(
                    model_inputs=model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    disable_cudnn=True,
                )
            raise

    def _generate(
        self,
        *,
        model_inputs: Any,
        max_new_tokens: int,
        temperature: float,
        disable_cudnn: bool,
    ):
        cudnn_context = (
            torch.backends.cudnn.flags(enabled=False)
            if disable_cudnn and self.model_device.type == "cuda"
            else nullcontext()
        )
        with cudnn_context:
            with torch.inference_mode():
                return self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )

    def _prepare_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        rendered_messages: list[dict[str, Any]] = []

        for message in messages:
            if isinstance(message.content, list):
                content_items: list[dict[str, Any]] = []
                for item in message.content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text":
                        content_items.append({"type": "text", "text": item.get("text", "")})
                    elif item.get("type") == "image_url":
                        image_url = item.get("image_url", {}).get("url")
                        if image_url:
                            content_items.append(
                                {
                                    "type": "image",
                                    "image": self._decode_image(image_url),
                                }
                            )
                rendered_messages.append({"role": message.role, "content": content_items})
            else:
                rendered_messages.append(
                    {
                        "role": message.role,
                        "content": [{"type": "text", "text": str(message.content)}],
                    }
                )

        return rendered_messages

    @staticmethod
    def _decode_image(data_url: str) -> Image.Image:
        if data_url.startswith("data:"):
            _, encoded = data_url.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        raise ValueError("Only data URL images are supported in the lightweight perception server.")

    @staticmethod
    def _extract_json_text(output_text: str) -> str:
        start = output_text.find("{")
        end = output_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = output_text[start : end + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                logger.warning("Model returned non-parseable JSON candidate, returning raw text.")
        return output_text


def _resolve_multimodal_model_class() -> tuple[type[Any], str]:
    try:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration, "Qwen3VLForConditionalGeneration"
    except ImportError:
        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText, "AutoModelForImageTextToText"
        except ImportError as exc:
            raise RuntimeError(
                "Could not import a multimodal model loader from transformers. "
                "Install or upgrade transformers with: "
                "pip install -U 'transformers>=4.57.3' accelerate safetensors"
            ) from exc


def _infer_model_device(model: Any) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def create_app(server: QwenPerceptionServer) -> FastAPI:
    app = FastAPI(title="Qwen Perception Server")

    @app.get("/v1/models")
    def list_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _check_api_key(server.config.api_key, authorization)
        return {
            "object": "list",
            "data": [
                {
                    "id": server.config.served_model_name,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def create_chat_completion(
        request: ChatCompletionRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_api_key(server.config.api_key, authorization)
        if request.model != server.config.served_model_name:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model '{request.model}'. Expected '{server.config.served_model_name}'.",
            )
        try:
            return server.create_completion(request)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Perception completion failed")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": str(exc),
                    "type": exc.__class__.__name__,
                    "traceback": traceback.format_exc(),
                },
            ) from exc

    return app


def _check_api_key(expected_api_key: str, authorization: str | None) -> None:
    if not expected_api_key:
        return
    if authorization != f"Bearer {expected_api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local Qwen VL model with an OpenAI-compatible API.")
    parser.add_argument(
        "--model-path",
        default="/c20250509/ZhongzhengWang/model/Qwen3_VL_8B",
        help="Path to the local Qwen VL model directory.",
    )
    parser.add_argument(
        "--served-model-name",
        default="Qwen3_VL_8B",
        help="Model id exposed by the server.",
    )
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--cuda-visible-devices",
        default="0,1",
        help="Restrict this perception server to the listed physical GPUs, e.g. '0,1'.",
    )
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument(
        "--force-disable-cudnn",
        action="store_true",
        help="Disable cuDNN for Qwen-VL generation. Useful on some servers when CUDA decode crashes.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        logger.info("Restricting perception server to physical GPUs %s", args.cuda_visible_devices)
    config = ServerConfig(
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        api_key=args.api_key,
        device=args.device,
        cuda_visible_devices=args.cuda_visible_devices,
        host=args.host,
        port=args.port,
        torch_dtype=args.torch_dtype,
        force_disable_cudnn=args.force_disable_cudnn,
    )
    logger.info("Loading local VL model from %s", config.model_path)
    server = QwenPerceptionServer(config)
    app = create_app(server)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
