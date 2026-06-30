from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("qwen_planning_server")


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
    host: str
    port: int
    torch_dtype: str


class QwenPlanningServer:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True,
        )
        dtype = getattr(torch, config.torch_dtype, torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=config.device,
        )
        self.model.eval()

    def create_completion(self, request: ChatCompletionRequest) -> dict[str, Any]:
        prompt = self._build_prompt(request.messages)
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=request.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0][model_inputs["input_ids"].shape[1] :]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
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
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(model_inputs["input_ids"].shape[1]),
                "completion_tokens": int(new_tokens.shape[0]),
                "total_tokens": int(model_inputs["input_ids"].shape[1] + new_tokens.shape[0]),
            },
        }

    def _build_prompt(self, messages: list[ChatMessage]) -> str:
        rendered_messages: list[dict[str, str]] = []
        for message in messages:
            if isinstance(message.content, list):
                text_parts = []
                for item in message.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                content = "\n".join(text_parts)
            else:
                content = str(message.content)
            rendered_messages.append({"role": message.role, "content": content})

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                rendered_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return "\n".join(f"{item['role']}: {item['content']}" for item in rendered_messages) + "\nassistant:"

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


def create_app(server: QwenPlanningServer) -> FastAPI:
    app = FastAPI(title="Qwen Planning Server")

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
        return server.create_completion(request)

    return app


def _check_api_key(expected_api_key: str, authorization: str | None) -> None:
    if not expected_api_key:
        return
    if authorization != f"Bearer {expected_api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local Qwen model with an OpenAI-compatible API.")
    parser.add_argument(
        "--model-path",
        default="/c20250509/ZhongzhengWang/model/Qwen3_8B",
        help="Path to the local Qwen model directory.",
    )
    parser.add_argument(
        "--served-model-name",
        default="Qwen3_8B",
        help="Model id exposed by the server.",
    )
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--device",
        default="auto",
        help="transformers device_map value, e.g. 'auto', 'cuda', or 'cpu'.",
    )
    parser.add_argument("--torch-dtype", default="bfloat16")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    config = ServerConfig(
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        api_key=args.api_key,
        device=args.device,
        host=args.host,
        port=args.port,
        torch_dtype=args.torch_dtype,
    )
    logger.info("Loading local model from %s", config.model_path)
    server = QwenPlanningServer(config)
    app = create_app(server)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
