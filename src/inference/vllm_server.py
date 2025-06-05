import os
import sys
import argparse
import json
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import uvicorn

from vllm import LLM, SamplingParams

app = FastAPI(title="vLLM Batched‐Chat & Generate Server")

# This will hold the single LLM instance once we call `server_main(...)`
_llm_instance: Optional[LLM] = None


class SamplingConfig(BaseModel):
    """Schema mirroring :class:`vllm.SamplingParams`."""

    model_config = ConfigDict(extra="allow")

    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: int = 16
    repetition_penalty: float = 1.0
    stop: Optional[List[str]] = None

    def to_sampling_params(self) -> SamplingParams:
        return SamplingParams(**self.model_dump())


class ChatRequest(BaseModel):
    """
    JSON schema for batched chat requests.
    - conversations: list of M conversations; each conversation is a list of messages,
      where each message is {"role": str, "content": str}.
    """

    conversations: List[List[Dict[str, str]]]
    params: SamplingConfig


class GenerateRequest(BaseModel):
    """
    JSON schema for batched generation (text completion) requests.
    - prompts: list of N prompt strings.
    """

    prompts: List[str]
    params: SamplingConfig


class ResponseOutput(BaseModel):
    """
    Response output model for chat and generate endpoints.
    - outputs: The output sequences of the request.
        Each distinct output is represented as a list of strings, where the number of items
        in the list corresponds to the number of outputs generated for each input.
    """

    outputs: List[List[str]]


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Simple health‐check. Returns 200 OK if the server is up.
    """
    return {"status": "ok"}


@app.post("/chat", response_model=ResponseOutput)
async def chat_endpoint(request: ChatRequest) -> ResponseOutput:
    """
    Batched chat endpoint. Expects JSON matching ChatRequest, calls `llm.chat(...)`
    on the entire batch, and returns a list of M generated strings.
    """
    global _llm_instance
    if _llm_instance is None:
        raise HTTPException(status_code=500, detail="LLM not initialized.")

    sampling_params = request.params.to_sampling_params()

    try:
        req_outputs = _llm_instance.chat(request.conversations, sampling_params=sampling_params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during vLLM inference: {e!r}")

    responses = []
    for res in req_outputs:
        responses.append([o.text for o in res.outputs])
    return ResponseOutput(outputs=responses)


@app.post("/generate", response_model=ResponseOutput)
async def generate_endpoint(request: GenerateRequest) -> ResponseOutput:
    """
    Batched generate endpoint. Expects JSON matching GenerateRequest, calls
    `llm.generate(...)` on the entire batch of N prompts, and returns a list of N strings.
    """
    global _llm_instance
    if _llm_instance is None:
        raise HTTPException(status_code=500, detail="LLM not initialized.")

    sampling_params = request.params.to_sampling_params()

    try:
        req_outputs = _llm_instance.generate(request.prompts, sampling_params=sampling_params, use_tqdm=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during vLLM inference: {e!r}")

    responses = []
    for res in req_outputs:
        responses.append([o.text for o in res.outputs])
    return ResponseOutput(outputs=responses)


def server_main(
    model_name: str,
    host: str,
    port: int,
    gpus: str,
    llm_kwargs: Optional[str] = None,
) -> None:
    """
    Entrypoint to run the FastAPI server. Called when this file is run with `--serve`.
    1) Sets CUDA_VISIBLE_DEVICES so only the given GPUs are visible to vLLM.
    2) Initializes a single vLLM( model=model_name ).
    3) Launches uvicorn(app) at host:port.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    global _llm_instance
    extra = json.loads(llm_kwargs) if llm_kwargs else {}
    _llm_instance = LLM(
        model=model_name,
        **extra,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Batched Chat & Generate Server")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run as FastAPI server (invokes server_main).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model ID or local path (e.g. 'meta-llama/Llama-3.2-1b-Instruct').",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host IP to bind the FastAPI server to.",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the FastAPI server to.")
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU indices visible to this process (e.g. '0' or '0,1').",
    )
    parser.add_argument(
        "--llm_kwargs",
        type=str,
        default=None,
        help="JSON string with additional arguments passed to vllm.LLM",
    )
    args = parser.parse_args()

    if args.serve:
        if args.model is None:
            print("ERROR: --model must be specified when using --serve", file=sys.stderr)
            sys.exit(1)

        server_main(
            model_name=args.model,
            host=args.host,
            port=args.port,
            gpus=args.gpus,
            llm_kwargs=args.llm_kwargs,
        )

        sys.exit(0)

    # If not invoked with --serve, do nothing (this file is meant to be imported or launched with --serve).
