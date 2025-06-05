import requests
from requests.exceptions import RequestException
from vllm import SamplingParams
from typing import List, Dict
from src.inference.vllm_server import (
    ResponseOutput,
    ChatRequest,
    GenerateRequest,
    SamplingConfig,
)


class VLLMClient:
    """
    Minimal client for a vLLM batched-chat & generate server.
    """

    def __init__(self, host: str, port: int, timeout: float = 30.0):
        """
        Args:
          host: Host where the vLLM server is running (e.g. "127.0.0.1").
          port: Port where the vLLM server listens (e.g. 8000).
          timeout: HTTP request timeout in seconds.
        """
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

        # Basic health check
        health_url = f"{self.base_url}/health"
        try:
            resp = requests.get(health_url, timeout=self.timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"Health check returned HTTP {resp.status_code}")
        except Exception as e:
            raise RuntimeError(f"Cannot reach vLLM server at {health_url}: {e!r}")

    def chat(
        self,
        conversations: List[List[Dict[str, str]]],
        sampling_params: SamplingParams,
    ) -> List[List[str]]:
        """
        Send a batch of M conversations to POST /chat, return a list of M generated strings.

        Raises RuntimeError on HTTP errors, non‐200 responses, or malformed JSON.
        """
        url = f"{self.base_url}/chat"

        params = SamplingConfig.model_validate(vars(sampling_params))
        payload = ChatRequest(
            conversations=conversations,
            params=params,
        )

        try:
            response = requests.post(url, json=payload.model_dump(mode="json"), timeout=self.timeout)
        except RequestException as e:
            raise RuntimeError(f"Failed to POST /chat → {e!r}")

        if response.status_code != 200:
            raise RuntimeError(f"/chat returned HTTP {response.status_code}: {response.text}")

        try:
            data = response.json()
        except ValueError as e:
            raise RuntimeError(f"Invalid JSON in /chat response: {e!r}")

        output = ResponseOutput.model_validate(data)
        return output.outputs

    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
    ) -> List[List[str]]:
        """
        Send a batch of N prompts to POST /generate, return a list of N generated strings.

        Raises RuntimeError on HTTP errors, non-200 responses, or malformed JSON.
        """
        url = f"{self.base_url}/generate"

        params = SamplingConfig.model_validate(vars(sampling_params))
        payload = GenerateRequest(
            prompts=prompts,
            params=params,
        )

        try:
            response = requests.post(url, json=payload.model_dump(mode="json"), timeout=self.timeout)
        except RequestException as e:
            raise RuntimeError(f"Failed to POST /generate → {e!r}")

        if response.status_code != 200:
            raise RuntimeError(f"/generate returned HTTP {response.status_code}: {response.text}")

        try:
            data = response.json()
        except ValueError as e:
            raise RuntimeError(f"Invalid JSON in /generate response: {e!r}")

        output = ResponseOutput.model_validate(data)
        return output.outputs
