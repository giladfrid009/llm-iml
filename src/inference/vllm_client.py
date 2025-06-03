import requests
from requests.exceptions import RequestException
from typing import List, Dict, Optional
from src.inference.vllm_server import ResponseOutput


class VLLMClient:
    """
    Minimal client for a vLLM batched‐chat & generate server.

    Usage:
        
        client = VLLMClient(host="127.0.0.1", port=8000, timeout=30.0)

        # Batched chat:
        convs = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's 2+2?"}
            ],
            [
                {"role": "system", "content": "You are a translator."},
                {"role": "user", "content": "Translate 'hello' to French."}
            ],
        ]
        answers = client.chat(
            conversations=convs,
            n=1,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            min_p=0.0,
            max_tokens=32,
            repetition_penalty=1.0,
            stop=None
        )
        # answers -> ["4", "\"bonjour\""]

        # Batched generate (completions):
        prompts = ["Once upon a time,", "In a galaxy far away,"]
        completions = client.generate(
            prompts=prompts,
            n=1,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            min_p=0.0,
            max_tokens=32,
            repetition_penalty=1.0,
            stop=None
        )
        # completions -> ["Once upon a time, there was...", "In a galaxy far away, ..."]
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
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        repetition_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> List[List[str]]:
        """
        Send a batch of M conversations to POST /chat, return a list of M generated strings.

        Raises RuntimeError on HTTP errors, non‐200 responses, or malformed JSON.
        """
        url = f"{self.base_url}/chat"
        payload = {
            "conversations": conversations,
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "stop": stop,
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
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
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        repetition_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> List[List[str]]:
        """
        Send a batch of N prompts to POST /generate, return a list of N generated strings.

        Raises RuntimeError on HTTP errors, non‐200 responses, or malformed JSON.
        """
        url = f"{self.base_url}/generate"
        payload = {
            "prompts": prompts,
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "stop": stop,
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
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
