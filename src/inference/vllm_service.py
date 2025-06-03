import os
import sys
import socket
import time
import subprocess
import atexit
from typing import List, Dict, Optional
import requests

from src.inference.vllm_client import VLLMClient

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLLMServer:
    """
    Spawns a subprocess running `python vllm_server.py --serve ...`.
    The subprocess hosts a FastAPI server that loads vLLM on specified GPUs,
    and answers /health, /chat, and /generate.

    Usage:
    
        server = VLLMServer(
            model_name="meta-llama/Llama-3.2-1b-Instruct",
            gpu_ids=[0,1],
            host="127.0.0.1",
            port=None  # auto‐pick a free port
        )
        server.start()
        # Now /health, /chat, /generate are available at server.host:server.port
        server.shutdown()
    """

    def __init__(
        self,
        model_name: str,
        gpu_ids: List[int],
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        dtype: str = "bfloat16",
        startup_timeout: float = 15.0,
        server_script_path: Optional[str] = None,
    ):
        """
        Args:
            model_name: HF model ID (e.g. "meta-llama/Llama-3.2-1b-Instruct").
            gpu_ids: List of GPU indices (e.g. [0], or [0,1]).
            host: Host/IP for the server (default "127.0.0.1").
            port: If None, auto-pick a free port; otherwise bind exactly to this port.
            dtype: Data type for model weights (e.g. "bfloat16", "float16", "float32").
            startup_timeout: Seconds to wait for GET /health to return 200.
            server_script_path: Path to vllm_server.py. If None, assume same directory.
        """
        self.model_name = model_name
        self.gpu_ids = gpu_ids.copy()
        self.host = host
        self.dtype = dtype
        self.startup_timeout = startup_timeout

        # Determine port
        if port is None:
            self.port = self._find_free_port()
        else:
            self.port = port

        # Locate vllm_server.py
        if server_script_path is None:
            this_dir = os.path.dirname(os.path.realpath(__file__))
            self.server_script = os.path.join(this_dir, "vllm_server.py")
        else:
            self.server_script = server_script_path

        if not os.path.isfile(self.server_script):
            raise FileNotFoundError(f"Cannot find server script at: {self.server_script}")

        self._process: Optional[subprocess.Popen] = None
        self._is_shut_down = False

        # Ensure cleanup at interpreter exit
        atexit.register(self._atexit_shutdown)

    @staticmethod
    def _find_free_port() -> int:
        """Ask the OS for a free TCP port (bind to port=0)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def start(self) -> None:
        """
        Launch a subprocess:
            python vllm_server.py --serve --model <model_name>
                --host <host> --port <port> --gpus <comma-joined gpu_ids>
                --dtype <dtype>

        Then poll GET /health until HTTP 200 or timeout. Raises RuntimeError on failure.
        """
        if self._process is not None:
            raise RuntimeError("vLLM server is already running.")

        # 1) Build subprocess env: inherit but override CUDA_VISIBLE_DEVICES
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpu_ids)

        # 2) Build the command
        cmd = [
            sys.executable,
            self.server_script,
            "--serve",
            "--model",
            self.model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--gpus",
            ",".join(str(g) for g in self.gpu_ids),
            "--dtype",
            self.dtype,
        ]
        logger.info(f"[VLLMServer] Launching subprocess:\n    {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,  # get strings
            )
        except Exception as e:
            raise RuntimeError(f"[VLLMServer] Failed to launch subprocess: {e!r}")

        # Poll GET /health until 200 or timeout
        health_url = f"http://{self.host}:{self.port}/health"
        t0 = time.time()
        while True:
            # 1) If subprocess died, capture logs and error
            if self._process.poll() is not None:
                out, err = self._process.communicate(timeout=1)
                raise RuntimeError("[VLLMServer] Subprocess terminated prematurely.\n" f"STDOUT:\n{out}\nSTDERR:\n{err}")
            # 2) Try health endpoint
            try:
                resp = requests.get(health_url, timeout=1.0)
                if resp.status_code == 200:
                    logger.info(f"[VLLMServer] Server is healthy at {health_url}")
                    break
            except Exception:
                pass

            # 3) Timeout?
            if time.time() - t0 > self.startup_timeout:
                self._terminate_process()
                raise RuntimeError(f"[VLLMServer] Timeout ({self.startup_timeout}s) waiting for health check.")
            time.sleep(0.1)

    def is_running(self) -> bool:
        """
        Return True if the subprocess exists and is still alive.
        """
        return self._process is not None and (self._process.poll() is None)

    def shutdown(self) -> None:
        """
        Gracefully terminate the server subprocess. Idempotent.
        """
        if self._is_shut_down:
            return
        self._is_shut_down = True

        if self._process is None:
            return

        logger.info("[VLLMServer] Shutting down subprocess...")
        self._terminate_process()
        self._process = None

    def _terminate_process(self) -> None:
        """
        Helper: send SIGTERM, wait 5s, then SIGKILL if still alive.
        """
        if self._process is None:
            return
        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("[VLLMServer] Did not exit after SIGTERM; forcing kill.")
                self._process.kill()
                self._process.wait(timeout=5)
        except Exception as e:
            logger.error(f"[VLLMServer] Error terminating subprocess: {e!r}")

    def _atexit_shutdown(self) -> None:
        """
        Called automatically at interpreter exit to ensure cleanup.
        """
        try:
            self.shutdown()
        except Exception:
            pass

    def __del__(self):
        # Fallback in case shutdown wasn’t explicitly called
        self._atexit_shutdown()


class VLLMService:
    """
    Convenience wrapper bundling VLLMServer + VLLMClient. You only need to interact
    with this one class:

        service = VLLMBatchedService(
            model_name="meta-llama/Llama-3.2-1b-Instruct",
            gpu_ids=[0,1],
            host="127.0.0.1",
            port=None
        )
        service.start()
        chat_answers = service.chat([...])         # batched chat
        gen_answers  = service.generate([...])     # batched generate
        service.shutdown()

    The server subprocess is spawned in `start()`, and the client is automatically created.
    """

    def __init__(
        self,
        model_name: str,
        gpu_ids: List[int],
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        dtype: str = "bfloat16",
        startup_timeout: float = 15.0,
        client_timeout: float = 30.0,
        server_script_path: Optional[str] = None,
    ):
        """
        Args:
            model_name: HF model ID (e.g. 'meta-llama/Llama-3.2-1b-Instruct').
            gpu_ids: List of GPU indices visible to server (e.g. [0], [0,1]).
            host: Host for the server (default '127.0.0.1').
            port: If None, auto-pick a free port. Otherwise bind exactly to this port.
            dtype: Data type for model weights (e.g. 'bfloat16', 'float16', 'float32').
            startup_timeout: Wait time (s) for server health check.
            client_timeout: HTTP timeout (s) for client operations.
            server_script_path: Path to `vllm_server.py`. If None, assume same directory.
        """
        self.server = VLLMServer(
            model_name=model_name,
            gpu_ids=gpu_ids,
            host=host,
            port=port,
            dtype=dtype,
            startup_timeout=startup_timeout,
            server_script_path=server_script_path,
        )
        self.client: Optional[VLLMClient] = None
        self.client_timeout = client_timeout

    def start(self) -> None:
        """
        1) Start the server subprocess (loads model on GPUs, polls health).
        2) Instantiate VLLMClient pointing to that server.
        """
        self.server.start()
        self.client = VLLMClient(host=self.server.host, port=self.server.port, timeout=self.client_timeout)

    def is_running(self) -> bool:
        """Return True if the server subprocess is still alive."""
        return self.server.is_running()

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
        Batched chat.
        Returns a list of lists, each sub-list corresponds to number of outputs (n).
        """
        if self.client is None:
            raise RuntimeError("Service not started. Call .start() first.")
        return self.client.chat(
            conversations=conversations,
            n=n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            stop=stop,
        )

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
        Batched generate.
        Returns a list of lists, each sub-list corresponds to number of outputs (n).
        Raises if not started.
        """
        if self.client is None:
            raise RuntimeError("Service not started. Call .start() first.")
        return self.client.generate(
            prompts=prompts,
            n=n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            stop=stop,
        )

    def shutdown(self) -> None:
        """Shut down the server subprocess. Idempotent."""
        self.server.shutdown()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
