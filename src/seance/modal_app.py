"""Serve an open model on Modal with vLLM's OpenAI-compatible server.

Setup (once): `uv run modal setup`, then create the API key the server checks:

    uv run modal secret create seance-vllm-key VLLM_API_KEY=<a long random string>

Deploy and point `.env` at it:

    uv run modal deploy src/seance/modal_app.py
    # SEANCE_BASE_URL=<printed URL>/v1, SEANCE_API_KEY=<same key>, SEANCE_MODEL=<model below>

Weights are cached in the `seance-hf-cache` Modal Volume (mounted as the
Hugging Face cache), so only the first cold start downloads them; vLLM's
compile cache has its own volume. Modal commits a volume's changes when the
container shuts down (including idle scaledown).

Thinking mode is turned off per request by the client
(`SEANCE_ENABLE_THINKING=false` sends chat_template_kwargs.enable_thinking).

Cost guards: `max_containers=1` means one GPU at most, however many requests
the runner sends; `@modal.concurrent` lets that one container batch many
requests (vLLM is built for it); `scaledown_window` stops the GPU after a few
idle minutes. Keep the runner's `--concurrency` at or below MAX_INPUTS.

Model and GPU are chosen at deploy time through environment variables and
baked into the image, so the container serves exactly what was deployed:

    SEANCE_MODAL_MODEL=Qwen/Qwen3-8B SEANCE_MODAL_GPU=L40S uv run modal deploy src/seance/modal_app.py
"""

from __future__ import annotations

import json
import os
import subprocess

import modal

VLLM_VERSION = "0.32.0"
MODEL = os.environ.get("SEANCE_MODAL_MODEL", "Qwen/Qwen3.8-27B-FP8")
GPU = os.environ.get("SEANCE_MODAL_GPU", "H100")
MAX_MODEL_LEN = int(os.environ.get("SEANCE_MODAL_MAX_MODEL_LEN", "40960"))
MAX_INPUTS = int(os.environ.get("SEANCE_MODAL_MAX_INPUTS", "32"))
IDLE_MINUTES = int(os.environ.get("SEANCE_MODAL_IDLE_MINUTES", "5"))
# A second model (e.g. a different-family monitor) deploys as its own app with
# its own URL, so the players' server is untouched.
APP_NAME = os.environ.get("SEANCE_MODAL_APP", "seance-vllm")
# The Qwen3 parser for the players' model; another family needs its own parser
# ("" for none). Extra vLLM flags go in SEANCE_MODAL_EXTRA_ARGS as a JSON list
# of strings (the image build can't hold shell quoting).
REASONING_PARSER = os.environ.get("SEANCE_MODAL_REASONING_PARSER", "qwen3")
EXTRA_ARGS = os.environ.get("SEANCE_MODAL_EXTRA_ARGS", "[]")
PORT = 8000

# A CUDA *devel* base, not debian_slim: FlashInfer (used by vLLM) compiles
# some kernels on first use and needs nvcc, or the engine dies at startup
# ("Could not find nvcc"). Its CUDA major (13) matches vLLM 0.32's PyTorch.
CUDA_IMAGE = "nvidia/cuda:13.0.1-devel-ubuntu22.04"

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .entrypoint([])  # drop the CUDA image's banner entrypoint
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "CUDA_HOME": "/usr/local/cuda",
            "SEANCE_MODAL_MODEL": MODEL,
            "SEANCE_MODAL_MAX_MODEL_LEN": str(MAX_MODEL_LEN),
            "SEANCE_MODAL_MAX_INPUTS": str(MAX_INPUTS),
            "SEANCE_MODAL_REASONING_PARSER": REASONING_PARSER,
            "SEANCE_MODAL_EXTRA_ARGS": EXTRA_ARGS,
        }
    )
)

hf_cache = modal.Volume.from_name("seance-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("seance-vllm-cache", create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU,
    max_containers=1,
    scaledown_window=IDLE_MINUTES * 60,
    timeout=24 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("seance-vllm-key")],
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=PORT, startup_timeout=20 * 60)
def serve() -> None:
    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--served-model-name",
        MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--api-key",
        os.environ["VLLM_API_KEY"],
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        str(MAX_INPUTS),
        "--gpu-memory-utilization",
        "0.90",
    ]
    # The reasoning parser puts thinking text in `reasoning_content`, so
    # structured output applies to the answer only (the Qwen3.8 vLLM recipe's
    # setting). With thinking off the template emits an empty think block and
    # the parser passes the answer through unchanged.
    if REASONING_PARSER:
        cmd += ["--reasoning-parser", REASONING_PARSER]
    cmd += json.loads(EXTRA_ARGS)
    subprocess.Popen(cmd)


@app.local_entrypoint()
def main() -> None:
    """`uv run modal run src/seance/modal_app.py`: print the endpoint to put in .env."""
    print(f"model: {MODEL}  gpu: {GPU}")
    print(f"SEANCE_BASE_URL={serve.get_web_url()}/v1")
