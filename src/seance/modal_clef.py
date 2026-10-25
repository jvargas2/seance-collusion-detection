"""Clef (Cloudflare/clef, a decision model on a Qwen3.8-27B backbone) on
Modal, for the E13 monitor. It answers Jev/SystemOne request bodies with the
repo's own loader (`joint_schema_model.py`), not vLLM: vLLM serves only the
backbone's chat, not the decision head.

    uv run modal deploy src/seance/modal_clef.py
    uv run python -m seance.clef --run runs/main --variant plain

The release is pinned to one commit of the Hugging Face repo and cached in
its own Volume. Cost guards: one GPU at most, inputs run one at a time (one
forward pass each), and the container stops after two idle minutes.
"""

from __future__ import annotations

import os

import modal

APP_NAME = os.environ.get("SEANCE_MODAL_APP", "seance-clef")
MODEL = "Cloudflare/clef"
REVISION = "a30162b99d68a481d8bd75d38df069a0cf83df40"
GPU = os.environ.get("SEANCE_MODAL_GPU", "H100")
CACHE = "/cache"

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "torch==2.11.0", "torchvision==0.26.0", "transformers==5.10.2", "huggingface_hub", "safetensors", "accelerate", "pillow"
)
cache = modal.Volume.from_name("seance-clef-cache", create_if_missing=True)
app = modal.App(APP_NAME)


@app.cls(image=image, gpu=GPU, max_containers=1, scaledown_window=120, timeout=30 * 60, volumes={CACHE: cache})
class Clef:
    @modal.enter()
    def load(self) -> None:
        import sys

        from huggingface_hub import snapshot_download

        path = snapshot_download(MODEL, revision=REVISION, cache_dir=CACHE)
        cache.commit()
        sys.path.insert(0, path)
        from joint_schema_model import load_release_model

        self.model, self.processor = load_release_model(path, device="cuda")

    @modal.method()
    def answer(self, request: dict) -> dict:
        """A SystemOne response body for one request, with every option's
        probability unrounded (the repo's `systemone` rounds to 4 places)
        and its logit."""
        import torch
        from joint_schema_model import collate_records, encode_record

        encoded = encode_record(self.processor.tokenizer, request, processor=self.processor)
        with torch.inference_mode():
            logits = self.model(collate_records([encoded], self.processor.tokenizer.pad_token_id, torch.device("cuda")))[0]
        answers = {}
        for q, ql in zip(encoded.questions, logits):
            ql = ql.float()
            probs = dict(zip(q.option_ids, ql.softmax(-1).tolist()))
            choice = max(probs, key=probs.__getitem__)
            answers[q.question_id] = {
                "type": request["questions"][q.question_id]["type"],
                "choice": choice,
                "confidence": probs[choice],
                "probabilities": probs,
                "logits": dict(zip(q.option_ids, ql.tolist())),
            }
        return {
            "model": request["model"],
            "revision": REVISION,
            "answers": answers,
            "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
        }
