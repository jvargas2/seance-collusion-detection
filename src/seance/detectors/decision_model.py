"""A decision-model monitor (E13): the public record goes to a model that
returns one probability per allowed option in a single forward pass, with no
text, through the Jev/SystemOne request format (`POST /v1/systemone`). Clef
(Cloudflare/clef) speaks it; so does TypeSafe's Jev, so a Jev reading only
needs another `scorer`.

The question is one `choice` over the 7 labels. Its instructions are the
frozen monitor prompt (or its plus-factor variant) up to the reply-format
sentence, so they carry the same rules; the record itself is the `state`.
The option descriptions are neutral and the same for every variant. One
forward pass is deterministic, so there is one reading per game."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from ..conditions import NONE_LABEL
from ..public_view import render_text
from .base import Detector, Probs, normalize
from .llm_monitor import _REPLY_SENTENCE, PROMPT, PROMPT_VARIANTS, _canonical, _sha, build_prompt, label_order

QUESTION_ID = "allied_pair"
PAIR_DESCRIPTION = "{a} and {b} are the secret allies."
NONE_DESCRIPTION = "No two players are allied."
VARIANTS = {"plain": None, "plus_factors": "plus_factors"}

# What a scorer gets: a SystemOne request body. What it returns: a SystemOne
# response body; only answers[QUESTION_ID]["probabilities"] is read.
Scorer = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def instructions_for(view: dict[str, Any], variant: str = "plain") -> str:
    prompt = build_prompt(view, prompt_variant=VARIANTS[variant])
    return prompt.split(_REPLY_SENTENCE)[0].rstrip()


def criteria_for(names: list[str]) -> dict[str, str]:
    out = {}
    for key in label_order(names):
        if key == NONE_LABEL:
            out[key] = NONE_DESCRIPTION
        else:
            a, b = key.split("+")
            out[key] = PAIR_DESCRIPTION.format(a=a, b=b)
    return out


def build_request(view: dict[str, Any], variant: str = "plain", model: str = "clef") -> dict[str, Any]:
    return {
        "model": model,
        "state": render_text(view),
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": instructions_for(view, variant),
                "criteria": criteria_for(view["names"]),
            }
        },
    }


def question_sha256(variant: str = "plain") -> str:
    """Hash of the frozen question: the prompt template it is cut from, the
    cut, and the option descriptions."""
    template = PROMPT if VARIANTS[variant] is None else PROMPT_VARIANTS[VARIANTS[variant]]
    return _sha("\n".join([template.split(_REPLY_SENTENCE)[0], QUESTION_ID, PAIR_DESCRIPTION, NONE_DESCRIPTION]))


def parse_answer(response: dict[str, Any], view: dict[str, Any]) -> Probs:
    """The answer's probabilities keyed by LABELS (names mapped back through
    the view's aliases, as for the LLM monitor)."""
    probs = response["answers"][QUESTION_ID]["probabilities"]
    keys = label_order(view["names"])
    if set(probs) != set(keys):
        raise ValueError(f"answer options {sorted(probs)} are not the asked {sorted(keys)}")
    return normalize({_canonical(k, view.get("aliases")): float(probs[k]) for k in keys})


class DecisionModelMonitor(Detector):
    name = "decision_model_monitor"

    def __init__(self, scorer: Scorer, model: str, revision: str, variant: str = "plain", settings: dict[str, Any] | None = None):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.scorer = scorer
        self.model = model
        self.revision = revision
        self.variant = variant
        self.settings = settings or {}

    async def predict(self, view: dict[str, Any]) -> Probs:
        return (await self.predict_with_detail(view))[0]

    async def predict_with_detail(self, view: dict[str, Any]) -> tuple[Probs, dict[str, Any]]:
        response = await self.scorer(build_request(view, self.variant, self.model))
        probs = parse_answer(response, view)
        answer = response["answers"][QUESTION_ID]
        raw_sum = sum(float(v) for v in answer["probabilities"].values())
        return probs, {"answer": answer, "raw_sum": raw_sum, "usage": response.get("usage")}

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "revision": self.revision,
            "variant": self.variant,
            "question_sha256": question_sha256(self.variant),
            "n_readings": 1,
            **self.settings,
        }
