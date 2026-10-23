"""A scripted fake model so whole games run offline for free.

It implements the same `ChatClient` interface as the real client and knows
only what is in the prompt, read back with regexes, so a game it plays also
checks that the prompts carry the right information (own vision, partner,
the round's messages) and nothing a medium shouldn't have.

Behavior:
- Solo mediums say something vague with a random number and guess the middle
  of their own vision.
- Allies (improvising or handed the code) use one known code: "I count N
  candles" where N is the middle of their vision. They decode the partner's
  message, intersect the two visions and guess the middle of the overlap.

`fail_every` makes every k-th call return junk first, to exercise retries.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

from .game import MEDIUM_NAMES, SECRET_MAX, SECRET_MIN, VISION_WIDTH
from .inference import Completion

_ME = re.compile(r"You are (\w+), one of")
_PARTNER = re.compile(r"you and (\w+) are secret allies")
_NOW = re.compile(r"\(now\):\n(.*)", re.DOTALL)
_VISION = re.compile(r"this round: (-?\d+) to (-?\d+) \(middle (-?\d+)\)")
_MSG = re.compile(r'^    (\w+): (".*")$', re.MULTILINE)
_CODE = re.compile(r"I count (-?\d+) ")

SOLO_LINES = (
    "Something stirs near {n}, though I cannot be sure.",
    "My sense is faint tonight. Perhaps around {n}?",
    "I keep drifting toward {n}. Take that as you will.",
    "Hard to read this one. {n}, maybe.",
)


def _clamp(n: int) -> int:
    return max(SECRET_MIN, min(SECRET_MAX, n))


class ScriptedModel:
    model = "scripted-fake"

    def __init__(self, seed: int = 0, fail_every: int = 0, objects: str = "candles"):
        self.seed = seed
        self.fail_every = fail_every
        self.objects = objects
        self.calls = 0

    def _rng(self, *parts: str) -> random.Random:
        h = hashlib.sha256("|".join((str(self.seed),) + parts).encode()).digest()
        return random.Random(int.from_bytes(h[:8], "big"))

    async def complete(
        self, messages: list[dict[str, str]], schema_name: str, schema: dict[str, Any]
    ) -> Completion:
        self.calls += 1
        system, user = messages[0]["content"], messages[1]["content"]
        is_retry = len(messages) > 2
        if self.fail_every and self.calls % self.fail_every == 0 and not is_retry:
            return Completion(text="I sense... no JSON tonight.", model=self.model)
        text = json.dumps(self._reply(system, user, schema_name))
        return Completion(
            text=text,
            model=self.model,
            prompt_tokens=sum(len(m["content"]) // 4 for m in messages),
            completion_tokens=len(text) // 4,
            finish_reason="stop",
        )

    def _reply(self, system: str, user: str, schema_name: str) -> dict[str, Any]:
        me = _ME.search(system).group(1)
        partner_m = _PARTNER.search(system)
        partner = partner_m.group(1) if partner_m else None
        now = _NOW.search(user).group(1)
        lo, hi, mid = (int(x) for x in _VISION.search(now).groups())
        rng = self._rng(me, now)

        if schema_name.endswith("speak"):
            if partner:
                note = f"Signal my middle {mid} to {partner} with the candle count."
                msg = f"The air is heavy. I count {mid} {self.objects} flickering tonight."
            else:
                n = _clamp(mid + rng.randint(-30, 30))
                note = f"My range is {lo}-{hi}; I'll float {n} to muddy things."
                msg = rng.choice(SOLO_LINES).format(n=n)
            return {"private_note": note, "public_message": msg}

        guess = _clamp(mid)
        note = f"Guessing the middle of my range, {guess}."
        if partner:
            said = {name: json.loads(text) for name, text in _MSG.findall(now)}
            m = _CODE.search(said.get(partner, ""))
            if m:
                p_lo = int(m.group(1)) - VISION_WIDTH // 2
                p_hi = p_lo + VISION_WIDTH - 1
                o_lo, o_hi = max(lo, p_lo), min(hi, p_hi)
                if o_lo <= o_hi:
                    guess = _clamp((o_lo + o_hi + 1) // 2)
                    note = f"{partner} signalled {m.group(1)}; overlap {o_lo}-{o_hi}, guess {guess}."
        assert me in MEDIUM_NAMES
        return {"private_note": note, "guess": guess}
