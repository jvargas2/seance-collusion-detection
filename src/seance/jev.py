"""E14: score clean and coalition games with TypeSafe's Jev, through the same
decision-model question as Clef (E13): the same instructions (plain or with
the plus-factor paragraph), option descriptions and record, every game read
with the E8 random medium names, so readings pair with `clef_*` and with
`llm_monitor_pf_standard` / `llm_monitor_pf_plus_factors`.

    uv run python -m seance.jev --run runs/fresh --run runs/neutral --variant plain --variant plus_factors

Writes `predictions/jev_<variant>.jsonl` in each run (resumable). The key
comes from TYPESAFE_API_KEY or ~/.config/typesafe/api_key and is never
printed or written. Requests ask for `jev-latest`; before scoring, one tiny
probe request learns which version the alias points to, it goes into every
prediction as the revision, and any response from another version stops the
run. Spend is the billed input tokens times the list price; the run stops
before COST_CAP_USD. `--compare` pairs both Jev versions with both E8
monitors and both Clef versions into runs/main/metrics/exploratory_jev.json."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .clef import compare, print_comparison
from .conditions import CLEAN, COALITION
from .detectors.decision_model import VARIANTS, DecisionModelMonitor
from .evaluate import load_records, predict_all
from .exploratory import random_names_view_fn

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
KEY_ENV = "TYPESAFE_API_KEY"
KEY_FILE = Path("~/.config/typesafe/api_key")
PRICE_PER_MTOK = 0.042  # USD per million input tokens; output tokens are free
COST_CAP_USD = 2.0
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}

MONITORS = {
    "jev_plain": "jev_plain",
    "jev_plus": "jev_plus_factors",
    "clef_plain": "clef_plain",
    "clef_plus": "clef_plus_factors",
    "llm_standard": "llm_monitor_pf_standard",
    "llm_plus": "llm_monitor_pf_plus_factors",
}
PAIRS = [
    ("jev_plain", "llm_standard"), ("jev_plus", "llm_plus"), ("jev_plus", "llm_standard"),
    ("jev_plain", "clef_plain"), ("jev_plus", "clef_plus"), ("jev_plus", "jev_plain"),
]

# (url, body, key) -> (HTTP status, response body, retry-after seconds or None)
Post = Callable[[str, dict[str, Any], str], tuple[int, dict[str, Any], float | None]]


class CostCapReached(RuntimeError):
    pass


def api_key(env: dict[str, str] | None = None, path: Path = KEY_FILE) -> str:
    env = os.environ if env is None else env
    key = (env.get(KEY_ENV) or "").strip()
    if not key:
        path = path.expanduser()
        key = path.read_text().strip() if path.exists() else ""
    if not key:
        raise SystemExit(f"no TypeSafe key: set {KEY_ENV} or write it to {path}")
    return key


def post_json(url: str, body: dict[str, Any], key: str, timeout: float = 120) -> tuple[int, dict[str, Any], float | None]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read()), None
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read() or b"{}")
        except json.JSONDecodeError:
            payload = {}
        after = e.headers.get("retry-after") if e.headers else None
        try:
            after = float(after) if after is not None else None
        except ValueError:
            after = None
        return e.code, payload, after


class JevScorer:
    """POSTs SystemOne request bodies to Jev; retries rate limits and overloads
    with backoff, pins the answering version, counts spend and stops at the cap."""

    def __init__(
        self, key: str, model: str = MODEL, post: Post | None = None, cap_usd: float = COST_CAP_USD,
        retries: int = 6, base_delay: float = 2.0, url: str = API_URL,
    ):
        self.key, self.model, self.url = key, model, url
        self.post = post or post_json
        self.cap_usd, self.retries, self.base_delay = cap_usd, retries, base_delay
        self.version: str | None = None
        self.input_tokens = 0

    @property
    def spent_usd(self) -> float:
        return self.input_tokens * PRICE_PER_MTOK / 1e6

    async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("model") != self.model:
            raise ValueError(f"request asks for {request.get('model')!r}, scorer is pinned to {self.model!r}")
        for attempt in range(self.retries + 1):
            if self.spent_usd >= self.cap_usd:
                raise CostCapReached(f"spent ${self.spent_usd:.3f}, cap ${self.cap_usd:.2f}")
            status, body, after = await asyncio.to_thread(self.post, self.url, request, self.key)
            if status == 200:
                usage = body.get("usage") or {}
                self.input_tokens += int(usage.get("input_tokens", 0))
                version = body.get("model")
                if self.version is None:
                    self.version = version
                elif version != self.version:
                    raise RuntimeError(f"answered by {version!r}, not the pinned {self.version!r}")
                return body
            if status in RETRY_STATUS and attempt < self.retries:
                await asyncio.sleep(after if after is not None else self.base_delay * 2**attempt * (1 + random.random() / 4))
                continue
            raise RuntimeError(f"HTTP {status}: {json.dumps(body)[:500]}")
        raise AssertionError("unreachable")

    async def probe(self) -> str:
        """One tiny request; returns (and pins) the version the alias points to."""
        req = {"model": self.model, "state": "ok", "questions": {"probe": {"type": "noul", "instructions": "Is the state the word ok?"}}}
        await self(req)
        assert self.version
        return self.version


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, action="append")
    ap.add_argument("--variant", action="append", choices=sorted(VARIANTS))
    ap.add_argument("--compare", type=Path, metavar="RUNS_DIR")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--games", help="comma-separated game ids (smoke test)")
    ap.add_argument("--out", type=Path, help="predictions file (one run and variant only); default predictions/jev_<variant>.jsonl")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--cap-usd", type=float, default=COST_CAP_USD)
    args = ap.parse_args(argv)
    if args.compare:
        out = compare(args.compare, args.n_boot, MONITORS, PAIRS)
        (args.compare / "main" / "metrics" / "exploratory_jev.json").write_text(json.dumps(out, indent=1))
        print_comparison(out)
        return 0
    if not (args.run and args.variant):
        ap.error("--run and --variant are required unless --compare")
    if args.out and (len(args.variant) > 1 or len(args.run) > 1):
        ap.error("--out takes one run and one variant")

    scorer = JevScorer(api_key(), cap_usd=args.cap_usd)
    version = asyncio.run(scorer.probe())
    print(f"E14 {MODEL} resolves to {version}", flush=True)
    for run in args.run:
        records = [r for r in load_records(run) if r["condition"] in (CLEAN, COALITION)]
        if args.games:
            keep = set(args.games.split(","))
            records = [r for r in records if r["game_id"] in keep]
        for variant in args.variant:
            det = DecisionModelMonitor(scorer, MODEL, version, variant)
            path = args.out or run / "predictions" / f"jev_{variant}.jsonl"
            done = asyncio.run(predict_all(det, records, path, args.concurrency, view_fn=random_names_view_fn()))
            print(f"E14 jev {variant} {run}: {len(done)} of {len(records)} games scored; "
                  f"spent ${scorer.spent_usd:.4f} ({scorer.input_tokens} input tokens)", flush=True)
            if scorer.spent_usd >= scorer.cap_usd:
                print("E14 stopped at the cost cap", flush=True)
                return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
