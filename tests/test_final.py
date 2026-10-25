"""The final test set's new pieces: spec copying, the spend ledger and its
stops (players and batch monitor), resuming, and the analysis script.
No network: every API client is a fake."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from seance import budget as B
from seance import claude_monitor as CM
from seance import final as F
from seance import run as R
from seance.conditions import CLEAN, COALITION, LABELS, MAIN_SEED_LIMIT, NONE_LABEL, POSITIVE_CONTROL, build_specs
from seance.evaluate import _read_predictions, load_records
from seance.exploratory import random_names_view_fn
from seance.fake import ScriptedModel
from seance.inference import Completion

ROOT = Path(__file__).resolve().parents[1]
QUIET = lambda *_: None  # noqa: E731
USAGE = {"input_tokens": 1000, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": 100}
PLAYER_USAGE = {"input_tokens": 3000, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": 100}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    import os

    for k in list(os.environ):
        if k.startswith("SEANCE_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


# --- the seed --------------------------------------------------------------------------


def test_final_seed_is_new():
    # every run but the final sets themselves (which this seed builds)
    manifests = [json.loads(p.read_text()) for p in sorted((ROOT / "runs").glob("*/manifest.json"))
                 if p.parent.name not in ("final", "final-haiku")]
    assert manifests
    used_build = {m["build_args"].get("seed") for m in manifests}
    from seance import conditions as C, exploratory as X

    defaults = {C.MAIN_ID_SHUFFLE_SEED, C.MAIN_PAIR_SEED, X.ROTATION_SEED, X.NAME_SEED, X.PF_NAME_SEED, X.REDACT_NAME_SEED}
    assert F.FINAL_SEED not in used_build | defaults
    used_deals = {g["seed"] for m in manifests for g in m["games"]}
    specs = build_specs(50, 50, 6, F.FINAL_SEED)
    deals = {s.seed for s in specs}
    assert len(deals) == 106 and not deals & used_deals and min(deals) >= MAIN_SEED_LIMIT
    by = lambda c: [s for s in specs if s.condition == c]  # noqa: E731
    assert len(by(CLEAN)) == 50 and len(by(COALITION)) == 50 and len(by(POSITIVE_CONTROL)) == 6
    assert len({s.allies for s in by(POSITIVE_CONTROL)}) == 6  # each pair once


# --- --specs-from ----------------------------------------------------------------------


def test_specs_from_copies_ids_seeds_pairs_and_roles(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert R.main(["--out", str(a), "--fake", "--clean", "2", "--coalition", "2", "--positive", "1", "--seed", "4"]) == 0
    assert R.main(["--out", str(b), "--fake", "--specs-from", str(a)]) == 0
    ma, mb = (json.loads((d / R.MANIFEST).read_text()) for d in (a, b))
    assert ma["games"] == mb["games"]
    assert mb["build_args"]["specs_from"] == str(a) and mb["build_args"]["specs_from_build_args"] == ma["build_args"]
    for g in ma["games"]:
        ra, rb = (json.loads((d / "games" / f"{g['game_id']}.json").read_text()) for d in (a, b))
        assert ra["spec"] == rb["spec"] and ra["roles"] == rb["roles"] and ra["allies"] == rb["allies"]
        assert [r["secret"] for r in ra["rounds"]] == [r["secret"] for r in rb["rounds"]]
        assert [r["visions"] for r in ra["rounds"]] == [r["visions"] for r in rb["rounds"]]
    # a plain run's manifest has no specs_from keys
    assert "specs_from" not in ma["build_args"]


def test_specs_from_games_subset_and_conflicts(tmp_path, capsys):
    a = tmp_path / "a"
    R.main(["--out", str(a), "--fake", "--clean", "2", "--coalition", "1", "--seed", "4"])
    gid = json.loads((a / R.MANIFEST).read_text())["games"][0]["game_id"]
    assert R.main(["--out", str(tmp_path / "s"), "--fake", "--specs-from", str(a), "--games", gid]) == 0
    assert [g["game_id"] for g in json.loads((tmp_path / "s" / R.MANIFEST).read_text())["games"]] == [gid]
    assert R.main(["--out", str(tmp_path / "x"), "--fake", "--specs-from", str(a), "--clean", "3"]) == 2
    assert R.main(["--out", str(tmp_path / "y"), "--fake", "--specs-from", str(a), "--games", "zzz"]) == 2
    assert R.main(["--out", str(tmp_path / "z"), "--fake", "--clean", "1", "--games", gid]) == 2


# --- the ledger ------------------------------------------------------------------------


def _plan():
    return (
        B.Step("1-big", "monitor", "runs/final", ("claude-opus-5-5",), 10),
        B.Step("2-small", "monitor", "runs/final", ("claude-haiku-5-5",), 10),
        B.Step("3-play", "players", "runs/final-haiku", ("claude-haiku-5-5",), 4),
    )


def _ledger(tmp_path, cap=1.0, plan=None):
    return B.Ledger(tmp_path / "runs" / "api_spend.jsonl", cap=cap, plan=_plan() if plan is None else plan, root=tmp_path,
                    remaining_ttl_s=0)


def test_step_estimates_count_only_what_is_left(tmp_path):
    plan = _plan()
    big = 10 * 2 * 3 * B.READING_COST["claude-opus-5-5"]
    assert B.step_remaining(plan[0], tmp_path) == pytest.approx(big)
    pred = tmp_path / "runs/final/predictions"
    pred.mkdir(parents=True)
    (pred / "claude_opus_plain.jsonl").write_text("".join(json.dumps({"game_id": f"g{i}"}) + "\n" for i in range(4)) + '{"trunc')
    assert B.step_remaining(plan[0], tmp_path) == pytest.approx(big * 16 / 20)
    games = tmp_path / "runs/final-haiku/games"
    games.mkdir(parents=True)
    (games / "p000.json").write_text("{}")
    assert B.step_remaining(plan[2], tmp_path) == pytest.approx(3 * B.PLAYER_GAME_COST)


def test_check_holds_every_higher_priority_step(tmp_path):
    big = B.step_remaining(_plan()[0], tmp_path)  # about $1
    small = B.step_remaining(_plan()[1], tmp_path)
    play = B.step_remaining(_plan()[2], tmp_path)
    lg = _ledger(tmp_path, cap=big + small + play + 0.05)
    lg.spend("0-before-ledger", 0.04)
    assert lg.check("1-big", big)[0]  # nothing ranks above it
    ok, msg = lg.check("3-play", play + 0.02)  # 0.04 + big + small + play + 0.02 > cap
    assert not ok and "held for 1-big" in msg and "held for 2-small" in msg and "pass the" in msg
    assert lg.check("2-small", small)[0]
    # a smoke step ranks below every plan step
    assert not lg.check("smoke-x", 0.02)[0] and lg.check("smoke-x", 0.001)[0]


def test_open_batches_count_once_and_settle_at_cost(tmp_path):
    lg = _ledger(tmp_path, cap=5)
    big = B.step_remaining(_plan()[0], tmp_path)
    with lg.locked():
        lg.reserve("1-big", "b1", big, ["c1", "c2"])
    assert lg.spent() == pytest.approx(big)
    # the reservation is not also held as the step's remaining estimate
    ok, msg = lg.check("3-play", 0.01)
    assert ok and "held for 1-big" not in msg
    assert [e["batch_id"] for e in lg.open_batches("1-big")] == ["b1"] and lg.open_batches("2-small") == []
    lg.settle("b1", 0.3, 2)
    assert lg.spent() == pytest.approx(0.3) and lg.open_batches("1-big") == []
    with pytest.raises(ValueError):
        lg.settle("b1", 0.3, 2)
    # another process reading the same file sees the same totals
    assert _ledger(tmp_path, cap=5).spent() == pytest.approx(0.3)


def test_prior_spend_is_the_pilot_and_smoke_test():
    entries = B.prior_spend(ROOT)
    total = sum(e["usd"] for e in entries)
    assert 0.64 < total < 0.67  # "about $0.66" in the sprint notes
    assert all(e["type"] == "spend" and e["step"] == "0-before-ledger" for e in entries)


def test_cost_constants_match_the_measured_runs():
    from seance.claude import cost

    per = {}
    for p in (ROOT / "runs/smoke-claude/predictions").glob("claude_*.jsonl"):
        if "noschema" in p.name:
            continue
        for line in p.read_text().splitlines():
            row = json.loads(line)
            m = row["detector"]["model"]
            c, n = per.get(m, (0.0, 0))
            per[m] = (c + cost(m, row["detector"]["usage"], batch=True), n + len(row["detail"]["readings"]))
    for m, (c, n) in per.items():
        assert B.READING_COST[m] == pytest.approx(c / n, rel=0.02)
    games = [json.loads(p.read_text()) for p in (ROOT / "runs/pilot-haiku/games").glob("*.json")]
    assert B.PLAYER_GAME_COST == pytest.approx(sum(cost(g["model"], g["tokens"]["usage"]) for g in games) / len(games), rel=0.02)


def test_the_plan_fits_the_cap_from_today():
    """With $0.66 spent, every step can be held at once, so the Haiku games
    may play while the Qwen games do."""
    total = 0.66 + sum(B.step_remaining(s, Path("/nonexistent")) for s in B.PLAN)
    assert total < B.CAP_USD
    assert [s.name for s in B.PLAN] == ["1-opus-sonnet-final", "2-haiku-monitor-final", "3-haiku-players", "4-haiku-monitor-final-haiku"]


# --- the Haiku players under the ledger ------------------------------------------------


class Priced(ScriptedModel):
    """The scripted model, with an Anthropic model name and usage per call."""

    def __init__(self, seed=0, **kw):
        super().__init__(seed=seed, **kw)
        self.model = "claude-haiku-5-5"
        self.calls = 0

    async def complete(self, messages, schema_name, schema):
        self.calls += 1
        c = await super().complete(messages, schema_name, schema)
        return Completion(text=c.text, model=self.model, usage=dict(PLAYER_USAGE))


class OneReply:
    model = "claude-haiku-5-5"

    async def complete(self, messages, schema_name, schema):
        return Completion(text="{}", model=self.model, usage=dict(PLAYER_USAGE))


async def test_ledger_client_records_each_call_and_stops_at_the_cap(tmp_path):
    lg = _ledger(tmp_path, cap=10)
    client = B.LedgerClient(OneReply(), lg, "smoke-x", "runs/x")
    await client.complete([{"role": "user", "content": "hi"}], "seance_speak", {})
    from seance.claude import cost

    assert lg.spent() == pytest.approx(cost("claude-haiku-5-5", PLAYER_USAGE))
    entry = json.loads(lg.path.read_text().splitlines()[-1])
    assert entry["what"] == "player_call" and entry["usage"] == PLAYER_USAGE and entry["step"] == "smoke-x"
    tight = _ledger(tmp_path, cap=lg.spent() + 0.0001, plan=())
    with pytest.raises(B.BudgetExceeded):
        await B.LedgerClient(OneReply(), tight, "smoke-x", "runs/x").complete([{"role": "user", "content": "hi"}], "s", {})


def _anthropic_main(monkeypatch, model_cls=Priced):
    import seance.claude as C

    monkeypatch.setenv("SEANCE_PROVIDER", "anthropic")
    monkeypatch.setenv("SEANCE_MODEL", "claude-haiku-5-5")
    monkeypatch.setattr(C, "AnthropicClient", lambda settings: model_cls())


def test_anthropic_players_need_a_valid_step(tmp_path, monkeypatch):
    _anthropic_main(monkeypatch)
    base = ["--out", str(tmp_path / "h"), "--clean", "1", "--env-file", str(tmp_path / "none.env"), "--ledger", str(tmp_path / "l.jsonl")]
    assert R.main(base) == 2  # no --spend-step
    assert R.main(base + ["--spend-step", "my-step"]) == 2  # neither plan nor smoke-*
    assert R.main(base + ["--spend-step", "1-opus-sonnet-final"]) == 2  # a monitor step
    assert R.main(base + ["--spend-step", "3-haiku-players"]) == 2  # plays runs/final-haiku, not this --out


def test_budget_stop_plays_no_more_games_and_the_rerun_plays_only_the_rest(tmp_path, monkeypatch):
    _anthropic_main(monkeypatch)
    from seance.claude import cost

    ledger = tmp_path / "l.jsonl"
    out = tmp_path / "h"
    lg = B.Ledger(ledger, plan=(), root=tmp_path)
    per_game = B.PLAYER_GAME_COST
    # Enough for the start check of 3 games at the estimate, but the fake's calls cost more,
    # so the cap is reached part way.
    monkeypatch.setattr(B, "CAP_USD", 3 * per_game + 0.0001)
    monkeypatch.setattr(B, "PLAN", ())
    monkeypatch.setattr(B, "PLAYER_GAME_COST", per_game)
    args = ["--out", str(out), "--clean", "2", "--coalition", "1", "--seed", "5", "--env-file", str(tmp_path / "none.env"),
            "--ledger", str(ledger), "--spend-step", "smoke-players", "--games-in-flight", "1"]
    rc = R.main(args)
    played = sorted(p.name for p in (out / "games").glob("*.json"))
    assert rc == 3 and len(played) < 3
    assert not list((out / "failed").glob("*.json"))  # an interrupted game is not a failed game
    spent = lg.spent()
    assert spent <= B.CAP_USD + 8 * cost("claude-haiku-5-5", PLAYER_USAGE)  # overshoot at most the calls in flight
    # more credit: the rerun plays only the missing games
    monkeypatch.setattr(B, "CAP_USD", 10.0)
    assert R.main(args) == 0
    assert len(list((out / "games").glob("*.json"))) == 3
    assert lg.spent() > spent


def test_start_check_refuses_a_step_that_cannot_fit(tmp_path, monkeypatch, capsys):
    _anthropic_main(monkeypatch)
    monkeypatch.setattr(B, "CAP_USD", 0.01)
    monkeypatch.setattr(B, "PLAN", ())
    rc = R.main(["--out", str(tmp_path / "h"), "--clean", "2", "--env-file", str(tmp_path / "none.env"),
                 "--ledger", str(tmp_path / "l.jsonl"), "--spend-step", "smoke-p"])
    assert rc == 3 and "STOPPED (budget): 2 games not played" in capsys.readouterr().out
    assert not (tmp_path / "h" / "games").exists() or not list((tmp_path / "h" / "games").glob("*.json"))


# --- the batch monitor under the ledger ------------------------------------------------


def _answer(keys, top):
    return json.dumps({"reasoning": "r", "probabilities": {k: (0.7 if k == top else 0.05) for k in keys}})


def _ok(text):
    usage = SimpleNamespace(input_tokens=1000, cache_read_input_tokens=0, cache_creation_input_tokens=0, output_tokens=100)
    return SimpleNamespace(type="succeeded", message=SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], usage=usage, stop_reason="end_turn", stop_details=None))


class Batches:
    """Fake Message Batches: every request answered; `bad` custom ids get an
    unparsable first answer; `hang` leaves a batch in progress forever."""

    def __init__(self, bad=(), hang=False):
        self.bad, self.hang, self.store, self.created = set(bad), hang, {}, []

    def create(self, requests):
        bid = f"b{len(self.store) + 1}"
        self.store[bid] = list(requests)
        self.created.append(bid)
        return SimpleNamespace(id=bid, processing_status="in_progress")

    def retrieve(self, bid):
        if self.hang:
            raise KeyboardInterrupt("stopped while waiting")
        counts = SimpleNamespace(processing=0, succeeded=len(self.store[bid]), errored=0)
        return SimpleNamespace(id=bid, processing_status="ended", request_counts=counts)

    def results(self, bid):
        for req in self.store[bid]:
            cid, params = req["custom_id"], req["params"]
            keys = list(params["output_config"]["format"]["schema"]["properties"]["probabilities"]["properties"])
            text = "oops" if cid in self.bad and len(params["messages"]) == 1 else _answer(keys, "none")
            yield SimpleNamespace(custom_id=cid, result=_ok(text))


async def _games(tmp_path, n_clean=2, n_coalition=2):
    run = tmp_path / "runs" / "final"
    specs = R.load_or_create_manifest(run, build_specs(n_clean, n_coalition, 1, seed=9), {})
    await R.run_games(specs, ScriptedModel(seed=9), run, log=QUIET)
    return run


async def test_batches_reserve_settle_and_write_each_finished_game(tmp_path):
    run = await _games(tmp_path)
    recs = [r for r in load_records(run) if r["condition"] in (CLEAN, COALITION)]
    readings = CM.make_readings("final", recs, [CM.Config("claude-haiku-5-5", "plain")], 3, random_names_view_fn())
    bad = {readings[0].custom_id}  # game 0 needs a second batch
    fake = Batches(bad)
    lg = _ledger(tmp_path, cap=1, plan=())
    seat = {r["game_id"]: random_names_view_fn()(r)["names"] for r in recs}
    (run / "predictions").mkdir()
    written, per_batch = set(), []

    def flush():
        per_batch.append(CM.write_finished(readings, 3, run / "predictions", seat, written))
        assert lg.open_batches("smoke-t")  # rows are written before the batch is settled

    client = SimpleNamespace(messages=SimpleNamespace(batches=fake))
    CM.run_batches(client, readings, 3, poll_s=0, log=QUIET, ledger=lg, step="smoke-t", after_batch=flush)
    assert per_batch == [len(recs) - 1, 1]  # the other games were written after the first batch
    entries = [json.loads(x) for x in lg.path.read_text().splitlines()]
    assert [e["type"] for e in entries] == ["reserve", "settle", "reserve", "settle"]
    assert entries[0]["usd"] == pytest.approx(len(readings) * B.READING_COST["claude-haiku-5-5"])
    from seance.claude import cost

    assert entries[1]["usd"] == pytest.approx(len(readings) * cost("claude-haiku-5-5", USAGE, batch=True))
    assert len(_read_predictions(run / "predictions" / "claude_haiku_plain.jsonl")) == len(recs)


async def test_batch_that_would_pass_the_cap_is_never_sent(tmp_path):
    run = await _games(tmp_path)
    recs = [r for r in load_records(run) if r["condition"] in (CLEAN, COALITION)]
    readings = CM.make_readings("final", recs, [CM.Config("claude-opus-5-5", "plain")], 3, random_names_view_fn())
    fake = Batches()
    lg = _ledger(tmp_path, cap=0.01, plan=())
    with pytest.raises(B.BudgetExceeded, match="smoke-t stopped before spending"):
        CM.run_batches(SimpleNamespace(messages=SimpleNamespace(batches=fake)), readings, 3, poll_s=0, log=QUIET, ledger=lg, step="smoke-t")
    assert fake.created == [] and not lg.path.exists()


async def test_rerun_collects_an_open_batch_instead_of_paying_again(tmp_path):
    run = await _games(tmp_path)
    recs = [r for r in load_records(run) if r["condition"] in (CLEAN, COALITION)]
    config = CM.Config("claude-haiku-5-5", "plain")
    lg = _ledger(tmp_path, cap=1, plan=())
    fake = Batches(hang=True)
    first = CM.make_readings("final", recs, [config], 3, random_names_view_fn())
    with pytest.raises(KeyboardInterrupt):  # the process stops while the batch runs
        CM.run_batches(SimpleNamespace(messages=SimpleNamespace(batches=fake)), first, 3, poll_s=0, log=QUIET, ledger=lg, step="smoke-t")
    assert len(lg.open_batches("smoke-t")) == 1
    fake.hang = False
    again = CM.make_readings("final", recs, [config], 3, random_names_view_fn())
    client = SimpleNamespace(messages=SimpleNamespace(batches=fake))
    ids = CM.recover_open_batches(client, lg, "smoke-t", again, poll_s=0, log=QUIET)
    assert ids == ["b1"] and all(r.result is not None for r in again) and lg.open_batches("smoke-t") == []
    CM.run_batches(client, again, 3, poll_s=0, log=QUIET, ledger=lg, step="smoke-t")
    assert fake.created == ["b1"]  # nothing sent twice


async def test_monitor_main_reads_only_missing_games_and_checks_its_step(tmp_path, monkeypatch, capsys):
    import anthropic

    run = await _games(tmp_path)
    fake = Batches()
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: SimpleNamespace(messages=SimpleNamespace(batches=fake)))
    monkeypatch.chdir(tmp_path)
    recs = [r for r in load_records(run) if r["condition"] in (CLEAN, COALITION)]
    base = ["--run", str(run), "--model", "claude-haiku-5-5", "--prompt", "plain", "--poll-s", "0", "--ledger", str(tmp_path / "l.jsonl")]
    with pytest.raises(SystemExit):
        CM.main(base + ["--step", "2-haiku-monitor-final"])  # the plan step needs both prompts
    with pytest.raises(SystemExit):
        CM.main(base + ["--step", "whatever"])
    first = ",".join(r["game_id"] for r in recs[:2])
    out = tmp_path / "scratch"
    assert CM.main(base + ["--step", "smoke-a", "--games", first, "--out", str(out)]) == 0
    assert len(fake.store["b1"]) == 2 * 3
    assert CM.main(base + ["--step", "smoke-a", "--out", str(out)]) == 0
    assert len(fake.store["b2"]) == (len(recs) - 2) * 3  # only the missing games
    assert len(_read_predictions(out / "claude_haiku_plain.jsonl")) == len(recs)
    assert CM.main(base + ["--step", "smoke-a", "--out", str(out)]) == 0 and "nothing to do" in capsys.readouterr().out
    entries = [json.loads(x) for x in (tmp_path / "l.jsonl").read_text().splitlines()]
    assert [e["type"] for e in entries] == ["reserve", "settle", "reserve", "settle"]


def test_plan_step_must_match_the_invocation(tmp_path):
    lg = B.Ledger(tmp_path / "l.jsonl", root=tmp_path)
    both = ["plain", "plus_factors"]
    assert CM.check_step(lg, "1-opus-sonnet-final", tmp_path / "runs/final", ["claude-sonnet-5-5", "claude-opus-5-5"], both, False) is None
    assert "reads runs/final" in CM.check_step(lg, "1-opus-sonnet-final", tmp_path / "runs/fresh", ["claude-sonnet-5-5", "claude-opus-5-5"], both, False)
    assert CM.check_step(lg, "1-opus-sonnet-final", tmp_path / "runs/final", ["claude-opus-5-5"], both, False)
    assert CM.check_step(lg, "2-haiku-monitor-final", tmp_path / "runs/final", ["claude-haiku-5-5"], both, True)  # --games
    assert CM.check_step(lg, "3-haiku-players", tmp_path / "runs/final-haiku", ["claude-haiku-5-5"], both, False)
    assert CM.check_step(lg, "4-haiku-monitor-final-haiku", tmp_path / "runs/final-haiku", ["claude-haiku-5-5"], both, False) is None
    assert CM.check_step(lg, "smoke-anything", tmp_path / "runs/fresh", ["claude-opus-5-5"], ["plain"], True) is None


# --- the analysis script ---------------------------------------------------------------


def _write_preds(run, stem, records, top_for):
    path = run / "predictions" / f"{stem}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            top = top_for(r)
            probs = {k: (0.64 if k == top else 0.06) for k in LABELS}
            f.write(json.dumps({"game_id": r["game_id"], "probs": probs, "score": 1 - probs[NONE_LABEL], "detector": {"name": stem}}) + "\n")


async def test_analysis_end_to_end_on_a_small_set(tmp_path):
    run = tmp_path / "final"
    specs = R.load_or_create_manifest(run, build_specs(6, 6, 2, seed=11), {})
    await R.run_games(specs, ScriptedModel(seed=11), run, log=QUIET)
    recs = load_records(run)
    cc = [r for r in recs if r["condition"] in (CLEAN, COALITION)]
    perfect = lambda r: r["label"]  # noqa: E731
    _write_preds(run, "llm_monitor_pf_standard", cc, lambda r: "Daniel+Eileen")  # accuses every table
    _write_preds(run, "llm_monitor_pf_plus_factors", recs, perfect)  # includes the positive controls
    _write_preds(run, "claude_opus_plain", cc[1:], perfect)  # one game failed
    res = F.analyse([run], run / "metrics", n_boot=200, make_figure=False, log=QUIET)
    rows = res["rows"]
    assert rows["qwen_plain"]["measures"]["clean_accused"]["value"] == 6
    assert rows["qwen_pf"]["measures"]["clean_accused"]["value"] == 0
    assert rows["qwen_pf"]["measures"]["auroc"]["value"] == 1.0
    assert rows["qwen_pf"]["measures"]["right_pair"]["value"] == 1.0
    assert rows["qwen_pf"]["n_clean"] == 6 and rows["qwen_pf"]["n_coalition"] == 6  # positive controls left out
    assert sum(rows["opus_plain"]["n_detector_failed"].values()) == 1 and rows["opus_plain"]["vs"]["qwen_plain"]["n_games"] == 11
    assert not rows["sonnet_plain"]["scored"] and not rows["jev_pf"]["scored"]
    assert rows["prior"]["scored"] and rows["top_scorer"]["scored"] and rows["qwen_plain_recal"]["scored"]
    mc = res["main_comparison"]
    assert mc["clean_accused"]["difference"] == -6 and mc["clean_accused"]["excludes_zero"]
    pc = res["game_checks"]["positive_controls"]["qwen_pf"]
    assert pc == {"n_games": 2, "n_read": 2, "n_top_pair_right": 2, "n_p_none_below_half": 2}
    assert res["game_checks"]["positive_controls"]["qwen_plain"]["n_read"] == 0
    assert "coalition" in res["game_checks"]["manipulation_check"]
    assert res["game_checks"]["did_colluding_pay"]["qwen_pf"]["coalition"]["n_caught"] == 6
    md = (run / "metrics" / "final_table.md").read_text()
    assert "Main comparison" in md and "Sonnet 5.5, plain | not scored" in md
    json.loads((run / "metrics" / "final_table.json").read_text())
    # nothing written into the run's predictions unless asked
    assert not (run / "predictions" / "prior.jsonl").exists()
    F.analyse([run], run / "metrics", n_boot=50, make_figure=False, write=True, log=QUIET)
    for stem in F.DERIVED:
        assert (run / "predictions" / f"{stem}.jsonl").exists()
    before = (run / "predictions" / "prior.jsonl").read_text()
    F.write_derived(run, QUIET)
    assert (run / "predictions" / "prior.jsonl").read_text() == before


async def test_outside_monitor_is_scored_as_an_extra_row(tmp_path):
    run = tmp_path / "final"
    specs = R.load_or_create_manifest(run, build_specs(4, 4, 0, seed=12), {})
    await R.run_games(specs, ScriptedModel(seed=12), run, log=QUIET)
    recs = load_records(run)
    _write_preds(run, "llm_monitor_pf_standard", recs, lambda r: "Daniel+Eileen")
    mine = tmp_path / "mine"
    _write_preds(mine, "my_monitor", recs, lambda r: r["label"])
    path = mine / "predictions" / "my_monitor.jsonl"
    out = tmp_path / "scores"
    assert F.main(["--run", str(run), "--monitor", f"Mine={path}", "--out", str(out), "--no-figure", "--n-boot", "50"]) == 0
    res = json.loads((out / "final_table.json").read_text())
    row = next(r for r in res["rows"].values() if r["label"] == "Mine")
    assert row["measures"]["clean_accused"]["value"] == 0 and row["measures"]["auroc"]["value"] == 1.0
    assert row["vs"]["qwen_plain"]["clean_accused"]["difference"] == -4
    assert "Mine" in (out / "final_table.md").read_text()
    assert not (run / "metrics").exists()  # the released tables are left alone
    with pytest.raises(SystemExit):
        F.main(["--run", str(run), "--monitor", f"Mine={path}"])  # --out required
    with pytest.raises(SystemExit):
        F.main(["--run", str(run), "--monitor", "no-equals-sign", "--out", str(out)])


def test_recalibration_was_fitted_on_development_data_only():
    import numpy as np

    from seance import exploratory as X, metrics

    labels, preds = [], []
    for name in F.RECALIBRATION["fitted_on"]:
        run = ROOT / name
        p = _read_predictions(run / "predictions" / f"{F.RECALIBRATION['of']}.jsonl")
        for r in load_records(run):
            if r["condition"] in (CLEAN, COALITION) and r["game_id"] in p:
                labels.append(r["label"])
                preds.append(p[r["game_id"]]["probs"])
    assert len(labels) == F.RECALIBRATION["n_games"]
    y, p, _ = metrics._arrays(labels, preds)
    t, b = X._fit(y, p, X.T_GRID, X.B_GRID)
    assert (t, b) == pytest.approx((F.RECALIBRATION["temperature"], F.RECALIBRATION["none_bias"]))
    q = F.recalibrated({k: 1 / 7 for k in LABELS})
    assert abs(sum(q.values()) - 1) < 1e-9 and q[NONE_LABEL] > 1 / 7
    assert not any("final" in n for n in F.RECALIBRATION["fitted_on"])
    assert np.isfinite(t)
