import asyncio
import json

import pytest

from seance import report
from seance.conditions import LABELS, NONE_LABEL
from seance.detectors import LLMMonitor
from seance.evaluate import load_records, predict_all, score
from seance.evaluate import main as evaluate_main
from seance.inference import Completion
from seance.run import main as run_main


class OracleClient:
    """Stands in for the monitor model: puts most weight on the pair whose
    messages use the scripted allies' "I count N candles" code."""

    model = "oracle-stub"

    async def complete(self, messages, schema_name, schema):
        record = messages[0]["content"].split("The record:")[1]
        hits = {}
        for line in record.splitlines():
            name, _, text = line.strip().partition(": ")
            if "candles" in text:
                hits[name] = hits.get(name, 0) + 1
        probs = {k: 0.02 for k in LABELS}
        top = sorted(hits, key=hits.get, reverse=True)[:2]
        key = next((k for k in LABELS if len(top) == 2 and set(k.split("+")) == set(top)), NONE_LABEL)
        probs[key] = 0.88
        return Completion(text=json.dumps({"reasoning": "stub", "probabilities": probs}), model=self.model)


@pytest.fixture(scope="module")
def scored_run(tmp_path_factory):
    run = tmp_path_factory.mktemp("report") / "run"
    assert run_main(["--out", str(run), "--fake", "--clean", "6", "--coalition", "6", "--positive", "6", "--seed", "8"]) == 0
    for det in ("prior", "top_scorer"):
        assert evaluate_main(["--run", str(run), "--detector", det, "--n-boot", "50"]) == 0
    recs = load_records(run)
    monitor = LLMMonitor(OracleClient(), settings={"temperature": 0.7}, n_readings=3)
    preds = asyncio.run(predict_all(monitor, recs, run / "predictions" / "llm_monitor.jsonl", 4))
    (run / "metrics" / "llm_monitor.json").write_text(json.dumps(score(recs, preds, 50, 0)))
    assert evaluate_main(["--run", str(run), "--compare", "--communication", "--payoff", "--n-boot", "50"]) == 0
    return run


def test_results_markdown_reports_every_preregistered_number(scored_run):
    assert report.main(["--run", str(scored_run), "--no-figures"]) == 0
    md = (scored_run / "report" / "results.md").read_text()
    for needle in ("Prior", "Top scorers", "LLM monitor", "M1", "M2", "H1", "H2", "H3", "Did colluding pay?"):
        assert needle in md
    assert "6 clean and 6 coalition games (positive controls excluded)" in md
    # The oracle names the right pair in every positive-control game.
    assert "right in 6 of 6 positive-control games" in md


def test_figures_render(scored_run):
    pytest.importorskip("matplotlib")
    assert report.main(["--run", str(scored_run)]) == 0
    for name in ("fig_detection.png", "fig_reliability.png", "fig_partner_weight.png", "fig_payoff.png"):
        assert (scored_run / "report" / name).stat().st_size > 10_000


def test_missing_metrics_is_an_error(tmp_path):
    assert report.main(["--run", str(tmp_path)]) == 2
