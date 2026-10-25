import json

import pytest

from seance.paper_figures import MODELS, main


def _row(v):
    m = {"value": v, "lo": v * 0.8, "hi": min(v * 1.2, 30)}
    return {"scored": True, "measures": {"clean_accused": m, "auroc": {"value": 0.7, "lo": 0.6, "hi": 0.8}}}


def test_figure_skips_unscored_models(tmp_path):
    pytest.importorskip("matplotlib")  # in the optional `report` group
    rows = {"qwen_plain": _row(25), "qwen_pf": _row(7), "clef_plain": {"scored": False}}
    table = tmp_path / "final_table.json"
    table.write_text(json.dumps({"n_clean": 30, "rows": rows}))
    out = tmp_path / "fig.png"
    assert main(["--table", str(table), "--label", "Qwen players", "--out", str(out)]) == 0
    assert out.stat().st_size > 0
    assert MODELS[0][1:] == ("qwen_plain", "qwen_pf")
