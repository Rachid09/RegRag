import json

from regrag import eval_retrieval as ev


def test_ablation_lists_only_methods_run_on_the_same_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "REPORTS", tmp_path)
    metrics = {"recall@1": 0.5, "mrr@20": 0.6}
    for method, seed in (("bm25", 42), ("hybrid", 42), ("dense", 7)):
        (tmp_path / f"retrieval_{method}.json").write_text(
            json.dumps({"split": "test", "n": 300, "seed": seed, "metrics": metrics})
        )
    table = ev.write_ablation("test", 300, 42).read_text()
    assert "| bm25 | 0.5000 | 0.6000 |" in table
    assert "| hybrid |" in table
    assert "| dense |" not in table  # different seed: not comparable
