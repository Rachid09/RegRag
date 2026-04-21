"""Retrieval eval: Recall@k and MRR@20 on a fixed random sample of test questions.

    uv run python -m regrag.eval_retrieval --method bm25    # 300 test questions, seed 42
    uv run python -m regrag.eval_retrieval --method dense   # needs the full Qdrant index

Dense: questions are embedded in batches through the cached Embedder, so a rerun makes no API
calls. BM25 runs locally and makes none at all.
Writes reports/retrieval_<method>.md (table) and .json (per-question ranks, for later analysis).
"""

import argparse
import json
import random
from pathlib import Path

from regrag import config
from regrag.corpus import QAItem, load_split
from regrag.embed import Embedder, query_text
from regrag.metrics import mean, recall_at_k, reciprocal_rank
from regrag.retrieve import N_PASSAGES, dense_search
from regrag.store import connect

KS = (1, 3, 5, 10, 20)
REPORTS = Path("reports")


def sample_questions(split: str, n: int, seed: int) -> list[QAItem]:
    items = load_split(split)
    return random.Random(seed).sample(items, n)


def score(rankings: list[list], items: list[QAItem]) -> dict:
    out = {
        f"recall@{k}": mean(
            [recall_at_k(r, q.gold, k) for r, q in zip(rankings, items, strict=True)]
        )
        for k in KS
    }
    out[f"mrr@{N_PASSAGES}"] = mean(
        [reciprocal_rank(r, q.gold) for r, q in zip(rankings, items, strict=True)]
    )
    return out


def run_dense(items: list[QAItem]) -> list[list]:
    emb = Embedder()
    vectors = emb.embed([query_text(q.question) for q in items], progress=True)
    client = connect()
    return [[h.key for h in dense_search(client, v)] for v in vectors]


def run_bm25(items: list[QAItem]) -> list[list]:
    from regrag.bm25 import build_bm25

    index = build_bm25()
    return [[h.key for h in index.search(q.question)] for q in items]


METHODS = {"dense": run_dense, "bm25": run_bm25}
SETUP = {
    "dense": f"embedding model `{config.EMBED_MODEL}` ({config.EMBED_DIMS} dims), "
    f"collection `{config.COLLECTION}`",
    "bm25": "rank-bm25 BM25Okapi (k1=1.5, b=0.75), index text = title + headings + passage",
}


def write_report(method: str, split: str, n: int, seed: int, metrics: dict, rankings, items):
    REPORTS.mkdir(exist_ok=True)
    cmd = (
        f"uv run python -m regrag.eval_retrieval --method {method} "
        f"--split {split} --n {n} --seed {seed}"
    )
    lines = [
        f"# Retrieval eval: {method}",
        "",
        f"Split `{split}`, {n} questions sampled with seed {seed}; {SETUP[method]}.",
        "Each question keeps its top 20 passages (50 chunks fetched, windows merged).",
        f"Reproduce: `{cmd}`",
        "",
        "| metric | value |",
        "|---|---|",
        *[f"| {k} | {v:.4f} |" for k, v in metrics.items()],
        "",
    ]
    (REPORTS / f"retrieval_{method}.md").write_text("\n".join(lines))
    per_q = [
        {
            "qid": q.qid,
            "gold": [list(g) for g in q.gold],
            "ranked": [list(k) for k in r],
            "rr": reciprocal_rank(r, q.gold),
        }
        for r, q in zip(rankings, items, strict=True)
    ]
    (REPORTS / f"retrieval_{method}.json").write_text(
        json.dumps(
            {"split": split, "n": n, "seed": seed, "metrics": metrics, "questions": per_q}, indent=1
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=sorted(METHODS), default="dense")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    items = sample_questions(args.split, args.n, args.seed)
    rankings = METHODS[args.method](items)
    metrics = score(rankings, items)
    write_report(args.method, args.split, args.n, args.seed, metrics, rankings, items)
    for k, v in metrics.items():
        print(f"{k:>10}  {v:.4f}")


if __name__ == "__main__":
    main()
