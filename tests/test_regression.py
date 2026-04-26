"""Re-run the BM25 retrieval eval and compare with the numbers published in the README.

Needs the downloaded data (CI downloads it); no API calls. A change to tokenizing, chunking or
the metrics that moves the published numbers makes this test fail.
"""

import pytest

from regrag.config import DOCS_DIR

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not DOCS_DIR.exists(), reason="run `uv run python -m regrag.download`"),
]

# 300 test questions, seed 42 (`uv run python -m regrag.eval_retrieval --method bm25`)
PUBLISHED = {
    "recall@1": 0.5073333333333333,
    "recall@3": 0.6492777777777777,
    "recall@5": 0.6748333333333333,
    "recall@10": 0.7331666666666666,
    "recall@20": 0.7762222222222223,
    "mrr@20": 0.6721164636017577,
}


def test_bm25_eval_matches_published_numbers():
    from regrag.eval_retrieval import run_bm25, sample_questions, score

    items = sample_questions("test", 300, 42)
    assert score(run_bm25(items), items) == pytest.approx(PUBLISHED, abs=1e-12)
