"""Optional integration test for the real khPOS corpus (skipped when data/raw is absent)."""

from pathlib import Path

import pytest

from kroch_chhmar.dictionary.compound_splitter import CompoundSplitter
from kroch_chhmar.dictionary.cost_estimator import CostEstimator
from kroch_chhmar.ingest.local_importers import HoldoutFilter, ingest_segmented_text_file

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw"


@pytest.mark.skipif(
    not (RAW / "khpos_train.pos").exists(),
    reason="khPOS raw data not downloaded (run `kroch-chhmar fetch khpos`)",
)
def test_khpos_close_test_is_removed_from_training() -> None:
    train = RAW / "khpos_train.pos"
    open_test = RAW / "khpos_open_test.pos"
    close_test = RAW / "khpos_close_test.pos"
    if not open_test.exists() or not close_test.exists():
        pytest.skip("khPOS test splits not downloaded")

    holdout = HoldoutFilter.from_files([open_test, close_test])
    # CLOSE-TEST is the first 1,000 lines of upstream train.all2.
    assert len(holdout) >= 1000

    estimator = CostEstimator()
    splitter = CompoundSplitter()
    ingest_segmented_text_file(train, estimator, splitter=splitter, holdout=holdout)

    assert holdout.skipped >= 1000
    # No held-out sentence signature may survive in the training corpus.
    leaked = holdout.keys & estimator.stats.seen_surface_keys
    assert not leaked, f"{len(leaked)} held-out sentences leaked into training"
