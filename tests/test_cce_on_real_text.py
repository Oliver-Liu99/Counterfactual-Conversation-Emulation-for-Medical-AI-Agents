"""End-to-end smoke test for the U3 verification script on real ACI-Bench text.

Skips when:
- ACI-Bench data is not present at data/raw/aci_bench/, OR
- torch is not installed, OR
- sentence-transformers is not installed.

When all three are available, runs a tiny version (n=10 encounters, 2
epochs) of the full verify_cce_on_real_text pipeline end-to-end and asserts
that all four diagnostic settings produce finite AUC / ESS / MMD numbers.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ACI_TRAIN_CSV = REPO_ROOT / "data" / "raw" / "aci_bench" / "data" / "challenge_data" / "train.csv"
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _has(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not ACI_TRAIN_CSV.exists(), reason="ACI-Bench data not present"),
    pytest.mark.skipif(not _has("torch"), reason="torch not installed"),
    pytest.mark.skipif(
        not _has("sentence_transformers"), reason="sentence-transformers not installed"
    ),
]


def _import_verify_module():
    """Import scripts/verify_cce_on_real_text.py as a module."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        return importlib.import_module("verify_cce_on_real_text")
    finally:
        # Leave path entry; harmless and avoids re-import surprises.
        pass


def test_verify_cce_on_real_text_tiny_end_to_end(tmp_path: Path) -> None:
    """Tiny smoke run: 10 encounters, 2 epochs, all 4 settings must produce
    finite numbers and write a JSON report.
    """
    import math

    mod = _import_verify_module()

    report = mod.run(
        output_dir=tmp_path,
        n=10,
        epochs=2,
        projection_dim=32,
        pca_dim=8,
        tau=0.07,
        lambda_hsic=0.0,
        lr=1e-3,
        seed=0,
    )

    # Basic shape checks
    assert "settings" in report
    settings = report["settings"]
    assert len(settings) == 4
    names = {s["setting"] for s in settings}
    assert "raw" in names
    assert "cce" in names
    assert any(n.startswith("raw+pca") for n in names)
    assert any(n.startswith("cce+pca") for n in names)

    for s in settings:
        # All numeric outputs must be finite real numbers.
        for k in ("classifier_auc", "ess", "ess_per_n", "mmd"):
            v = s[k]
            assert isinstance(v, float)
            assert not math.isnan(v), f"{s['setting']}.{k} is NaN"
            assert math.isfinite(v) or v == 0.0, f"{s['setting']}.{k} not finite"
        assert 0.0 <= s["classifier_auc"] <= 1.0
        assert 0.0 <= s["ess_per_n"] <= 1.0 + 1e-6
        assert s["n_train"] == 10

    # JSON file persisted
    out_json = tmp_path / "cce_real_diagnostics.json"
    assert out_json.exists()

    # Report fields the paper consumes
    assert "headline" in report
    headline = report["headline"]
    for k in ("auc_raw", "auc_cce", "delta_auc_raw_minus_cce"):
        assert k in headline
        assert math.isfinite(headline[k])
