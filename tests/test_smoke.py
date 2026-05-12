"""
Smoke tests for the pas-mri-augmentation-radiomics repository.

These tests do **not** train or run inference. They verify that:
1. Every Python module in the repository imports cleanly (no path leaks,
   no missing dependencies, no syntax errors).
2. The audit pipeline runs end-to-end on a tiny synthetic CSV pair and
   produces the expected JSON keys.

Run from the repository root:
    pytest tests/

If you do not have pytest installed:
    python -m unittest tests/test_smoke.py
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestImports(unittest.TestCase):
    """Every top-level module must import without error."""

    TOP_LEVEL_MODULES = [
        "augmentations.afa",
        "augmentations.dual_norm",
        "augmentations.mixup_binary",
        "augmentations.cutmix_binary",
    ]

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT))

    def test_imports(self):
        for name in self.TOP_LEVEL_MODULES:
            with self.subTest(module=name):
                importlib.import_module(name)


class TestAuditPipeline(unittest.TestCase):
    """audit_numbers.py runs end-to-end on a tiny synthetic dataset."""

    def test_audit_with_synthetic_csvs(self):
        """Build a minimal pair of CSVs that satisfy the joins, and run the
        audit script in a temporary repo root. Verify the output JSON has
        the expected top-level keys."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "shareable").mkdir()
            (tmp / "radiomics" / "comparisons" / "2d").mkdir(parents=True)
            (tmp / "paper").mkdir()

            # Twelve augmentation models (paper scope), 2 base modalities,
            # 5 corruptions, two corruption modes (3d + clean) — minimal grid.
            models = [
                f"{td}_unetplusplus_{aug}"
                for td in ("BTFE", "TSE")
                for aug in ("1Fold", "mixup", "cutmix",
                            "afa", "mixup_afa", "cutmix_afa")
            ]
            corruptions = ["bias_field", "ghosting", "kspace_sub",
                           "rician_noise", "spike_noise"]

            metric_rows = []
            percorr_rows = []
            for m in models:
                td = "BTFE" if m.startswith("BTFE_") else "TSE"
                for bd in ("BTFE", "TSE"):
                    # one clean cell per (model, base_domain)
                    metric_rows.append((m, m, td, "clean", bd, "clean", 0,
                                        "clean", int(td == bd), 0,
                                        0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                    percorr_rows.append((m, m, bd, "clean", "clean",
                                         0.05, 0.04, 0.0, 100))
                    for corr in corruptions:
                        metric_rows.append((
                            m, m, td, f"{corr}_s3_3d", bd, corr, 3, "3d",
                            int(td == bd), 0,
                            0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        ))
                        percorr_rows.append((m, m, bd, corr, "3d",
                                             0.10, 0.08, 0.01, 100))

            import pandas as pd
            metrics_cols = ["model", "model_display", "train_domain",
                            "test_variant", "base_domain", "corruption",
                            "severity", "corruption_mode", "same_domain",
                            "fold", "dice", "iou", "sens", "prec", "spec",
                            "hd95", "msd"]
            percorr_cols = ["model", "model_display", "base_domain",
                            "corruption", "corruption_mode",
                            "feature_rel_error",
                            "feature_rel_error_median",
                            "feature_rel_error_std", "n"]
            pd.DataFrame(metric_rows, columns=metrics_cols).to_csv(
                tmp / "shareable" / "all_test_metrics.csv", index=False)
            pd.DataFrame(percorr_rows, columns=percorr_cols).to_csv(
                tmp / "radiomics" / "comparisons" / "2d"
                / "per_corruption_summary.csv", index=False)

            # 102-row feature CSV (any 102 placeholder features satisfy the
            # asserted feature count).
            feat_rows = [(f"original_test_feat{i}", 0.1, 0.1, 0.05, 0.5, 100)
                         for i in range(102)]
            pd.DataFrame(feat_rows, columns=["feature", "rel_mean",
                                             "rel_median", "rel_std",
                                             "rel_max", "n"]).to_csv(
                tmp / "radiomics" / "comparisons" / "2d"
                / "summary_by_feature.csv", index=False)

            # Run the real audit script against the synthetic root.
            audit_script = ROOT / "paper" / "audit_numbers.py"
            test_script = tmp / "paper" / "audit_numbers.py"
            test_script.write_text(audit_script.read_text())

            result = subprocess.run(
                [sys.executable, str(test_script)],
                cwd=tmp,
                capture_output=True, text=True,
            )
            self.assertEqual(
                result.returncode, 0,
                msg=f"audit failed:\nstdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}")

            audit_out = tmp / "paper" / "audit_3d_results.json"
            self.assertTrue(audit_out.is_file())
            data = json.loads(audit_out.read_text())
            for key in ("scope", "correlations_corrupted_only",
                        "catastrophic", "table_rows",
                        "mixup_afa_winrate", "per_corruption"):
                self.assertIn(key, data)
            self.assertEqual(data["scope"]["n_total_cells_3d"], 144)
            self.assertEqual(data["scope"]["n_corrupted_cells_3d"], 120)


if __name__ == "__main__":
    unittest.main()
