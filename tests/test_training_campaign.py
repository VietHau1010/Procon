"""Fresh audit and convincing-win requirements for long training campaigns."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hexudon.training import campaign, trainer, storage


class CampaignTest(unittest.TestCase):
    def test_convincing_requires_many_cases_zero_losses_and_broad_wins(self):
        def report(wins, ties, losses=0, passed=True):
            return dict(wins=wins, ties=ties, losses=losses, passed=passed)
        self.assertTrue(campaign.convincing(report(15, 9)))
        self.assertFalse(campaign.convincing(report(2, 0)))
        self.assertFalse(campaign.convincing(report(10, 14)))
        self.assertFalse(campaign.convincing(report(15, 8, 1, False)))
        self.assertFalse(campaign.convincing(report(15, 9, passed=False)))

    def test_fresh_audits_are_disjoint_and_repeatable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            manifest = trainer.initialize(root, {
                "train": [("random8", 1)], "validation": [("random8", 2)],
                "test": [("random8", 3)]}, iterations=0, role_iterations=0)
            split, checksum = campaign._fresh_audit(root, manifest, 1)
            first = storage.read_json(root / "data" / (split + ".json"))
            self.assertEqual(len(first), 24)
            self.assertEqual(campaign._fresh_audit(root, manifest, 1), (split, checksum))
            split2, _ = campaign._fresh_audit(root, manifest, 2)
            second = storage.read_json(root / "data" / (split2 + ".json"))
            self.assertFalse({c["id"] for c in first} & {c["id"] for c in second})
            self.assertEqual(storage.verify_run(root), manifest)

    def test_budget_stops_between_trials_without_claiming_success(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "campaign"
            ref = Path(d) / "reference"
            specs = {"train": [("random8", 1)], "validation": [("random8", 2)],
                     "test": [("random8", 3)]}
            trainer.initialize(ref, specs, iterations=0, role_iterations=0)
            # Use tiny maps in this test; actual campaign uses broader datasets.
            with patch.object(campaign, "_case_specs", return_value=specs):
                campaign.initialize(root, ref, iterations=0, role_iterations=0)
            tied = {"wins": 0, "ties": 1, "losses": 0, "passed": True, "rows": [1]}
            with patch.object(campaign, "evaluate", return_value=tied):
                state = campaign.run(root, max_trials=2, hours=1, jobs=1)
            self.assertEqual(state["trials"], 2)
            self.assertEqual(state["accepted"], 0)
            self.assertFalse(state["qualified"])
            self.assertFalse((root / "qualified.json").exists())
            self.assertEqual(storage.load_policy(ref), storage.read_json(ref / "baseline.json")["weights"])
