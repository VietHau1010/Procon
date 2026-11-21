"""Comparisons must isolate the common decoder and use frozen model types."""

from pathlib import Path
import tempfile
import unittest

from hexudon.offline.contest_compare import ContestBot, load_baseline, source_hashes
from hexudon.offline.mapgen import generate
from hexudon.training.storage import snapshot_sources


class CompleteSnapshotComparisonTest(unittest.TestCase):
    def test_snapshot_includes_shared_decoder_and_accepts_working_tree_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot_sources(root)
            hashes = source_hashes(root)
            self.assertIn("planner/decoder.py", hashes)
            self.assertIn("planner_contest/tanker.py", hashes)
            cls = load_baseline(root)
            cfg = generate(seed=8, size=8, agents=3, days=4)
            cfg.players = 59
            bot = ContestBot(cls, cfg, "frozen", 0, 0, 0, None)
            self.assertIsNot(bot.strategist.config, cfg)
            self.assertEqual(bot.strategist.config.players, 59)
            self.assertEqual(bot.select_roles().count(1), 1)
            self.assertNotEqual(bot.strategist.__class__.__module__.split(".")[0], "hexudon")
            path = root / "baseline" / "planner" / "decoder.py"
            path.write_text(path.read_text(encoding="utf-8") + "\n# snapshot hash test\n", encoding="utf-8")
            self.assertNotEqual(source_hashes(root)["planner/decoder.py"], hashes["planner/decoder.py"])
