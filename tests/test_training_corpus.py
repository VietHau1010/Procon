"""Diverse scenarios are unique, reproducible, and separated by source family."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from hexudon.model import MatchConfig
from hexudon.offline.mapgen import generate
from hexudon.training import corpus, trainer, storage


class TrainingCorpusTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.web = self.root/"web"
        self.web.mkdir()
        self.base = generate(seed=123, size=8, agents=3, spots=8, chains=4, days=4)
        (self.web/"known.json").write_text(json.dumps(self.base.to_json()), encoding="utf-8")

    def build(self, name="corpus", **kwargs):
        root = self.root/name
        manifest = corpus.build(self.web, root, random_families=3, per_family=17, seed=234, **kwargs)
        return root, manifest

    def test_all_profiles_are_valid_and_preserve_base(self):
        original = self.base.to_json()
        identities = set()
        for index, profile in enumerate(corpus.PROFILES):
            cfg = corpus.vary(self.base, profile, seed=900+index)
            corpus.validate_config(cfg)
            self.assertGreaterEqual(cfg.n_agents, 3)
            self.assertLessEqual(cfg.n_agents, 8)
            identities.add(corpus.case_identity(cfg.to_json()))
        self.assertEqual(self.base.to_json(), original)
        self.assertEqual(len(identities), len(corpus.PROFILES))

    def test_corpus_has_unique_cases_and_family_separated_holdouts(self):
        root, manifest = self.build()
        self.assertEqual(len(manifest["cases"]), 68)
        ids, families = set(), {}
        for item in manifest["cases"]:
            payload = json.loads((root/item["path"]).read_text(encoding="utf-8"))
            corpus.validate_config(MatchConfig.from_json(payload))
            self.assertEqual(corpus.case_identity(payload), item["id"])
            self.assertNotIn(item["id"], ids)
            ids.add(item["id"])
            self.assertEqual(families.setdefault(item["family"], item["split"]), item["split"])
        for family in manifest["families"]:
            if family["kind"] == "known-web":
                self.assertEqual(family["split"], "train")
        self.assertEqual(set(manifest["counts"]), {"train", "validation", "test"})

    def test_reproducible_output_without_overwriting_existing_corpus(self):
        root, first = self.build("a")
        _, second = self.build("b")
        self.assertEqual(first, second)
        before = (root/"manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "exists"):
            corpus.build(self.web, root)
        self.assertEqual((root/"manifest.json").read_bytes(), before)

    def test_batches_rotate_without_recycling_or_touching_other_splits(self):
        root, manifest = self.build()
        first = corpus.batch(root, "train", 4, index=0)
        second = corpus.batch(root, "train", 4, index=1)
        self.assertFalse({c["id"] for c in first} & {c["id"] for c in second})
        self.assertEqual(first, corpus.batch(root, "train", 4, index=0))
        self.assertTrue(all(c["split"] == "train" for c in first+second))
        with self.assertRaises(ValueError):
            corpus.batch(root, "train", 10000)

    def test_batch_detects_modified_case_and_leaked_family(self):
        root, manifest = self.build()
        selected = corpus.batch(root, "train", 1)[0]
        path = root/selected["path"]
        old = path.read_bytes()
        payload = json.loads(old)
        payload["fuelLimits"] += 1
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "modified"):
            corpus.batch(root, "train", 1)
        path.write_bytes(old)
        manifest["cases"][1]["split"] = "test"
        (root/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Family leaked"):
            corpus.batch(root, "train", 1)

    def test_training_can_preserve_per_case_team_count(self):
        specs = {}
        for index, split in enumerate(("train", "validation", "test")):
            cfg = generate(seed=200+index, size=8, agents=3, days=4)
            cfg.players = (4, 16, 59)[index]
            path = self.root/(split+".json")
            path.write_text(json.dumps(cfg.to_json()), encoding="utf-8")
            specs[split] = [(str(path), 0)]
        run = self.root/"run"
        manifest = trainer.initialize(run, specs, iterations=1, population=1, players=None)
        self.assertIsNone(manifest["settings"]["players"])
        for split, expected in zip(("train", "validation", "test"), (4, 16, 59)):
            self.assertEqual(storage.read_json(run/"data"/(split+".json"))[0]["config"]["players"], expected)

    def test_init_batch_records_selected_families_without_test_evaluation(self):
        root, _ = self.build()
        run = self.root/"batch-run"
        corpus.init_batch(root, run, size=4, iterations=1, population=1)
        storage.verify_run(run)
        chosen = storage.read_json(run/"corpus-selection.json")
        ownership = {}
        for split, cases in chosen.items():
            self.assertEqual(len(cases), 4)
            for case in cases:
                self.assertEqual(ownership.setdefault(case["family"], split), split)
        self.assertEqual(storage.read_json(run/"state.json")["trials"], 0)
        self.assertFalse((run/"evaluations").exists())

    def test_validator_rejects_unreachable_or_invalid_stock(self):
        cfg = copy.deepcopy(self.base)
        cfg.cells = [3]*cfg.n_cells
        for p in set(cfg.agent_starts) | {s.pos for s in cfg.spots}:
            cfg.cells[p] = 0
        cfg.__post_init__()
        with self.assertRaises(ValueError):
            corpus.validate_config(cfg)
