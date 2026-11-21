"""Offline learning: immutable baseline, disjoint data, resume and promotion."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hexudon.training import storage, trainer
from hexudon.training.evaluator import evaluate, summarize
from hexudon.training.inference import trained_strategist


SPECS = {"train": [("random8", 1)], "validation": [("random8", 2)],
         "test": [("random8", 3)]}


class TrainingTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "run"

    def init(self, **kwargs):
        return trainer.initialize(self.root, SPECS, iterations=1,
                                  role_iterations=0, population=2, **kwargs)

    def test_snapshot_is_verified_and_baseline_is_initial_champion(self):
        manifest = self.init()
        self.assertEqual(storage.verify_run(self.root), manifest)
        self.assertEqual(storage.load_policy(self.root), manifest["baseline_weights"])
        self.assertFalse((self.root / "baseline" / ".env").exists())
        self.assertFalse((self.root / "baseline" / "client").exists())
        self.assertNotEqual(storage.load_baseline(self.root)["model"].__name__, "hexudon.model")

    def test_existing_directory_is_not_overwritten(self):
        self.init()
        before = (self.root / "manifest.json").read_bytes()
        with self.assertRaises(ValueError):
            self.init()
        self.assertEqual((self.root / "manifest.json").read_bytes(), before)

    def test_same_map_in_two_splits_is_rejected_before_writing(self):
        specs = dict(SPECS, validation=[("random8", 1)])
        with self.assertRaisesRegex(ValueError, "Rò rỉ"):
            trainer.initialize(self.root, specs)
        self.assertFalse(self.root.exists())

    def test_rescheduled_sample_map_is_still_the_same_map(self):
        a = {"map": {"cells": [0]}, "startsAt": 10, "players": 4, "daySeconds": [2]}
        b = dict(a, startsAt=20, players=59, daySeconds=[45])
        self.assertEqual(trainer.map_identity(a), trainer.map_identity(b))

    def test_changed_baseline_data_or_manifest_are_detected(self):
        self.init()
        path = self.root / "baseline" / "rules.py"
        original = path.read_bytes()
        path.write_bytes(original + b"\n# changed\n")
        with self.assertRaisesRegex(ValueError, "source"):
            storage.verify_run(self.root)
        path.write_bytes(original)
        path = self.root / "data" / "train.json"
        path.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Dataset"):
            storage.verify_run(self.root)

    def test_mutation_is_repeatable_bounded_and_cannot_change_priority(self):
        manifest = self.init()
        base = manifest["baseline_weights"]
        one = trainer.mutate(base, base, 3, 1, 2)
        self.assertEqual(one, trainer.mutate(base, base, 3, 1, 2))
        self.assertNotEqual(one, base)
        self.assertEqual(one["B"], base["B"])
        self.assertEqual(storage.validate_weights(one, base), one)
        for key, value in (("B", 1), ("serving", float("nan")), ("near_chain", -1)):
            broken = dict(base, **{key: value})
            with self.assertRaises(ValueError):
                storage.validate_weights(broken, base)

    def test_bad_model_is_rejected(self):
        self.init()
        path = self.root / "champion.json"
        obj = storage.read_json(path)
        obj["baseline_id"] = "not-this-run"
        storage.atomic_json(path, obj)
        with self.assertRaises(ValueError):
            storage.load_policy(self.root)

    def test_writer_lock_prevents_two_trainers(self):
        self.init()
        with storage.writer_lock(self.root):
            with self.assertRaises(ValueError):
                with storage.writer_lock(self.root):
                    self.fail("second writer entered")
        self.assertFalse((self.root / ".writer.lock").exists())

    def test_gate_rejects_one_loss_even_when_totals_are_higher(self):
        rows = [{"baseline": [10, 20, 30], "candidate": [10, 19, 200], "invalid": [0]*4},
                {"baseline": [10, 20, 30], "candidate": [10, 40, 60], "invalid": [0]*4}]
        self.assertFalse(summarize(rows)["passed"])
        rows[0]["candidate"] = rows[0]["baseline"]
        self.assertTrue(summarize(rows)["passed"])
        rows[1]["invalid"][1] = 1
        self.assertFalse(summarize(rows)["passed"])

    def test_interrupt_does_not_promote_a_partial_trial_and_resume_repeats_it(self):
        self.init()
        attempts = []

        def interrupted(directory, split, left, right, *args):
            attempts.append(dict(right))
            raise KeyboardInterrupt

        with patch.object(trainer, "evaluate", interrupted):
            state = trainer.run(self.root, log=lambda s: None)
        self.assertEqual(state["trials"], 0)
        self.assertEqual(state["promotions"], 0)
        tied = {"passed": True, "wins": 0, "ties": 1, "losses": 0, "invalid": 0}

        def resumed(directory, split, left, right, *args):
            attempts.append(dict(right))
            self.assertEqual(split, "train")
            return tied

        with patch.object(trainer, "evaluate", resumed):
            state = trainer.run(self.root, log=lambda s: None)
        self.assertEqual(attempts[0], attempts[1])
        self.assertEqual(state["generation"], 1)
        self.assertEqual(state["trials"], 2)

    def test_only_complete_train_and_validation_pass_promotes_no_test_leak(self):
        self.init()
        calls = []
        passed = {"passed": True, "wins": 1, "ties": 0, "losses": 0, "invalid": 0}

        def evaluator(directory, split, left, right, *args):
            calls.append(split)
            return passed

        with patch.object(trainer, "evaluate", evaluator):
            state = trainer.run(self.root, log=lambda s: None)
        self.assertEqual(state["promotions"], 2)
        self.assertNotIn("test", calls)
        self.assertIn("validation", calls)
        self.assertEqual(storage.load_policy(self.root), state["weights"])
        manifest = storage.verify_run(self.root)
        self.assertNotEqual(state["weights"], manifest["baseline_weights"])

    def test_validation_loss_blocks_model_even_when_train_wins(self):
        self.init()

        def evaluator(directory, split, *args):
            return {"passed": split == "train", "wins": 1, "ties": 0, "losses": int(split != "train")}

        with patch.object(trainer, "evaluate", evaluator):
            state = trainer.run(self.root, log=lambda s: None)
        self.assertEqual(state["promotions"], 0)
        self.assertEqual(state["champion_trial"], "baseline")

    def test_small_real_match_and_frozen_inference(self):
        manifest = self.init()
        base = manifest["baseline_weights"]
        report = evaluate(self.root, "train", base, base, manifest)
        self.assertEqual(report["losses"], 0)
        self.assertEqual(report["wins"], 0)
        self.assertEqual(report["invalid"], 0)
        self.assertEqual(report["rows"][0]["roles"][0].count(1), 1)
        obj = storage.read_json(self.root / "data" / "train.json")[0]["config"]
        bot = trained_strategist(self.root, obj, search_iterations=1, role_iterations=0)
        self.assertEqual(bot.select_roles().count(1), 1)

    def test_test_evaluation_does_not_change_champion_or_checkpoint(self):
        self.init()
        before = (self.root / "state.json").read_bytes()
        champ = (self.root / "champion.json").read_bytes()
        report = trainer.evaluate_policy(self.root)
        self.assertEqual(report["losses"], 0)
        self.assertEqual((self.root / "state.json").read_bytes(), before)
        self.assertEqual((self.root / "champion.json").read_bytes(), champ)
        self.assertTrue(list((self.root / "evaluations").glob("test-*.json")))

    def test_cli_has_offline_training_commands(self):
        from hexudon.cli import build_parser
        args = build_parser().parse_args(["train", "run", "--run-dir", "x", "--generations", "0"])
        self.assertEqual(args.generations, 0)
        self.assertFalse(hasattr(args, "live"))
