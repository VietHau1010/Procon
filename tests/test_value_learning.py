"""Counterfactual data, train-only normalizers and experimental future-value model."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hexudon.model import MatchConfig, Spot
from hexudon.training.storage import read_json, atomic_json, digest, json_bytes
from hexudon.training.value import data, model
from hexudon.training.value.features import FEATURES, extract, future_bounds


def tiny_config(offset=0, days=3):
    return MatchConfig(width=6, height=1, cells=[0]*6,
                       spots=[Spot(0, 1, 2), Spot(1, 3, 2), Spot(2, 5, 2)],
                       agent_starts=[0, 2, 4], fuel_limits=8+offset,
                       day_steps=[6+offset]*days, day_seconds=[5]*days,
                       players=4, busy_threshold=1, jammed_threshold=3)


class ValueLearningTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "value"
        self.specs = {s: [(s, i)] for i, s in enumerate(("train", "validation", "test"))}

    def init(self, days=3):
        def load(name, seed=0):
            return tiny_config(seed, days)
        with patch("hexudon.training.trainer.load_or_generate", load):
            return data.initialize(self.root, self.specs, iterations=1, rollout_iterations=0,
                                   candidates=3, states_per_map=2)

    def test_generate_complete_future_labels_and_resume_without_overwrite(self):
        cfg = self.init()
        self.assertEqual(data.collect(self.root, "train"), 1)
        cases = read_json(self.root / "data" / "train.json")
        path = self.root / "samples" / "train" / (cases[0]["id"] + ".json")
        before = path.read_bytes()
        self.assertEqual(data.collect(self.root, "train"), 0)
        self.assertEqual(path.read_bytes(), before)
        groups, _ = data.read_groups(self.root, "train", cfg)
        self.assertEqual(len(groups), 2)
        self.assertGreater(sum(len(g["rows"]) for g in groups), 2)
        for group in groups:
            self.assertEqual(sum(row["baseline"] for row in group["rows"]), 1)
            for row in group["rows"]:
                self.assertEqual(row["rollout_days"], group["days_left"])
                self.assertEqual(row["total_gain"], [a+b for a, b in zip(row["gain_today"], row["future"])])
                self.assertEqual(len(row["features"]), len(FEATURES))
                self.assertTrue(all(0 <= v <= b for v, b in zip(row["future"], row["future_bounds"])))

    def test_incomplete_rollout_never_produces_a_sample_shard(self):
        self.init()
        with patch.object(data, "rollout", return_value=([0, 0, 0], -1)):
            with self.assertRaisesRegex(ValueError, "chưa đủ"):
                data.collect(self.root)
        self.assertFalse(list((self.root / "samples").glob("**/*.json")))

    def test_final_day_has_no_future_reward(self):
        cfg = self.init(days=1)
        data.collect(self.root)
        groups, _ = data.read_groups(self.root, "train", cfg)
        for group in groups:
            for row in group["rows"]:
                self.assertEqual(row["future"], [0, 0, 0])
                self.assertEqual(row["future_bounds"], [0, 0, 0])

    def test_fit_does_not_read_test_and_evaluation_does_not_deploy(self):
        self.init()
        data.collect(self.root, "train")
        data.collect(self.root, "validation")
        report = model.fit(self.root)
        self.assertIn("validation", report)
        self.assertFalse((self.root / "samples" / "test").exists())
        artifact = read_json(self.root / "value-model.json")
        self.assertTrue(artifact["experimental"])
        self.assertFalse(artifact["automatic_live_deployment"])
        normal_model_before = (self.root / "champion.json").read_bytes()
        data.collect(self.root, "test")
        result = model.evaluate(self.root, "test")
        self.assertGreater(result["metrics"]["rows"], 0)
        self.assertEqual((self.root / "champion.json").read_bytes(), normal_model_before)

    def test_tampered_sample_is_detected(self):
        cfg = self.init()
        data.collect(self.root)
        case = read_json(self.root / "data" / "train.json")[0]
        path = self.root / "samples" / "train" / (case["id"] + ".json")
        obj = read_json(path)
        obj["groups"][0]["rows"][0]["future"][2] += 1
        atomic_json(path, obj)
        with self.assertRaises(ValueError):
            data.read_groups(self.root, "train", cfg)

    def test_regression_learns_future_and_bounds_predictions(self):
        def row(x):
            features = [0.0]*len(FEATURES)
            features[0] = x
            return {"features": features, "future": [0, 10*x, 100*x],
                    "future_bounds": [0, 10, 100], "gain_today": [0, 1, 1],
                    "total_gain": [0, 1+10*x, 1+100*x], "baseline": True, "candidate": 0}
        groups = [{"id": str(i), "rows": [row(x)]} for i, x in enumerate((0, 0.25, 0.5, 0.75, 1))]
        learned = model.fit_ridge(groups, 0.01)
        small = model.predict_future(learned, row(0.2)["features"], [0, 10, 100])
        big = model.predict_future(learned, row(0.8)["features"], [0, 10, 100])
        self.assertEqual(small[0], 0)
        self.assertGreater(big[1], small[1])
        self.assertGreater(big[2], small[2])
        self.assertEqual(model.predict_future(learned, row(1)["features"], [0, 0, 0]), [0, 0, 0])
        with self.assertRaises(ValueError):
            model.predict_future(learned, [float("nan")]*len(FEATURES), [0, 10, 100])
        self.assertLess(model.metrics(learned, groups)["mean_mse"], 0.001)

    def test_branch_state_is_isolated_and_features_detect_fuel_changes(self):
        self.init()
        api = data.modules_for(self.root)
        cfg = api["model"].MatchConfig.from_json(tiny_config().to_json())
        state = {"day": 0, "status": {}, "yesterday_status": None, "previous_stay": [None]*4,
                 "history": [], "weights": api["objective"].default_contest_weights(),
                 "visited": [set() for _ in range(4)],
                 "agents": [[api["model"].AgentState(k, p, cfg.fuel_limits)
                             for k, p in zip([1, 0, 0], cfg.agent_starts)] for _ in range(4)]}
        forecasts = [api["traffic"].TrafficForecast(cfg, n_patrols=2) for _ in range(4)]
        results, contexts, _ = data._results(api, cfg, state, forecasts, 0, 0)
        branch = data._advance(api, cfg, state, results)
        original_fuel = state["agents"][0][1].fuel
        branch["agents"][0][1].fuel = 0
        self.assertEqual(state["agents"][0][1].fuel, original_fuel)
        ctx = contexts[0]
        values = extract(ctx, results[0], api, forecasts[0])
        less_fuel = results[0].end_fuel[:]
        results[0].end_fuel[1] = 0
        altered = extract(ctx, results[0], api, forecasts[0])
        results[0].end_fuel = less_fuel
        self.assertNotEqual(values, altered)
        self.assertEqual(len(future_bounds(ctx, results[0])), 3)
