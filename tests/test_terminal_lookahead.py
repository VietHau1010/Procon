"""Next-day parking is opt-in, simulator-checked, and bounded by a deadline."""

import unittest
from unittest.mock import patch

from hexudon.model import AgentState, DayState, MatchConfig, Spot
from hexudon.planner_contest import engine
from hexudon.planner_contest.decoder import Decoded
from hexudon.planner_contest.lookahead import (
    choose_terminal_plan, parking_candidates, projected_roads, two_day_key,
)
from hexudon.planner_contest.objective import score_day
from hexudon.planner_contest.strategy import MatchStrategist
from hexudon.simulator import DayResult, run_day, wait_plan


def context():
    cfg = MatchConfig(width=6, height=1, cells=[0]*6,
                      spots=[Spot(0, 1, 2), Spot(0, 3, 2)],
                      agent_starts=[1, 0], fuel_limits=20,
                      day_steps=[2, 4], day_seconds=[5, 5], players=1,
                      busy_threshold=1, jammed_threshold=3)
    return engine.make_context(cfg, 0, [AgentState(0, 1, 0), AgentState(1, 0, 0)],
                               {}, [2, 2], set())


def idle(ctx):
    plan = wait_plan(len(ctx.agents), ctx.steps)
    result = run_day(ctx.config, ctx.steps, ctx.agents, plan, ctx.stocks,
                     ctx.road_status, strict_fuel=True)
    return Decoded(plan, result, score_day(ctx, result))


class TerminalLookaheadTest(unittest.TestCase):
    def test_tanker_parking_refills_with_real_fuel_and_preserves_collections(self):
        ctx = context()
        before = idle(ctx)
        candidates = parking_candidates(ctx, before)
        self.assertTrue(candidates)
        self.assertEqual(before.result.end_fuel[0], 0)
        self.assertTrue(any(c.result.end_fuel[0] == 20 for c in candidates))
        for candidate in candidates:
            check = run_day(ctx.config, ctx.steps, ctx.agents, candidate.plan,
                            ctx.stocks, ctx.road_status, strict_fuel=True)
            self.assertTrue(check.ok)
            self.assertEqual(check.collections, before.result.collections)
            self.assertEqual([a.kind for a in ctx.agents], [0, 1])

    def test_real_next_day_can_prefer_prepositioned_tanker(self):
        ctx = context()
        before = idle(ctx)
        after, report = choose_terminal_plan(ctx, before, iterations=0)
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["selected"])
        self.assertGreater(tuple(report["best"]), tuple(report["baseline"]))
        self.assertEqual(after.result.servings, before.result.servings)
        self.assertEqual(after.result.brands, before.result.brands)
        self.assertEqual(after.result.end_fuel[0], ctx.config.fuel_limits)

    def test_no_speculative_use_of_refill_received_at_end_of_wait(self):
        ctx = context()
        ctx.config.spots = [Spot(0, 3, 2)]
        ctx.config.day_steps = [4, 4]
        ctx.config.__post_init__()
        ctx = engine.make_context(ctx.config, 0,
                                  [AgentState(0, 2, 0), AgentState(1, 0, 0)], {}, [2], set())
        path = ctx.book.options(0, 2)[0].path
        car, cell = [], 0
        for nxt in path:
            car.append(ctx.config.dir_index[cell][nxt])
            cell = nxt
        plan = [[-4], car]
        result = run_day(ctx.config, 4, ctx.agents, plan, [2], strict_fuel=True)
        self.assertTrue(result.ok)
        self.assertEqual(result.end_fuel[0], 20)
        before = Decoded(plan, result, score_day(ctx, result))
        self.assertEqual(parking_candidates(ctx, before), [])

    def test_expired_deadline_and_final_day_do_not_run_projection(self):
        ctx = context()
        before = idle(ctx)
        with patch("hexudon.planner_contest.lookahead.project_next_day") as project:
            self.assertIs(choose_terminal_plan(ctx, before, deadline=0)[0], before)
            ctx.days_left = 0
            self.assertIs(choose_terminal_plan(ctx, before)[0], before)
        project.assert_not_called()

    def test_incomplete_comparison_keeps_incumbent(self):
        ctx = context()
        before = idle(ctx)
        alternative = parking_candidates(ctx, before)[0]
        now = [0.0]

        def project(*args, **kwargs):
            now[0] = 100.0
            return before

        with patch("hexudon.planner_contest.lookahead.parking_candidates", return_value=[alternative]), \
             patch("hexudon.planner_contest.lookahead.time.monotonic", side_effect=lambda: now[0]), \
             patch("hexudon.planner_contest.lookahead.project_next_day", side_effect=project):
            after, report = choose_terminal_plan(ctx, before, deadline=50)
        self.assertIs(after, before)
        self.assertEqual(report["status"], "deadline")
        self.assertFalse(report["selected"])

    def test_equal_projected_score_keeps_incumbent(self):
        ctx = context()
        before = idle(ctx)
        alternatives = parking_candidates(ctx, before)
        with patch("hexudon.planner_contest.lookahead.parking_candidates", return_value=alternatives), \
             patch("hexudon.planner_contest.lookahead.project_next_day", return_value=before):
            after, report = choose_terminal_plan(ctx, before)
        self.assertIs(after, before)
        self.assertFalse(report["selected"])

    def test_two_day_key_prioritizes_types_over_servings(self):
        ctx = context()
        today = idle(ctx)
        high_q = Decoded([], DayResult(ok=True, brands=set(), collections=[(1, 0, 0)]*10000), 0)
        new_type = Decoded([], DayResult(ok=True, brands={1}, collections=[]), 0)
        self.assertGreater(two_day_key(ctx, today, new_type), two_day_key(ctx, today, high_q))

    def test_projection_cannot_ignore_candidate_own_road_traffic(self):
        ctx = context()
        ctx.config.cells[0] = 1
        ctx.config.__post_init__()
        ctx = engine.make_context(ctx.config, 0, ctx.agents, {}, [2, 2], set())
        result = idle(ctx).result
        result.stay[0] = 100
        self.assertEqual(projected_roads(ctx, result)[0], 2)

    def test_default_engine_and_strategist_do_not_enable_experiment(self):
        ctx = context()
        with patch("hexudon.planner_contest.lookahead.choose_terminal_plan") as choose:
            self.assertTrue(engine.plan_day(ctx, max_iterations=0).result.ok)
        choose.assert_not_called()
        strategist = MatchStrategist(ctx.config, search_iterations=0, fast_roles=True)
        self.assertFalse(strategist.enable_lookahead)

    def test_opt_in_engine_emits_projection_report(self):
        ctx = context()
        reports = []
        result = engine.plan_day(ctx, max_iterations=0, enable_dense=False,
                                  enable_lookahead=True, on_lookahead=reports.append)
        self.assertTrue(result.result.ok)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["phase"], "lookahead")

    def test_submission_can_accept_validated_equal_q_with_better_future(self):
        ctx = context()
        before = idle(ctx)
        after = parking_candidates(ctx, before)[0]
        after.score = before.score - 10000
        strategist = MatchStrategist(ctx.config, search_iterations=0, enable_lookahead=True)
        sent = []

        def fake_plan_day(ctx, on_plan=None, on_lookahead=None, **kwargs):
            on_plan(before)
            on_lookahead({"selected": True, "status": "complete"})
            on_plan(after)
            return after

        with patch("hexudon.planner_contest.strategy.plan_day", side_effect=fake_plan_day):
            applied = strategist.plan_day(DayState(0, ctx.agents, {}), record=False,
                                           submit=lambda plan: sent.append(plan) or True)
        self.assertIs(applied, after)
        self.assertEqual(sent, [before.plan, after.plan])
