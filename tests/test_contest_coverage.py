"""Coverage selection must use actual simulator results, not targeted spots."""

import unittest
from unittest.mock import patch

from hexudon.model import AgentState, DayState, MatchConfig, Spot
from hexudon.planner.solution import DaySolution, Meet
from hexudon.planner_contest.coverage import _insert, rescue_coverage, rescue_servings
from hexudon.planner_contest.decoder import Decoded, decode
from hexudon.planner_contest.engine import make_context
from hexudon.planner_contest.objective import coverage_key
from hexudon.planner_contest.strategy import MatchStrategist
from hexudon.simulator import DayResult, run_day


def context(steps=4, fuel=20, target=1):
    cfg = MatchConfig(width=4, height=1, cells=[0]*4,
                      spots=[Spot(0, 0, 2), Spot(0, 2, 2), Spot(1, target, 2)],
                      agent_starts=[0], fuel_limits=fuel,
                      day_steps=[steps, steps], day_seconds=[5, 5], players=1,
                      busy_threshold=1, jammed_threshold=3)
    return make_context(cfg, 0, [AgentState(0, 0, fuel)], {}, [2, 2, 2], {0, 1})


class CoverageTest(unittest.TestCase):
    def test_contest_rescue_keeps_the_preexisting_all_day_policy(self):
        from hexudon.planner_contest.engine import plan_day
        from hexudon.planner_contest import engine
        ctx = context()
        sol = DaySolution(tours=[[0, 1]])
        ctx.days_left = 1
        called = []
        with patch.object(engine, "rescue_coverage",
                          side_effect=lambda c, s, d, **kw: (called.append(c.day) or s, d)):
            plan_day(ctx, max_iterations=0)
        self.assertEqual(called, [0])

    def test_contest_rescue_rejects_lower_coverage_candidate(self):
        # The rescue/polish boundary accepts candidates using the actual
        # collected-brand key, not a larger scalar heuristic score.
        ctx = context()
        incumbent = decode(ctx, DaySolution(tours=[[0, 2]]))
        worse = decode(ctx, DaySolution(tours=[[0]]))
        worse.score = incumbent.score + 1_000_000
        self.assertGreater(coverage_key(ctx, incumbent), coverage_key(ctx, worse))

    def test_missing_brand_is_rescued_even_if_its_old_stop_is_truncated(self):
        ctx = context(steps=6, target=3)
        ctx.day = 1
        ctx.days_left = 0
        # Brand 1 is targeted after a wait that consumes all useful travel time.
        sol = DaySolution(tours=[[1, 2]], waits={(0, 0): 2})
        before = decode(ctx, sol)
        self.assertEqual(before.result.brands, {0})
        fixed, result = rescue_coverage(ctx, sol, before)
        self.assertEqual(result.result.brands, {0, 1})
        self.assertTrue(run_day(ctx.config, ctx.steps, ctx.agents, result.plan,
                                ctx.stocks, strict_fuel=True).ok)
        self.assertEqual(sol.waits, {(0, 0): 2})

    def test_rescue_does_not_trade_tomorrows_value_for_daily_types(self):
        ctx = context(steps=6, target=3)
        sol = DaySolution(tours=[[1, 2]], waits={(0, 0): 2})
        before = decode(ctx, sol)
        with patch("hexudon.planner_contest.coverage.end_value",
                   side_effect=lambda ctx, r: -100 if 1 in r.brands else 0):
            _, result = rescue_coverage(ctx, sol, before)
        self.assertIs(result, before)

    def test_serving_rescue_keeps_coverage_and_adds_a_real_same_brand_serving(self):
        cfg = MatchConfig(width=4, height=1, cells=[0]*4,
                          spots=[Spot(0, 0, 2), Spot(0, 2, 2), Spot(1, 3, 2)],
                          agent_starts=[0], fuel_limits=20,
                          day_steps=[4, 4], day_seconds=[5, 5], players=1,
                          busy_threshold=1, jammed_threshold=3)
        ctx = make_context(cfg, 1, [AgentState(0, 0, 20)], {}, [2, 2, 2], {0, 1})
        sol = DaySolution(tours=[[0]])
        before = decode(ctx, sol)
        self.assertEqual((len(before.result.brands), before.result.servings), (1, 1))
        improved_sol, improved = rescue_servings(ctx, sol, before)
        self.assertEqual(improved.result.brands, before.result.brands)
        self.assertGreater(improved.result.servings, before.result.servings)
        self.assertTrue(run_day(cfg, ctx.steps, ctx.agents, improved.plan,
                                ctx.stocks, strict_fuel=True).ok)

    def test_serving_rescue_returns_incumbent_when_q_cannot_improve(self):
        ctx = context(steps=1)
        sol = DaySolution(tours=[[0]])
        before = decode(ctx, sol)
        self.assertIs(rescue_servings(ctx, sol, before)[1], before)

    def test_serving_rescue_moves_truncated_same_brand_stop_forward(self):
        cfg = MatchConfig(width=6, height=1, cells=[0]*6,
                          spots=[Spot(0, 0, 1), Spot(0, 2, 1), Spot(1, 4, 1)],
                          agent_starts=[0], fuel_limits=20,
                          day_steps=[4], day_seconds=[5], players=1,
                          busy_threshold=1, jammed_threshold=3)
        ctx = make_context(cfg, 0, [AgentState(0, 0, 20)], {}, [1, 1, 1], set())
        # The intended brand-0 stop at cell 2 is behind an unreachable cell-4
        # stop, so it exists in the tour but has not produced a real serving.
        sol = DaySolution(tours=[[2, 1]])
        before = decode(ctx, sol)
        self.assertEqual(before.result.brands, {0})
        improved_sol, improved = rescue_servings(ctx, sol, before)
        self.assertEqual(improved.result.brands, {0})
        self.assertGreater(improved.result.servings, before.result.servings)
        self.assertEqual(improved_sol.tours[0][0], 1)

    def test_empty_stock_and_unreachable_brand_are_not_counted(self):
        ctx = context(steps=1)
        sol = DaySolution(tours=[[0]])
        before = decode(ctx, sol)
        _, result = rescue_coverage(ctx, sol, before)
        self.assertIs(result, before)
        ctx = context(steps=6)
        ctx.stocks[2] = 0
        before = decode(ctx, sol)
        _, result = rescue_coverage(ctx, sol, before)
        self.assertIs(result, before)

    def test_no_fuel_does_not_create_a_phantom_collection(self):
        ctx = context(fuel=0)
        sol = DaySolution(tours=[[0]])
        before = decode(ctx, sol)
        _, result = rescue_coverage(ctx, sol, before)
        self.assertIs(result, before)

    def test_insertion_preserves_or_shifts_rendezvous_and_waits(self):
        sol = DaySolution(tours=[[0, 1]], meets=[Meet(0, -1, 0), Meet(0, 1, 0)],
                          waits={(0, -1): 1, (0, 1): 2})
        inserted = _insert(sol, 0, 1, 2)
        self.assertEqual(inserted.tours, [[0, 2, 1]])
        self.assertEqual(inserted.meets, [Meet(0, -1, 0), Meet(0, 2, 0)])
        self.assertEqual(inserted.waits, {(0, -1): 1, (0, 2): 2})
        cut = _insert(sol, 0, 1, 2, cut_tail=True)
        self.assertEqual(cut.meets, [Meet(0, -1, 0)])
        self.assertEqual(cut.waits, {(0, -1): 1})

    def test_deadline_and_evaluation_budget_return_incumbent(self):
        ctx = context()
        sol = DaySolution(tours=[[0]])
        before = decode(ctx, sol)
        self.assertIs(rescue_coverage(ctx, sol, before, deadline=0)[1], before)
        self.assertIs(rescue_coverage(ctx, sol, before, max_evaluations=0)[1], before)

    def test_archive_prefers_types_over_arbitrary_scalar_score(self):
        ctx = context()
        a = Decoded([], DayResult(ok=True, brands={0}), 1_000_000)
        b = Decoded([], DayResult(ok=True, brands={0, 1}), -1_000_000)
        self.assertGreater(coverage_key(ctx, b), coverage_key(ctx, a))
        ctx.visited_brands = {0}
        fresh_only = Decoded([], DayResult(ok=True, brands={1}), 0)
        self.assertGreater(coverage_key(ctx, fresh_only), coverage_key(ctx, a))

    def test_submit_cannot_overwrite_more_types_with_high_heuristic(self):
        ctx = context()
        strat = MatchStrategist(ctx.config)
        strat.visited_brands = {0, 1}
        better = decode(ctx, DaySolution(tours=[[0, 2]]))
        worse = decode(ctx, DaySolution(tours=[[0]]))
        worse.score = better.score + 1_000_000
        plans = []

        def fake(ctx, on_plan=None, **kwargs):
            on_plan(better)
            on_plan(worse)
            return worse

        with patch("hexudon.planner_contest.strategy.plan_day", fake):
            applied = strat.plan_day(DayState(0, ctx.agents, {}),
                                     submit=lambda plan: plans.append(plan) or True, record=False)
        self.assertEqual(plans, [better.plan])
        self.assertIs(applied, better)

    def test_alns_archive_retains_more_types_despite_lower_score(self):
        from hexudon.planner_contest.alns import improve
        ctx = context()
        low = Decoded([], DayResult(ok=True, brands={0}), 10000)
        high = Decoded([], DayResult(ok=True, brands={0, 1}), 1)
        with patch("hexudon.planner_contest.alns.decode", side_effect=[low, high]), \
             patch("hexudon.planner_contest.alns.repair", side_effect=lambda ctx, sol, *a, **k: sol):
            result = improve(ctx, start=ctx.empty_solution(), max_iterations=1)
        self.assertIs(result.decoded, high)
