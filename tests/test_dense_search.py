"""Beam proposals are bounded, never phantom fuel, and validated as a team."""

import unittest
from unittest.mock import patch

from hexudon.model import AgentState, DayState, MatchConfig, Spot
from hexudon.planner.solution import DaySolution, Meet
from hexudon.planner_contest.engine import make_context
from hexudon.planner_contest.decoder import decode
from hexudon.planner_contest.dense_search import abundant, beam_tours, replace_tour, improve_dense
from hexudon.simulator import run_day


def context(fuel=20, steps=10):
    cfg = MatchConfig(width=6, height=1, cells=[0]*6,
                      spots=[Spot(0, p, 2) for p in range(1, 6)],
                      agent_starts=[0], fuel_limits=fuel, day_steps=[steps], day_seconds=[5],
                      players=1, busy_threshold=1, jammed_threshold=3)
    return make_context(cfg, 0, [AgentState(0, 0, fuel)], {}, [2]*5, set())


class DenseSearchTest(unittest.TestCase):
    def test_contest_package_exports_integrated_dense_strategy(self):
        from hexudon import planner_contest
        self.assertIs(planner_contest.improve_dense, improve_dense)
        self.assertIs(planner_contest.beam_tours, beam_tours)

    def test_engine_runs_dense_by_default_and_honors_explicit_disable(self):
        from hexudon.planner_contest import engine
        ctx = context()
        with patch.object(engine, "improve_dense", wraps=improve_dense) as dense:
            result = engine.plan_day(ctx, max_iterations=0)
        dense.assert_called_once()
        self.assertTrue(result.result.ok)
        with patch.object(engine, "improve_dense", wraps=improve_dense) as dense:
            result = engine.plan_day(ctx, max_iterations=0, enable_dense=False)
        dense.assert_not_called()
        self.assertTrue(result.result.ok)

    def test_default_match_strategist_uses_dense_and_submits_valid_plan(self):
        from hexudon.planner_contest import engine, MatchStrategist
        ctx = context()
        strategist = MatchStrategist(ctx.config, search_iterations=0, role_iterations=0)
        plans = []
        with patch.object(engine, "improve_dense", wraps=improve_dense) as dense:
            result = strategist.plan_day(DayState(0, ctx.agents, {}), record=False,
                                          submit=lambda plan: plans.append(plan) or True)
        dense.assert_called_once()
        self.assertTrue(plans)
        self.assertEqual(plans[-1], result.plan)
        for plan in plans:
            self.assertTrue(run_day(ctx.config, ctx.steps, ctx.agents, plan,
                                    ctx.stocks, strict_fuel=True).ok)

    def test_dense_predicate_uses_every_stock_not_average(self):
        ctx = context()
        self.assertTrue(abundant(ctx))
        ctx.stocks[0] = 0
        self.assertFalse(abundant(ctx))

    def test_beam_dispatch_skips_maps_with_mostly_unique_brands(self):
        ctx = context()
        ctx.config.spots = [Spot(i, s.pos, s.stocks) for i, s in enumerate(ctx.config.spots)]
        ctx.config.__post_init__()
        self.assertFalse(abundant(ctx))

    def test_beam_fits_step_and_real_initial_fuel(self):
        ctx = context(fuel=2, steps=10)
        tours = beam_tours(ctx, 0, width=8)
        self.assertTrue(tours)
        for tour in tours:
            self.assertEqual(len(set(tour)), len(tour))
            self.assertLessEqual(len(tour), 2)
            dec = decode(ctx, DaySolution(tours=[tour]))
            self.assertTrue(dec.result.ok)
        self.assertEqual(beam_tours(ctx, 0, deadline=0), [])

    def test_beam_can_return_to_requested_parking(self):
        ctx = context(steps=10)
        tours = beam_tours(ctx, 0, width=16, end_spot=2)
        self.assertTrue(tours)
        self.assertTrue(all(t[-1] == 2 for t in tours))

    def test_replacement_remaps_old_meets_and_drops_removed_waits(self):
        original = DaySolution(tours=[[0, 1, 2]], meets=[Meet(0, -1, 0), Meet(0, 1, 0)],
                               waits={(0, -1): 1, (0, 1): 2, (0, 0): 4})
        trial = replace_tour(original, 0, [2, 1, 3])
        self.assertEqual(trial.meets, original.meets)
        self.assertEqual(trial.waits, {(0, -1): 1, (0, 1): 2})
        self.assertEqual(original.tours, [[0, 1, 2]])

    def test_beam_increases_real_servings_keeping_types(self):
        ctx = context()
        sol = DaySolution(tours=[[0, 1, 2]])
        before = decode(ctx, sol)
        _, after = improve_dense(ctx, sol, before, width=16)
        self.assertEqual(after.result.brands, before.result.brands)
        self.assertEqual(after.result.servings, 5)
        self.assertTrue(run_day(ctx.config, ctx.steps, ctx.agents, after.plan,
                                ctx.stocks, strict_fuel=True).ok)
        self.assertIs(improve_dense(ctx, sol, before, deadline=0)[1], before)

    def test_empty_fuel_keeps_incumbent(self):
        ctx = context(fuel=0)
        sol = ctx.empty_solution()
        before = decode(ctx, sol)
        self.assertIs(improve_dense(ctx, sol, before)[1], before)
