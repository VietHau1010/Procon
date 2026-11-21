"""Route micro-optimization preserves anchors and improves real collections."""

import unittest
from unittest.mock import patch

from hexudon.model import MatchConfig, Spot, AgentState, DayState
from hexudon.planner.solution import DaySolution, Meet
from hexudon.planner_contest.engine import make_context
from hexudon.planner_contest.decoder import decode
from hexudon.planner_contest.local_search import reorder, shorten_route, shortcut_and_fill, _cost
from hexudon.planner_contest.strategy import MatchStrategist
from hexudon.simulator import run_day


def context():
    cfg = MatchConfig(width=6, height=1, cells=[0]*6,
                      spots=[Spot(0, p, 1) for p in range(1, 6)],
                      agent_starts=[0], fuel_limits=20, day_steps=[10], day_seconds=[5],
                      players=1, busy_threshold=1, jammed_threshold=3)
    return make_context(cfg, 0, [AgentState(0, 0, 20)], {}, [1]*5, set())


class LocalRouteTest(unittest.TestCase):
    def test_destroy_keeps_start_refuel_wait_and_meet(self):
        from hexudon.planner_contest.alns import _remove_stops
        sol = DaySolution(tours=[[0, 1, 2]], meets=[Meet(0, -1, 0), Meet(0, 2, 0)],
                          waits={(0, -1): 5, (0, 2): 3})
        altered = _remove_stops(sol, [(0, 1)])
        self.assertEqual(altered.meets, [Meet(0, -1, 0), Meet(0, 1, 0)])
        self.assertEqual(altered.waits, {(0, -1): 5, (0, 1): 3})
        self.assertEqual(sol.waits, {(0, -1): 5, (0, 2): 3})

    def test_archive_keeps_productive_restart_even_on_last_iteration(self):
        from hexudon.planner_contest import alns
        from hexudon.planner_contest.decoder import Decoded
        from hexudon.simulator import DayResult
        ctx = context()
        low = Decoded([], DayResult(ok=True, brands={0}, collections=[(1, 0, 0)]), 1)
        high = Decoded([], DayResult(ok=True, brands={0}, collections=[(1, 0, 0)]*2), 2)
        with patch.object(alns, "RESTART_AFTER", 1), \
             patch.object(alns, "decode", side_effect=[low, low, high]), \
             patch.object(alns, "repair", side_effect=lambda c, s, *a, **k: s):
            result = alns.improve(ctx, start=ctx.empty_solution(), max_iterations=1)
        self.assertIs(result.decoded, high)

    def test_reorder_attaches_meets_and_waits_to_occurrences(self):
        sol = DaySolution(tours=[[1, 1, 2]], meets=[Meet(0, -1, 0), Meet(0, 0, 0)],
                          waits={(0, -1): 1, (0, 1): 4})
        altered = reorder(sol, 0, [1, 0, 2])
        self.assertEqual(altered.meets, [Meet(0, -1, 0), Meet(0, 1, 0)])
        self.assertEqual(altered.waits, {(0, -1): 1, (0, 0): 4})
        self.assertEqual(sol.waits, {(0, -1): 1, (0, 1): 4})
        with self.assertRaises(ValueError):
            reorder(sol, 0, [0, 0, 2])

    def test_directed_full_cost_decreases_without_changing_end_spot(self):
        ctx = context()
        sol = DaySolution(tours=[[2, 0, 1, 3]])
        altered = shorten_route(ctx, sol, 0)
        self.assertLess(_cost(ctx, 0, altered.tours[0])[0], _cost(ctx, 0, sol.tours[0])[0])
        self.assertLessEqual(_cost(ctx, 0, altered.tours[0])[1], _cost(ctx, 0, sol.tours[0])[1])
        self.assertEqual(altered.tours[0][-1], sol.tours[0][-1])
        self.assertEqual(sorted(altered.tours[0]), sorted(sol.tours[0]))

    def test_saved_steps_fill_an_extra_spot_with_valid_fuel(self):
        ctx = context()
        sol = DaySolution(tours=[[2, 0, 1, 3]])
        before = decode(ctx, sol)
        _, after = shortcut_and_fill(ctx, sol, before)
        self.assertGreater(after.result.servings, before.result.servings)
        self.assertEqual(after.result.servings, 5)
        self.assertEqual(after.result.brands, before.result.brands)
        self.assertTrue(run_day(ctx.config, ctx.steps, ctx.agents, after.plan,
                                ctx.stocks, strict_fuel=True).ok)

    def test_deadline_and_completed_stock_do_not_touch_incumbent(self):
        ctx = context()
        sol = DaySolution(tours=[[2, 0, 1, 3]])
        before = decode(ctx, sol)
        self.assertIs(shortcut_and_fill(ctx, sol, before, deadline=0)[1], before)
        complete_sol = DaySolution(tours=[[0, 1, 2, 3, 4]])
        complete = decode(ctx, complete_sol)
        self.assertIs(shortcut_and_fill(ctx, complete_sol, complete)[1], complete)

    def test_no_submission_regression_in_q_despite_higher_heuristic(self):
        ctx = context()
        better = decode(ctx, DaySolution(tours=[[0, 1, 2, 3, 4]]))
        worse = decode(ctx, DaySolution(tours=[[0]]))
        worse.score = better.score + 1000000
        strat = MatchStrategist(ctx.config)
        strat.visited_brands = {0}
        sent = []

        def fake(ctx, on_plan=None, **kwargs):
            on_plan(better)
            on_plan(worse)
            return worse

        with patch("hexudon.planner_contest.strategy.plan_day", fake):
            applied = strat.plan_day(DayState(0, ctx.agents, {}), record=False,
                                     submit=lambda plan: sent.append(plan) or True)
        self.assertIs(applied, better)
        self.assertEqual(sent, [better.plan])
