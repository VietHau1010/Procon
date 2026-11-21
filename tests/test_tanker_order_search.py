"""Appointment order must improve real collections, not just meeting counts."""

import copy
import random
import unittest
from unittest.mock import patch

from hexudon.model import AgentState, DayState, MatchConfig, Spot
from hexudon.planner.solution import DaySolution, Meet
from hexudon.planner_contest import decoder
from hexudon.planner_contest.engine import make_context, safe_solution
from hexudon.planner_contest.tanker import ordered_meetings
from hexudon.offline.mapgen import generate
from hexudon.simulator import run_day


def counterexample(enabled=False):
    cells = [3]*64
    for c in range(8):
        cells[c] = 0
    cells[7], cells[8] = 1, 2
    cfg = MatchConfig(width=8, height=8, cells=cells,
                      spots=[Spot(0, 5, 2), Spot(1, 6, 2), Spot(2, 2, 2),
                             Spot(3, 3, 2), Spot(4, 4, 2)],
                      agent_starts=[4, 0, 1], fuel_limits=20,
                      day_steps=[16]*4, day_seconds=[5]*4,
                      players=4, busy_threshold=1, jammed_threshold=3)
    agents = [AgentState(0, 4, 1), AgentState(0, 0, 2), AgentState(1, 1, 0)]
    ctx = make_context(cfg, 1, agents, {}, [2]*5, set(), tanker_order_search=enabled)
    sol = DaySolution(tours=[[0, 1], [2, 3, 4]],
                      meets=[Meet(0, 0, 0), Meet(1, 0, 0)], waits={(0, 0): 8, (1, 0): 2})
    return ctx, sol


class TankerOrderSearchTest(unittest.TestCase):
    def test_counterexample_recovers_missed_type_with_one_tanker(self):
        ctx, sol = counterexample()
        original = copy.deepcopy(sol)
        before = decoder.decode(ctx, sol)
        ctx.tanker_order_search = True
        after = decoder.decode(ctx, sol)
        self.assertEqual((len(before.result.brands), before.result.servings), (4, 4))
        self.assertEqual((len(after.result.brands), after.result.servings), (5, 6))
        self.assertEqual(ctx.n_cars, 1)
        self.assertEqual(sol, original)
        self.assertTrue(run_day(ctx.config, ctx.steps, ctx.agents, after.plan,
                                ctx.stocks, ctx.road_status, strict_fuel=True).ok)

    def test_strategist_flag_reaches_the_day_context(self):
        from hexudon.planner_contest.strategy import MatchStrategist
        ctx, sol = counterexample()
        for enabled, expected in ((False, 4), (True, 6)):
            strategist = MatchStrategist(ctx.config, search_iterations=0,
                                          enable_tanker_search=enabled)
            seen = []

            def day(day_ctx, **kwargs):
                seen.append(day_ctx.tanker_order_search)
                return decoder.decode(day_ctx, sol)

            with patch("hexudon.planner_contest.strategy.plan_day", side_effect=day):
                result = strategist.plan_day(DayState(ctx.day, ctx.agents, {}), record=False)
            self.assertEqual(seen, [enabled])
            self.assertEqual(result.result.servings, expected)

    def test_beam_can_find_order_that_earliest_deadline_misses(self):
        ctx, _ = counterexample()
        ctx.agents[ctx.car_agent(0)].pos = 0
        events = [(1, Meet(0, 0, 0), (4, 1, 8)), (1, Meet(1, 0, 0), (1, 1, 9))]
        deadline = ordered_meetings(ctx, 0, events, {}, "deadline")
        beam = ordered_meetings(ctx, 0, events, {}, "beam")
        self.assertEqual(deadline[0][1].patrol, 0)
        self.assertEqual([event[1].patrol for event in beam], [1, 0])
        self.assertEqual({event[1] for event in beam}, {event[1] for event in events})

    def test_starting_refill_waits_for_first_reflection_not_step_zero(self):
        ctx, _ = counterexample()
        event = [(0, Meet(0, -1, 0), (4, 0, 0))]
        for mode in ("deadline", "beam"):
            ordered = ordered_meetings(ctx, 0, event, {(0, -1): 1}, mode)
            self.assertEqual(ordered[0][2], (4, 1, 1))

    def test_default_and_no_dropped_meetings_do_not_run_alternatives(self):
        ctx, sol = counterexample()
        baseline = decoder.decode(ctx, sol)
        with patch.object(decoder, "_decode", return_value=baseline) as decode:
            self.assertIs(decoder.decode(ctx, sol), baseline)
            decode.assert_called_once()
        ctx.tanker_order_search = True
        baseline.dropped_meets = 0
        with patch.object(decoder, "_decode", return_value=baseline) as decode:
            self.assertIs(decoder.decode(ctx, sol), baseline)
            decode.assert_called_once()

    def test_higher_heuristic_or_more_meetings_does_not_replace_equal_ldq(self):
        ctx, sol = counterexample(enabled=True)
        ctx.tanker_order_search = False
        baseline = decoder.decode(ctx, sol)
        ctx.tanker_order_search = True
        candidate = copy.deepcopy(baseline)
        candidate.dropped_meets = 0
        candidate.score += 1000000
        with patch.object(decoder, "_decode", side_effect=[baseline, candidate, candidate]):
            self.assertIs(decoder.decode(ctx, sol), baseline)

    def test_small_random_states_are_valid_and_today_never_regresses(self):
        for seed in range(16):
            cfg = generate(seed=seed, size=8, agents=3, days=4)
            rng = random.Random(seed)
            agents = [AgentState(k, p, rng.randrange(cfg.fuel_limits+1) if k == 0 else 0)
                      for k, p in zip([0, 1, 0], cfg.agent_starts)]
            ctx = make_context(cfg, 0, agents, {}, [s.stocks for s in cfg.spots], set())
            sol, before = safe_solution(ctx, seed)
            original = copy.deepcopy(sol)
            ctx.tanker_order_search = True
            after = decoder.decode(ctx, sol)
            self.assertTrue(after.result.ok)
            self.assertTrue(before.result.brands <= after.result.brands)
            self.assertGreaterEqual((len(after.result.brands), after.result.servings),
                                    (len(before.result.brands), before.result.servings))
            self.assertEqual(sol, original)
            self.assertTrue(run_day(cfg, ctx.steps, agents, after.plan, ctx.stocks,
                                    strict_fuel=True).ok)

    def test_order_cache_returns_fresh_lists_and_is_bounded(self):
        ctx, _ = counterexample()
        events = [(1, Meet(0, 0, 0), (4, 1, 8)), (1, Meet(1, 0, 0), (1, 1, 9))]
        first = ordered_meetings(ctx, 0, events, {}, "beam")
        first.clear()
        self.assertEqual(len(ordered_meetings(ctx, 0, events, {}, "beam")), 2)
        self.assertEqual(len(ctx._tanker_order_cache), 1)
        for index in range(130):
            changed = [(1, Meet(0, 0, 0), (4, 1, 8+index)), events[1]]
            ordered_meetings(ctx, 0, changed, {}, "beam")
        self.assertLessEqual(len(ctx._tanker_order_cache), 128)
