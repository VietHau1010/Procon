"""Regression cases for scoring, role evaluation and optimistic bounds."""

import unittest
from unittest.mock import patch

from hexudon import rules
from hexudon.model import AgentState, MatchConfig, Spot
from hexudon.planner_contest.bounds import day_bounds
from hexudon.planner_contest.construct import InsertionState
from hexudon.planner_contest.construct import trim_excess_stock
from hexudon.planner.solution import DaySolution, Meet
from hexudon.planner_contest.engine import make_context
from hexudon.planner_contest.objective import end_value, score_day
from hexudon.planner_contest.polish import fill_idle_suffixes
from hexudon.planner_contest.roles import select_roles, _rollout_contest
from hexudon.simulator import DayResult


def config():
    return MatchConfig(width=8, height=8, cells=[rules.PLAIN]*64,
                       spots=[Spot(0, 34, 1), Spot(1, 35, 2)],
                       agent_starts=[34, 34, 35], fuel_limits=100,
                       day_steps=[4, 4], day_seconds=[5, 5], players=1,
                       busy_threshold=1, jammed_threshold=3)


class AdaptiveContestTest(unittest.TestCase):
    def test_regret_repair_builds_a_valid_productive_plan(self):
        import random
        from hexudon.planner_contest.construct import repair
        from hexudon.planner_contest.decoder import decode

        cfg = config()
        ctx = make_context(cfg, 0, [AgentState(0, p, 100) for p in cfg.agent_starts],
                           {}, [1, 2], set())
        solution = repair(ctx, ctx.empty_solution(), random.Random(0), mode="regret")
        self.assertTrue(any(solution.tours))
        decoded = decode(ctx, solution)
        self.assertTrue(decoded.result.ok)
        self.assertEqual(decoded.result.brands, {0, 1})

    def test_role_refinement_with_regret_completes_match(self):
        # Force the repair operator that crashed during online role selection.
        from hexudon.planner_contest.alns import Roulette
        original = Roulette.pick

        def pick(roulette, rng):
            return "regret" if "regret" in roulette.names else original(roulette, rng)

        with patch.object(Roulette, "pick", pick):
            score, days = _rollout_contest(config(), [0, 1, 0], None, 0, None, 8)
        self.assertEqual(days, 2)
        self.assertEqual(score[:2], (2, 4))

    def test_rollout_replenishes_daily_stock(self):
        cfg = config()
        cfg.day_steps = [1, 1]
        score, days = _rollout_contest(cfg, [0, 0, 0], None, 0, None, 0)
        self.assertEqual(days, 2)
        self.assertEqual(score, (2, 4, 4))

    def test_exhausted_spot_has_no_marginal_gain(self):
        cfg = config()
        ctx = make_context(cfg, 0, [AgentState(0, p, 100) for p in cfg.agent_starts],
                           {}, [0, 2], set())
        self.assertEqual(InsertionState(ctx).gain_of(0), 0)

    def test_duplicate_parking_respects_stock_and_brand(self):
        cfg = config()
        ctx = make_context(cfg, 0, [AgentState(0, p, 100) for p in cfg.agent_starts],
                           {}, [1, 2], {0, 1})
        ctx.weights.near_chain = 0
        result = DayResult(ok=True, end_pos=[34, 34, 35], end_fuel=[100]*3, brands={0, 1})
        # Two brands, two servings and a small full-fuel bonus. No cell-ID penalty.
        self.assertAlmostEqual(end_value(ctx, result), 1050)

    def test_daily_bound_limits_each_spot_by_stock(self):
        cfg = config()
        ctx = make_context(cfg, 0, [AgentState(0, p, 100) for p in cfg.agent_starts],
                           {}, [1, 2], set())
        self.assertEqual(day_bounds(ctx)["servings"], 3)
        self.assertEqual(day_bounds(ctx)["types"], 2)

    def test_productive_parking_scores_better_in_this_case(self):
        cfg = config()
        agents = [AgentState(0, p, 100) for p in cfg.agent_starts]
        ctx = make_context(cfg, 0, agents, {}, [1, 2], set())
        no_q = DayResult(ok=True, end_pos=[0, 1, 2], end_fuel=[0, 0, 0], brands={0})
        one_q = DayResult(ok=True, end_pos=[34, 35, 36], end_fuel=[100, 100, 100],
                          brands={0}, collections=[(1, 0, 0)])
        self.assertGreater(score_day(ctx, one_q), score_day(ctx, no_q))

    def test_trim_preserves_refuel_and_wait_anchors(self):
        cfg = config()
        ctx = make_context(cfg, 0, [AgentState(0, p, 100) for p in cfg.agent_starts],
                           {}, [1, 2], set())
        sol = DaySolution(tours=[[0], [0], [0]],
                          meets=[Meet(0, 0, 0)], waits={(1, 0): 2})
        trimmed = trim_excess_stock(ctx, sol)
        self.assertEqual(trimmed.tours, [[0], [0], []])
        self.assertEqual(trimmed.meets, sol.meets)
        self.assertEqual(trimmed.waits, sol.waits)
        self.assertEqual(sol.tours, [[0], [0], [0]])

    def test_polish_returns_to_preserve_tomorrows_brand_coverage(self):
        from hexudon.simulator import run_day
        from hexudon.planner_contest.decoder import Decoded
        cfg = MatchConfig(width=4, height=1, cells=[0]*4,
                          spots=[Spot(0, 0, 1), Spot(1, 1, 2)],
                          agent_starts=[0, 1, 0], fuel_limits=20,
                          day_steps=[6, 6], day_seconds=[5, 5], players=1,
                          busy_threshold=1, jammed_threshold=3)
        agents = [AgentState(0, 0, 20), AgentState(0, 1, 0), AgentState(1, 0, 0)]
        ctx = make_context(cfg, 0, agents, {}, [1, 2], {0, 1})
        plan = [[-6], [-6], [-6]]
        result = run_day(cfg, 6, agents, plan, ctx.stocks)
        polished = fill_idle_suffixes(ctx, Decoded(plan, result, score_day(ctx, result)))
        self.assertEqual(polished.result.servings, 3)
        self.assertEqual(polished.result.end_pos, [0, 1, 0])
        self.assertGreaterEqual(polished.result.end_fuel[0], result.end_fuel[0])
        self.assertEqual(polished.result.brands, {0, 1})
        self.assertEqual(polished.plan[0][0], -1)
        self.assertTrue(run_day(cfg, 6, agents, polished.plan, ctx.stocks,
                                strict_fuel=True).ok)

    def test_polish_expired_deadline_returns_incumbent(self):
        from hexudon.planner_contest.engine import safe_solution
        cfg = config()
        ctx = make_context(cfg, 0, [AgentState(0, p, 100) for p in cfg.agent_starts],
                           {}, [1, 2], set())
        _, decoded = safe_solution(ctx)
        self.assertIs(fill_idle_suffixes(ctx, decoded, deadline=0), decoded)

    def test_excursion_without_refill_does_not_spend_tomorrows_fuel(self):
        from hexudon.simulator import run_day
        from hexudon.planner_contest.decoder import Decoded
        cfg = MatchConfig(width=4, height=1, cells=[0]*4,
                          spots=[Spot(0, 0, 1), Spot(1, 1, 2)],
                          agent_starts=[0, 1], fuel_limits=20,
                          day_steps=[6, 6], day_seconds=[5, 5], players=1,
                          busy_threshold=1, jammed_threshold=3)
        agents = [AgentState(0, 0, 20), AgentState(0, 1, 0)]
        ctx = make_context(cfg, 0, agents, {}, [1, 2], {0, 1})
        plan = [[-6], [-6]]
        result = run_day(cfg, 6, agents, plan, ctx.stocks)
        incumbent = Decoded(plan, result, score_day(ctx, result))
        self.assertIs(fill_idle_suffixes(ctx, incumbent), incumbent)

    def test_no_spots_and_no_patrols_are_valid(self):
        from hexudon.planner_contest.engine import safe_solution
        cfg = config()
        cfg.spots = []
        cfg.__post_init__()
        ctx = make_context(cfg, 0, [AgentState(1, p, 100) for p in cfg.agent_starts],
                           {}, [], set())
        _, decoded = safe_solution(ctx)
        self.assertTrue(decoded.result.ok)
        self.assertEqual(decoded.result.servings, 0)

    def test_short_refinement_cannot_replace_complete_rollout(self):
        cfg = config()
        from hexudon.planner.roles import RoleCandidate
        first, second = [0, 1, 0], [0, 0, 0]
        opts = [RoleCandidate(first, "base"), RoleCandidate(second, "other")]
        with patch("hexudon.planner_contest.roles.choose_roles_for_match", return_value=first), \
             patch("hexudon.planner_contest.roles.candidates", return_value=opts), \
             patch("hexudon.planner_contest.roles._rollout_contest",
                   side_effect=[((2, 4, 9), 2), ((2, 4, 8), 2),
                                ((1, 1, 1), 1), ((2, 4, 10), 2)]):
            chosen, _ = select_roles(cfg, budget_seconds=10)
        self.assertEqual(chosen, first)

    def test_polisher_uses_feasible_idle_suffix_and_keeps_it_valid(self):
        cfg = config()
        agents = [AgentState(0, 34, 100), AgentState(0, 34, 100),
                  AgentState(0, 35, 100)]
        ctx = make_context(cfg, 0, agents, {}, [1, 2], set())
        plan = [[-4], [-4], [-4]]
        from hexudon.simulator import run_day
        from hexudon.planner_contest.decoder import Decoded
        result = run_day(cfg, 4, agents, plan, ctx.stocks, {}, strict_fuel=True)
        initial = Decoded(plan, result, 0)
        polished = fill_idle_suffixes(ctx, initial)
        self.assertTrue(polished.result.ok)
        self.assertGreater(polished.result.servings, result.servings)

    def test_incomplete_role_rollout_is_discarded(self):
        cfg = config()
        fallback = [0, 1, 0]
        from hexudon.planner.roles import RoleCandidate
        opts = [RoleCandidate(fallback, "base"), RoleCandidate([0, 0, 0], "other")]
        with patch("hexudon.planner_contest.roles.choose_roles_for_match", return_value=fallback), \
             patch("hexudon.planner_contest.roles.candidates", return_value=opts), \
             patch("hexudon.planner_contest.roles._rollout_contest",
                   side_effect=[((2, 4, 4), 2), ((99, 99, 99), 1), ((2, 4, 4), 2)]):
            roles, _ = select_roles(cfg, iterations_per_day=0)
        self.assertEqual(roles, fallback)
