"""Kiểm thử toàn diện gói chiến thuật thi đấu mới hexudon/planner_contest."""

import unittest
from hexudon import rules
from hexudon.model import AgentState, MatchConfig, Spot
from hexudon.planner_contest import (
    choose_roles_for_match,
    construct,
    decode,
    default_contest_weights,
    end_value,
    greedy_fill,
    improve,
    make_context,
    plan_day,
    safe_solution,
    score_day,
)
from hexudon.planner_contest.alns import destroy_chain_dup_safe
from hexudon.planner.solution import DaySolution, Meet
from hexudon.simulator import run_day


class PlannerContestTest(unittest.TestCase):
    def setUp(self):
        width, height = 20, 20
        cells = [rules.PLAIN] * (width * height)
        spot_cells = [
            21, 25, 43, 67, 89, 105, 120, 142, 165, 188,
            210, 225, 248, 270, 292, 315, 338, 355, 372, 390
        ]
        spots = [Spot(brand=i % 10, pos=pos, stocks=4) for i, pos in enumerate(spot_cells)]
        agent_starts = [20, 22, 100, 102]
        self.config = MatchConfig(
            width=width,
            height=height,
            cells=cells,
            spots=spots,
            agent_starts=agent_starts,
            fuel_limits=200,
            day_steps=[100, 150, 200],
            day_seconds=[5, 5, 5],
            players=1,
            busy_threshold=1,
            jammed_threshold=2,
        )

    def test_choose_roles(self):
        roles_4 = choose_roles_for_match(4, 200, 20, 20)
        self.assertEqual(roles_4, [rules.PATROL, rules.PATROL, rules.PATROL, rules.REFUEL])
        roles_5 = choose_roles_for_match(5, 200, 20, 20)
        self.assertEqual(roles_5, [rules.PATROL, rules.PATROL, rules.PATROL, rules.PATROL, rules.REFUEL])

    def test_contest_roles_always_use_one_refuel_agent(self):
        config = self.config
        config.agent_starts = list(range(7))
        config.__post_init__()
        roles = choose_roles_for_match(config)
        self.assertEqual(roles.count(rules.REFUEL), 1)

    def test_role_rollout_candidates_all_have_one_refuel_agent(self):
        from hexudon.planner_contest.roles import select_roles
        from hexudon.offline.mapgen import generate

        config = generate(seed=17, size=30, agents=8, days=2)
        config.fuel_limits = 90
        config.__post_init__()
        selected, candidates = select_roles(config, budget_seconds=0, iterations_per_day=1)
        self.assertEqual(selected.count(rules.REFUEL), 1)
        self.assertTrue(candidates)
        self.assertTrue(all(c.kinds.count(rules.REFUEL) == 1 for c in candidates))

    def test_role_selector_discards_zero_and_multiple_refuel_candidates(self):
        from unittest.mock import patch
        from hexudon.offline.mapgen import generate
        from hexudon.planner.roles import RoleCandidate
        from hexudon.planner_contest.roles import select_roles

        cfg = generate(seed=23, size=8, agents=5, days=2)
        one = [0, 0, 0, 0, 1]
        opts = [RoleCandidate([0]*5, "zero"), RoleCandidate(one, "one"),
                RoleCandidate([0, 0, 1, 1, 1], "three")]
        with patch("hexudon.planner_contest.roles.choose_roles_for_match", return_value=one), \
             patch("hexudon.planner_contest.roles.candidates", return_value=opts), \
             patch("hexudon.planner_contest.roles._rollout_contest",
                   return_value=((1, 2, 3), cfg.n_days)):
            chosen, evaluated = select_roles(cfg, iterations_per_day=1)
        self.assertEqual(chosen.count(rules.REFUEL), 1)
        self.assertTrue(all(c.kinds.count(rules.REFUEL) == 1 for c in evaluated))

    def test_default_weights(self):
        w = default_contest_weights()
        self.assertEqual(w.serving, 20.0)
        self.assertEqual(w.fuel_bonus, 0.01)
        self.assertEqual(w.strand, 2.0)
        self.assertEqual(w.end_on_spot, 0.5)

    def test_construct_and_decode(self):
        agents = [
            AgentState(rules.PATROL, self.config.agent_starts[0], 200),
            AgentState(rules.PATROL, self.config.agent_starts[1], 200),
            AgentState(rules.PATROL, self.config.agent_starts[2], 200),
            AgentState(rules.REFUEL, self.config.agent_starts[3], 200),
        ]
        ctx = make_context(self.config, 0, agents, {}, [4] * 20, set())
        sol, decoded = safe_solution(ctx)
        self.assertTrue(decoded.result.ok)
        self.assertGreater(decoded.result.servings, 0)
        self.assertGreater(len(decoded.result.brands), 0)
        self.assertEqual(len(decoded.plan), 4)

    def test_destroy_chain_dup_safe(self):
        import random
        agents = [
            AgentState(rules.PATROL, self.config.agent_starts[0], 200),
            AgentState(rules.REFUEL, self.config.agent_starts[1], 200),
        ]
        # Spot 0 has stock 1, but both patrols visit spot 0 (counts = 2 > stock = 1)
        ctx = make_context(self.config, 0, agents, {}, [1] + [4] * 19, set())
        sol = DaySolution(tours=[[0], [0]], meets=[], waits={})
        rng = random.Random(42)
        sol_after = destroy_chain_dup_safe(ctx, sol, rng, q=1)
        # Exactly one excess visit is removed, leaving 1 visit
        total_remaining = sum(len(t) for t in sol_after.tours)
        self.assertEqual(total_remaining, 1)

    def test_plan_day_alns(self):
        agents = [
            AgentState(rules.PATROL, self.config.agent_starts[0], 200),
            AgentState(rules.PATROL, self.config.agent_starts[1], 200),
            AgentState(rules.PATROL, self.config.agent_starts[2], 200),
            AgentState(rules.REFUEL, self.config.agent_starts[3], 200),
        ]
        ctx = make_context(self.config, 0, agents, {}, [4] * 20, set())
        dec = plan_day(ctx, max_iterations=50, seed=1)
        self.assertTrue(dec.result.ok)
        self.assertGreater(dec.result.servings, 0)
        self.assertEqual(len(dec.plan), 4)


if __name__ == "__main__":
    unittest.main()
