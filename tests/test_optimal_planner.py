"""Kiểm thử bộ lập lịch tối ưu OptimalContestPlanner."""

import unittest
from hexudon import rules
from hexudon.model import AgentState, MatchConfig, Spot
from hexudon.planner_contest.engine import make_context, safe_solution
from hexudon.planner_contest.optimal import (
    OptimalContestPlanner,
    TARGET_CELLS,
    SPARSE_CLUSTER,
    TOP_LEFT_CORNER,
    CORE_CLUSTER,
    STRATEGIC_HUBS,
)
from hexudon.simulator import run_day


class TestOptimalContestPlanner(unittest.TestCase):
    def setUp(self):
        # Tạo map 30x30 với 30 targets
        width, height = 30, 30
        cells = [rules.PLAIN] * (width * height)
        # 30 target cells
        spots = [Spot(brand=i % 15, pos=pos, stocks=4) for i, pos in enumerate(TARGET_CELLS)]
        agent_starts = [83, 338, 389, 93, 250]
        self.config = MatchConfig(
            width=width,
            height=height,
            cells=cells,
            spots=spots,
            agent_starts=agent_starts,
            fuel_limits=288,
            day_steps=[149, 85, 239, 74, 150, 202],
            day_seconds=[5] * 6,
            players=1,
            busy_threshold=1,
            jammed_threshold=2,
        )

    def test_precompute_dijkstra(self):
        planner = OptimalContestPlanner(self.config)
        start_cells = self.config.agent_starts
        road_status = {}
        dist_steps, dist_fuel, paths = planner.precompute_all_pairs(start_cells, road_status)

        # Kiểm tra O(1) tra cứu
        for t in TARGET_CELLS:
            self.assertIn((start_cells[0], t), dist_steps)
            self.assertIn((start_cells[0], t), paths)
            self.assertGreater(len(paths[(start_cells[0], t)]), 0)

    def test_day_tiering_short_day(self):
        planner = OptimalContestPlanner(self.config)
        agents = [
            AgentState(rules.PATROL, self.config.agent_starts[0], 288),
            AgentState(rules.REFUEL, self.config.agent_starts[1], 288),
            AgentState(rules.PATROL, self.config.agent_starts[2], 288),
            AgentState(rules.PATROL, self.config.agent_starts[3], 288),
            AgentState(rules.PATROL, self.config.agent_starts[4], 288),
        ]
        # Ngày ngắn: 74 bước (như Day 3)
        routes, plan = planner.plan_day(agents, 74, {})
        res = run_day(self.config, 74, agents, plan, [4] * 30, {})
        self.assertTrue(res.ok)

        # Kiểm tra không có xe nào đi vào SPARSE_CLUSTER trong ngày ngắn
        for aid in [0, 2, 3, 4]:
            r_set = set(routes.get(aid, []))
            self.assertEqual(len(r_set.intersection(SPARSE_CLUSTER)), 0)

    def test_day_tiering_long_day(self):
        planner = OptimalContestPlanner(self.config)
        agents = [
            AgentState(rules.PATROL, self.config.agent_starts[0], 288),
            AgentState(rules.REFUEL, self.config.agent_starts[1], 288),
            AgentState(rules.PATROL, self.config.agent_starts[2], 288),
            AgentState(rules.PATROL, self.config.agent_starts[3], 288),
            AgentState(rules.PATROL, self.config.agent_starts[4], 288),
        ]
        # Ngày dài: 239 bước (như Day 2)
        routes, plan = planner.plan_day(agents, 239, {})
        res = run_day(self.config, 239, agents, plan, [4] * 30, {})
        self.assertTrue(res.ok)
        # Trong ngày dài, refuel car cắm chốt tại 475
        self.assertEqual(routes[1], [475])

    def test_safe_solution_integration(self):
        agents = [
            AgentState(rules.PATROL, self.config.agent_starts[0], 288),
            AgentState(rules.REFUEL, self.config.agent_starts[1], 288),
            AgentState(rules.PATROL, self.config.agent_starts[2], 288),
            AgentState(rules.PATROL, self.config.agent_starts[3], 288),
            AgentState(rules.PATROL, self.config.agent_starts[4], 288),
        ]
        ctx = make_context(self.config, 0, agents, {}, [4] * 30, set())
        sol, decoded = safe_solution(ctx)
        self.assertTrue(decoded.result.ok)
        self.assertGreater(decoded.result.servings, 0)
        self.assertGreater(len(decoded.result.brands), 0)


if __name__ == "__main__":
    unittest.main()
