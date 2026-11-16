import sys
sys.path.insert(0, '.')
import json
import heapq
import time
from typing import List, Dict, Tuple, Set, Optional, Sequence

from hexudon import rules
from hexudon.model import MatchConfig, AgentState, Spot
from hexudon.simulator import run_day, DayResult
from hexudon.planner.decoder import Decoded, compact_plan

TARGET_CELLS = [
    5, 34, 40, 142, 197, 233, 308, 338, 475, 477,
    481, 487, 515, 573, 579, 589, 614, 638, 679, 706,
    708, 722, 757, 770, 776, 813, 831, 842, 845, 876
]

SPARSE_CLUSTER = {142, 197, 233}
TOP_LEFT_CORNER = {5, 34, 40}
CORE_CLUSTER = set(TARGET_CELLS) - SPARSE_CLUSTER - TOP_LEFT_CORNER
STRATEGIC_HUBS = [475, 579, 487, 831, 308]


class OptimalContestPlanner:
    def __init__(
        self,
        config: MatchConfig,
        cell_types: Optional[Dict[int, int]] = None,
        targets: Optional[List[int]] = None,
        strategic_hubs: Optional[List[int]] = None,
    ):
        self.config = config
        self.adj = config.neighbors
        self.max_fuel = config.fuel_limits
        
        # Determine targets
        if targets is not None:
            self.targets = list(targets)
        elif all(s.pos in set(TARGET_CELLS) for s in config.spots):
            self.targets = list(TARGET_CELLS)
        else:
            self.targets = [s.pos for s in config.spots]
            
        # Determine cell types
        if cell_types is not None:
            self.cell_types = dict(cell_types)
        else:
            self.cell_types = {s.pos: s.brand for s in config.spots}
            
        self.strategic_hubs = strategic_hubs or [h for h in STRATEGIC_HUBS if h in self.targets]
        if not self.strategic_hubs:
            self.strategic_hubs = [self.targets[0]]

    def get_step_and_fuel_cost(self, cell: int, road_condition: Dict[int, int]) -> Tuple[int, int]:
        terrain = self.config.cells[cell]
        if terrain == rules.ROAD:
            step_cost = rules.STEP_ROAD[road_condition.get(cell, rules.SMOOTH)]
            fuel_cost = rules.FUEL_ROAD
        else:
            step_cost = rules.step_cost(terrain)
            fuel_cost = rules.fuel_cost(terrain)
        return step_cost, fuel_cost

    def precompute_all_pairs(
        self, start_cells: List[int], road_condition: Dict[int, int]
    ) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], Dict[Tuple[int, int], List[int]]]:
        """
        TRỤ CỘT 1: Tiền xử lý Dijkstra từ 30 ô target + vị trí xuất phát của các xe.
        O(1) lookup cho steps, fuel, và shortest paths.
        """
        important_nodes = set(self.targets) | set(start_cells)
        dist_steps = {}
        dist_fuel = {}
        paths = {}

        for src in important_nodes:
            pq = [(0, 0, src)]
            best = {src: (0, 0)}
            parent = {src: None}

            while pq:
                s_cost, f_cost, u = heapq.heappop(pq)
                if (s_cost, f_cost) > best[u]:
                    continue

                # Iterate through valid neighbors
                for d_idx, v in enumerate(self.adj[u]):
                    if v < 0 or self.config.cells[v] == rules.POND:
                        continue
                    ds, df = self.get_step_and_fuel_cost(u, road_condition)
                    ns, nf = s_cost + ds, f_cost + df
                    if v not in best or (ns, nf) < best[v]:
                        best[v] = (ns, nf)
                        parent[v] = u
                        heapq.heappush(pq, (ns, nf, v))

            for dst in important_nodes:
                if dst in best:
                    dist_steps[(src, dst)] = best[dst][0]
                    dist_fuel[(src, dst)] = best[dst][1]
                    cur = dst
                    path = []
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    paths[(src, dst)] = path[::-1]
                else:
                    dist_steps[(src, dst)] = float("inf")
                    dist_fuel[(src, dst)] = float("inf")
                    paths[(src, dst)] = []

        return dist_steps, dist_fuel, paths

    def evaluate_route(
        self,
        route: List[int],
        start_cell: int,
        start_fuel: int,
        max_steps: int,
        refuel_schedule: Dict[int, Tuple[int, int]],
        dist_steps: Dict[Tuple[int, int], int],
        dist_fuel: Dict[Tuple[int, int], int],
    ) -> Tuple[bool, int, int, int]:
        cur_cell = start_cell
        cur_fuel = start_fuel
        cur_step = 0
        collected = 0

        idx = 0
        if route and route[0] == start_cell:
            cur_step += 1
            collected += 1
            if start_cell in refuel_schedule:
                t_min, t_max = refuel_schedule[start_cell]
                if t_min <= cur_step <= t_max:
                    cur_fuel = self.max_fuel
            idx = 1

        for i in range(idx, len(route)):
            nxt = route[i]
            ds = dist_steps.get((cur_cell, nxt), float("inf"))
            df = dist_fuel.get((cur_cell, nxt), float("inf"))

            if cur_fuel < df or cur_step + ds > max_steps:
                return False, cur_step, cur_fuel, collected

            cur_step += ds
            cur_fuel -= df
            cur_cell = nxt
            collected += 1

            if cur_cell in refuel_schedule:
                t_min, t_max = refuel_schedule[cur_cell]
                if t_min <= cur_step <= t_max:
                    cur_fuel = self.max_fuel

        return True, cur_step, cur_fuel, collected

    def optimize_single_agent_route(
        self,
        agent_id: int,
        start_cell: int,
        start_fuel: int,
        max_steps: int,
        allowed_targets: Set[int],
        mandatory_targets: Set[int],
        refuel_schedule: Dict[int, Tuple[int, int]],
        dist_steps: Dict[Tuple[int, int], int],
        dist_fuel: Dict[Tuple[int, int], int],
        allow_top_left_end: bool,
    ) -> List[int]:
        """
        TRỤ CỘT 2 & 4: Khởi tạo tham lam tỷ suất (Score/Step), 2-Opt local search,
        và Greedy Insertion vét cạn bước thừa.
        """
        route = []
        visited = set()
        cur_cell = start_cell
        cur_fuel = start_fuel
        cur_step = 0

        if start_cell in allowed_targets:
            route.append(start_cell)
            visited.add(start_cell)
            cur_step += 1

        # 1. Khởi tạo tham lam
        while True:
            best_nxt = None
            best_score = -1e9

            for cand in allowed_targets:
                if cand in visited:
                    continue
                ds = dist_steps.get((cur_cell, cand), float("inf"))
                df = dist_fuel.get((cur_cell, cand), float("inf"))
                if cur_step + ds > max_steps or cur_fuel < df:
                    continue

                score = -ds
                if cand in mandatory_targets:
                    score += 50.0
                if cand in refuel_schedule and cur_fuel - df < self.max_fuel // 2:
                    t_min, t_max = refuel_schedule[cand]
                    if t_min <= cur_step + ds <= t_max:
                        score += 35.0

                remaining_steps = max_steps - (cur_step + ds)
                if remaining_steps < 18 and not allow_top_left_end:
                    if cand in TOP_LEFT_CORNER or cand in SPARSE_CLUSTER:
                        score -= 30.0
                    elif cand in CORE_CLUSTER:
                        score += 10.0

                if score > best_score:
                    best_score = score
                    best_nxt = cand

            if best_nxt is None:
                break

            ds = dist_steps[(cur_cell, best_nxt)]
            df = dist_fuel[(cur_cell, best_nxt)]
            cur_step += ds
            cur_fuel -= df
            cur_cell = best_nxt
            route.append(best_nxt)
            visited.add(best_nxt)

            if cur_cell in refuel_schedule:
                t_min, t_max = refuel_schedule[cur_cell]
                if t_min <= cur_step <= t_max:
                    cur_fuel = self.max_fuel

        # 2. 2-Opt Local Search
        improved = True
        while improved:
            improved = False
            _, best_total_steps, _, _ = self.evaluate_route(
                route, start_cell, start_fuel, max_steps, refuel_schedule, dist_steps, dist_fuel
            )
            start_idx = 1 if (route and route[0] == start_cell) else 0
            for i in range(start_idx, len(route) - 2):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    ok, new_steps, _, _ = self.evaluate_route(
                        new_route, start_cell, start_fuel, max_steps, refuel_schedule, dist_steps, dist_fuel
                    )
                    if ok and new_steps < best_total_steps:
                        route = new_route
                        best_total_steps = new_steps
                        improved = True
                        break
                if improved:
                    break

        # 3. Greedy Insertion (vét cạn bước thừa)
        unvisited = list(allowed_targets - set(route))
        inserted = True
        while inserted and unvisited:
            inserted = False
            best_candidate_route = None
            best_steps = 1e9

            start_idx = 1 if (route and route[0] == start_cell) else 0
            for cand in unvisited:
                for pos in range(start_idx, len(route) + 1):
                    if pos == len(route) and not allow_top_left_end and cand in TOP_LEFT_CORNER:
                        continue
                    test_route = route[:pos] + [cand] + route[pos:]
                    ok, t_steps, _, _ = self.evaluate_route(
                        test_route, start_cell, start_fuel, max_steps, refuel_schedule, dist_steps, dist_fuel
                    )
                    if ok and t_steps < best_steps:
                        best_steps = t_steps
                        best_candidate_route = (cand, test_route)

            if best_candidate_route is not None:
                cand, route = best_candidate_route
                unvisited.remove(cand)
                inserted = True

        return route

    def plan_day(
        self,
        agents_state: Sequence[AgentState],
        max_steps: int,
        road_condition: Dict[int, int],
    ) -> Tuple[Dict[int, List[int]], List[List[int]]]:
        """
        Điều phối toàn đội trong ngày.
        Trả về: (agent_routes, plan_tokens)
        """
        start_cells = [a.pos for a in agents_state]
        dist_steps, dist_fuel, paths = self.precompute_all_pairs(start_cells, road_condition)

        patrol_ids = [i for i, a in enumerate(agents_state) if a.kind == rules.PATROL]
        refuel_ids = [i for i, a in enumerate(agents_state) if a.kind == rules.REFUEL]

        # TRỤ CỘT 3: Refuel Hub selection
        refuel_cell = agents_state[refuel_ids[0]].pos if refuel_ids else start_cells[0]
        if max_steps >= 170 and 475 in self.targets:
            hub = 475
        else:
            hub = min(self.strategic_hubs, key=lambda h: dist_steps.get((refuel_cell, h), float("inf")))

        hub_arrival = dist_steps.get((refuel_cell, hub), 0)
        refuel_schedule = {hub: (hub_arrival, max_steps)}

        # TRỤ CỘT 2: Phân tầng mục tiêu theo độ dài ngày
        if max_steps <= 155:
            base_targets = set(self.targets) - SPARSE_CLUSTER
        else:
            base_targets = set(self.targets)

        # TRỤ CỘT 4: Anti-clumping ở ngõ cụt 34
        top_left_anchor = 34 if 34 in self.targets else (min(TOP_LEFT_CORNER) if TOP_LEFT_CORNER.intersection(self.targets) else None)
        if top_left_anchor is not None and patrol_ids:
            closest_to_top_left = min(
                patrol_ids, key=lambda aid: dist_steps.get((agents_state[aid].pos, top_left_anchor), float("inf"))
            )
        else:
            closest_to_top_left = patrol_ids[0] if patrol_ids else 0

        collected_types = set()
        agent_routes = {}

        for aid in patrol_ids:
            missing_types = set(self.cell_types.values()) - collected_types
            mandatory_cells = {
                c for c in base_targets if self.cell_types.get(c) in missing_types
            }

            allowed = set(base_targets)
            if max_steps <= 90 and aid != closest_to_top_left:
                allowed -= TOP_LEFT_CORNER

            allow_top_left_end = (aid == closest_to_top_left)

            route = self.optimize_single_agent_route(
                agent_id=aid,
                start_cell=agents_state[aid].pos,
                start_fuel=agents_state[aid].fuel,
                max_steps=max_steps,
                allowed_targets=allowed,
                mandatory_targets=mandatory_cells,
                refuel_schedule=refuel_schedule,
                dist_steps=dist_steps,
                dist_fuel=dist_fuel,
                allow_top_left_end=allow_top_left_end,
            )
            agent_routes[aid] = route
            for c in route:
                if c in self.cell_types:
                    collected_types.add(self.cell_types[c])

        for rid in refuel_ids:
            agent_routes[rid] = [hub]

        # Convert agent_routes to token plan
        plan = self._build_token_plan(agents_state, agent_routes, max_steps, dist_steps, paths, road_condition)
        return agent_routes, plan

    def _build_token_plan(
        self,
        agents_state: Sequence[AgentState],
        agent_routes: Dict[int, List[int]],
        max_steps: int,
        dist_steps: Dict[Tuple[int, int], int],
        paths: Dict[Tuple[int, int], List[int]],
        road_condition: Dict[int, int],
    ) -> List[List[int]]:
        dir_index = self.config.dir_index
        plan = []

        for i, agent in enumerate(agents_state):
            cur = agent.pos
            route = agent_routes.get(i, [])
            tokens = []
            used_steps = 0

            if agent.kind == rules.REFUEL:
                # Car goes to hub, then waits
                hub = route[0] if route else cur
                if cur != hub:
                    path = paths.get((cur, hub), [])
                    for step_idx in range(len(path) - 1):
                        u, v = path[step_idx], path[step_idx + 1]
                        ds, _ = self.get_step_and_fuel_cost(u, road_condition)
                        if used_steps + ds > max_steps:
                            break
                        d = dir_index[u].get(v)
                        if d is not None:
                            tokens.append(d)
                            used_steps += ds
                            cur = v
                remaining = max_steps - used_steps
                if remaining > 0:
                    tokens.append(-remaining)
                plan.append(tokens)
                continue

            # Patrol route
            route_idx = 0
            if route and route[0] == cur:
                route_idx = 1

            for tgt in route[route_idx:]:
                if cur == tgt:
                    continue
                path = paths.get((cur, tgt), [])
                if not path:
                    continue
                for step_idx in range(len(path) - 1):
                    u, v = path[step_idx], path[step_idx + 1]
                    ds, _ = self.get_step_and_fuel_cost(u, road_condition)
                    if used_steps + ds > max_steps:
                        break
                    d = dir_index[u].get(v)
                    if d is not None:
                        tokens.append(d)
                        used_steps += ds
                        cur = v
                if used_steps >= max_steps:
                    break

            remaining = max_steps - used_steps
            if remaining > 0:
                tokens.append(-remaining)
            plan.append(tokens)

        return compact_plan(plan)

print("OptimalContestPlanner class defined successfully.")
# Append test runner
with open("replays/contest-087d9b9a-t78-20261112-165702.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('path') == '/api/game/config':
            cfg_dict = entry['response']
        if entry.get('path') == '/api/game/state' and entry.get('response', {}).get('day') == 0:
            st0_dict = entry['response']
            break

import json
cfg = MatchConfig.from_json(cfg_dict)
st0_road = {int(k): v for k, v in st0_dict.get('road_condition', {}).items()}
agents = [
    AgentState(rules.PATROL, cfg.agent_starts[0], cfg.fuel_limits),
    AgentState(rules.REFUEL, cfg.agent_starts[1], cfg.fuel_limits),
    AgentState(rules.PATROL, cfg.agent_starts[2], cfg.fuel_limits),
    AgentState(rules.PATROL, cfg.agent_starts[3], cfg.fuel_limits),
]

planner = OptimalContestPlanner(cfg)
t0 = time.time()
routes, plan = planner.plan_day(agents, cfg.day_steps[0], st0_road)
elapsed = time.time() - t0

print(f"Plan generated in {elapsed*1000:.2f} ms")
for aid, r in routes.items():
    print(f"  Agent {aid}: {len(r)} targets -> {r[:10]}")

# Simulate in golden simulator
res = run_day(cfg, cfg.day_steps[0], agents, plan, [s.stocks for s in cfg.spots], st0_road)
print(f"Simulation result: ok={res.ok}, brands={len(res.brands)}, servings={res.servings}, end_pos={res.end_pos}, end_fuel={res.end_fuel}")
if not res.ok:
    print(f"Error: {res.error}")
