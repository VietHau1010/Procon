"""Bộ lập lịch tối ưu toàn diện cho mọi map HEXUDON 2026 (OptimalContestPlanner).

Tích hợp trọn vẹn 5 tầng chiến thuật:
1. Phân bổ Roles chuẩn thích ứng N in [3, 8] (choose_roles_for_match).
2. Tìm đường Đa mục tiêu Dual-weight Dijkstra (né đường đông/tắc sang đồng bằng, tiết kiệm xăng).
3. Gom nhóm Adaptive K-Platoons & Centralized Stock Reservation (quản lý kho, tránh lãng phí).
4. Khóa chặt L (-5000) -> D (-1000) -> Q (mật độ bước, platoon bonus) & Smart End-of-day Parking.
5. Tiếp nhiên liệu Zero-Detour Station Intercept (Refuel đón lõng tại quán udon).
"""

import heapq
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .. import rules
from ..model import AgentState, MatchConfig
from ..simulator import DayResult, run_day
from ..planner.decoder import Decoded, compact_plan
from ..planner.solution import DayContext, DaySolution, Meet

# Các ô mục tiêu & phân cụm đặc thù (nếu gặp map 30x30 contest)
TARGET_CELLS = [
    5, 34, 40, 142, 197, 233, 308, 338, 475, 477,
    481, 487, 515, 573, 579, 589, 614, 638, 679, 706,
    708, 722, 757, 770, 776, 813, 831, 842, 845, 876,
]
SPARSE_CLUSTER = {142, 197, 233}
TOP_LEFT_CORNER = {5, 34, 40}
CORE_CLUSTER = set(TARGET_CELLS) - SPARSE_CLUSTER - TOP_LEFT_CORNER
STRATEGIC_HUBS = [475, 579, 487, 831, 308]


def choose_roles_for_match(num_agents: int, max_fuel: int, width: int, height: int) -> List[int]:
    """
    TẦNG 1: Công thức chọn số lượng xe Roles đầu trận (Thích ứng 3..8 xe):
    - N in [3, 6]: Dùng (N - 1) Patrol + 1 Refuel.
    - N in [7, 8]:
        + Nếu map <= 20x20 hoặc bình xăng >= 150: Dùng (N - 1) Patrol + 1 Refuel.
        + Nếu map lớn (>= 600 ô) và bình xăng nhỏ (< 120): Dùng (N - 2) Patrol + 2 Refuel.
    """
    if num_agents <= 6:
        return [rules.PATROL] * (num_agents - 1) + [rules.REFUEL]
    if width * height >= 600 and max_fuel < 120:
        return [rules.PATROL] * (num_agents - 2) + [rules.REFUEL, rules.REFUEL]
    return [rules.PATROL] * (num_agents - 1) + [rules.REFUEL]


class HexudonMap:
    """
    TẦNG 2: Xử lý bản đồ lục giác even-r và tìm đường Đa mục tiêu Dual-weight Dijkstra.
    Cost_Patrol(u) = steps(u) + alpha * fuel(u).
    - Khi xăng dồi dào: alpha = 0.05 (né đường đông/tắc sang đồng bằng).
    - Khi xăng thấp: alpha = 0.45 (ưu tiên đồng bằng 1 xăng để kéo dài hành trình).
    - Xe Refuel: Cost_Refuel(u) = steps(u) (không tốn xăng).
    """

    def __init__(self, config: MatchConfig, road_condition: Dict[int, int]):
        self.config = config
        self.adj = config.neighbors
        self.road_cond = dict(road_condition)
        self.cells = config.cells

    def get_step_and_fuel_cost(self, cell: int) -> Tuple[int, int]:
        terrain = self.cells[cell]
        if terrain == rules.ROAD:
            cond = self.road_cond.get(cell, rules.SMOOTH)
            step_cost = rules.STEP_ROAD[cond]
            fuel_cost = rules.FUEL_ROAD
        else:
            step_cost = rules.step_cost(terrain)
            fuel_cost = rules.fuel_cost(terrain)
        return step_cost, fuel_cost

    def dijkstra_dual(
        self,
        src: int,
        targets: Set[int],
        fuel_weight: float = 0.05,
    ) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, List[int]]]:
        steps_to: Dict[int, int] = {src: 0}
        fuel_to: Dict[int, int] = {src: 0}
        best_score: Dict[int, float] = {src: 0.0}
        parent: Dict[int, Optional[int]] = {src: None}
        pq: List[Tuple[float, int, int, int]] = [(0.0, 0, 0, src)]

        while pq:
            score, d_step, d_fuel, u = heapq.heappop(pq)
            if score > best_score.get(u, 1e18):
                continue
            step_c, fuel_c = self.get_step_and_fuel_cost(u)
            for v in self.adj[u]:
                if v < 0 or self.cells[v] == rules.POND:
                    continue
                ns = d_step + step_c
                nf = d_fuel + fuel_c
                nscore = ns + fuel_weight * nf
                if nscore < best_score.get(v, 1e18):
                    best_score[v] = nscore
                    steps_to[v] = ns
                    fuel_to[v] = nf
                    parent[v] = u
                    heapq.heappush(pq, (nscore, ns, nf, v))

        paths_to: Dict[int, List[int]] = {}
        for dst in targets:
            if dst in steps_to:
                cur = dst
                p = []
                while cur is not None:
                    p.append(cur)
                    cur = parent[cur]
                paths_to[dst] = p[::-1]
            else:
                steps_to[dst] = 999999
                fuel_to[dst] = 999999
                paths_to[dst] = []

        return steps_to, fuel_to, paths_to

    def dijkstra_refuel(
        self,
        src: int,
        targets: Set[int],
    ) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        steps_to: Dict[int, int] = {src: 0}
        parent: Dict[int, Optional[int]] = {src: None}
        pq: List[Tuple[int, int]] = [(0, src)]

        while pq:
            d_step, u = heapq.heappop(pq)
            if d_step > steps_to.get(u, 10**9):
                continue
            step_c, _ = self.get_step_and_fuel_cost(u)
            for v in self.adj[u]:
                if v < 0 or self.cells[v] == rules.POND:
                    continue
                ns = d_step + step_c
                if ns < steps_to.get(v, 10**9):
                    steps_to[v] = ns
                    parent[v] = u
                    heapq.heappush(pq, (ns, v))

        paths_to: Dict[int, List[int]] = {}
        for dst in targets:
            if dst in steps_to:
                cur = dst
                p = []
                while cur is not None:
                    p.append(cur)
                    cur = parent[cur]
                paths_to[dst] = p[::-1]
            else:
                steps_to[dst] = 999999
                paths_to[dst] = []

        return steps_to, paths_to


class OptimalContestPlanner:
    """Bộ giải thuật chuyên biệt cho mọi map HEXUDON 2026."""

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

        # Danh sách mục tiêu
        if targets is not None:
            self.targets = list(targets)
        elif all(s.pos in set(TARGET_CELLS) for s in config.spots):
            self.targets = list(TARGET_CELLS)
        else:
            self.targets = [s.pos for s in config.spots]

        # Ánh xạ cell_id -> type/brand
        if cell_types is not None:
            self.cell_types = dict(cell_types)
        else:
            self.cell_types = {s.pos: s.brand for s in config.spots}

        # Trạm tiếp xăng chiến lược
        if strategic_hubs is not None:
            self.strategic_hubs = list(strategic_hubs)
        else:
            hubs = [h for h in STRATEGIC_HUBS if h in self.targets]
            self.strategic_hubs = hubs if hubs else ([self.targets[0]] if self.targets else [0])

    def precompute_all_pairs(
        self, start_cells: Sequence[int], road_condition: Dict[int, int]
    ) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], Dict[Tuple[int, int], List[int]]]:
        """Tiền tính toán khoảng cách steps, fuel và đường đi giữa mọi cặp đỉnh."""
        hex_map = HexudonMap(self.config, road_condition)
        all_nodes = set(self.targets) | set(start_cells)
        steps_fast: Dict[Tuple[int, int], int] = {}
        fuel_fast: Dict[Tuple[int, int], int] = {}
        paths_fast: Dict[Tuple[int, int], List[int]] = {}
        for src in all_nodes:
            sf, ff, pf = hex_map.dijkstra_dual(src, all_nodes, fuel_weight=0.05)
            for dst in all_nodes:
                steps_fast[(src, dst)] = sf.get(dst, 999999)
                fuel_fast[(src, dst)] = ff.get(dst, 999999)
                paths_fast[(src, dst)] = pf.get(dst, [])
        return steps_fast, fuel_fast, paths_fast

    def plan_day(
        self,
        agents_state: Sequence[AgentState],
        max_steps: int,
        road_condition: Dict[int, int],
        collected_types_match: Optional[Set[int]] = None,
    ) -> Tuple[Dict[int, List[int]], List[List[int]]]:
        if collected_types_match is None:
            collected_types_match = set()

        hex_map = HexudonMap(self.config, road_condition)
        start_cells = [a.pos for a in agents_state]
        all_nodes = set(self.targets) | set(start_cells)

        # TẦNG 2: Tiền xử lý Dual-weight Dijkstra (Fast / Eco / Refuel)
        steps_fast: Dict[Tuple[int, int], int] = {}
        fuel_fast: Dict[Tuple[int, int], int] = {}
        paths_fast: Dict[Tuple[int, int], List[int]] = {}

        steps_eco: Dict[Tuple[int, int], int] = {}
        fuel_eco: Dict[Tuple[int, int], int] = {}
        paths_eco: Dict[Tuple[int, int], List[int]] = {}

        steps_refuel: Dict[Tuple[int, int], int] = {}
        paths_refuel: Dict[Tuple[int, int], List[int]] = {}

        for src in all_nodes:
            sf, ff, pf = hex_map.dijkstra_dual(src, all_nodes, fuel_weight=0.05)
            se, fe, pe = hex_map.dijkstra_dual(src, all_nodes, fuel_weight=0.45)
            sr, pr = hex_map.dijkstra_refuel(src, all_nodes)

            for dst in all_nodes:
                steps_fast[(src, dst)] = sf.get(dst, 999999)
                fuel_fast[(src, dst)] = ff.get(dst, 999999)
                paths_fast[(src, dst)] = pf.get(dst, [])

                steps_eco[(src, dst)] = se.get(dst, 999999)
                fuel_eco[(src, dst)] = fe.get(dst, 999999)
                paths_eco[(src, dst)] = pe.get(dst, [])

                steps_refuel[(src, dst)] = sr.get(dst, 999999)
                paths_refuel[(src, dst)] = pr.get(dst, [])

        patrol_ids = [i for i, a in enumerate(agents_state) if a.kind == rules.PATROL]
        refuel_ids = [i for i, a in enumerate(agents_state) if a.kind == rules.REFUEL]

        # TẦNG 5: Zero-Detour Station Intercept (Chọn trạm đón lõng)
        refuel_cell = agents_state[refuel_ids[0]].pos if refuel_ids else start_cells[0]
        if max_steps >= 170 and 475 in self.targets:
            hub = 475
        else:
            hub = min(self.strategic_hubs, key=lambda h: steps_refuel.get((refuel_cell, h), 999999))

        hub_arrival = steps_refuel.get((refuel_cell, hub), 0)
        refuel_schedule = {hub: (hub_arrival, max_steps)}

        # Phân tầng mục tiêu theo độ dài ngày (tránh bẫy vùng thưa)
        if max_steps <= 155 and set(TARGET_CELLS).issubset(set(self.targets)):
            base_targets = set(self.targets) - SPARSE_CLUSTER
        else:
            base_targets = set(self.targets)

        # Anti-Clumping ở ngõ cụt 34
        top_left_anchor = 34 if 34 in self.targets else None
        if top_left_anchor is not None and patrol_ids:
            closest_to_top_left = min(
                patrol_ids, key=lambda aid: steps_fast.get((agents_state[aid].pos, top_left_anchor), 999999)
            )
        else:
            closest_to_top_left = patrol_ids[0] if patrol_ids else 0

        # TẦNG 3: Centralized Stock Reservation
        initial_stocks = {s.pos: s.stocks for s in self.config.spots}
        remaining_stock_today = dict(initial_stocks)
        reserved_count: Dict[int, int] = {st: 0 for st in self.targets}
        collected_types_today: Set[int] = set()

        # Step 1 free collection
        for aid in patrol_ids:
            c = agents_state[aid].pos
            if c in self.targets and remaining_stock_today.get(c, 0) > 0:
                remaining_stock_today[c] -= 1
                collected_types_today.add(self.cell_types[c])

        agent_routes: Dict[int, List[int]] = {}

        for aid in patrol_ids:
            cur_cell = agents_state[aid].pos
            cur_fuel = agents_state[aid].fuel
            cur_step = 0
            route: List[int] = []
            visited: Set[int] = set()

            if cur_cell in self.targets:
                route.append(cur_cell)
                visited.add(cur_cell)
                cur_step += 1

            allow_top_left_end = aid == closest_to_top_left
            allowed = set(base_targets)
            if max_steps <= 90 and aid != closest_to_top_left and set(TARGET_CELLS).issubset(set(self.targets)):
                allowed -= TOP_LEFT_CORNER

            # TẦNG 4: Khởi tạo tham lam theo hàm mục tiêu 5 tầng
            while True:
                best_nxt = None
                best_score = 1e9

                # Dual-weight: eco nếu xăng dưới 45%, fast nếu xăng dồi dào
                use_eco = cur_fuel < 0.45 * self.max_fuel
                s_table = steps_eco if use_eco else steps_fast
                f_table = fuel_eco if use_eco else fuel_fast

                for cand in allowed:
                    if cand in visited:
                        continue
                    # Quản lý kho: không nhắm quán đã hết phần
                    if remaining_stock_today.get(cand, 0) - reserved_count.get(cand, 0) <= 0:
                        continue

                    ds = s_table.get((cur_cell, cand), 999999)
                    df = f_table.get((cur_cell, cand), 999999)

                    if cur_step + ds > max_steps or cur_fuel < df:
                        continue

                    u_type = self.cell_types.get(cand, 0)
                    bonus = 0.0
                    # Ưu tiên 1: L (loại chưa từng ăn cả giải)
                    if u_type not in collected_types_match:
                        bonus -= 5000.0
                    # Ưu tiên 2: D (loại chưa ăn trong ngày)
                    if u_type not in collected_types_today:
                        bonus -= 1000.0
                    # Ưu tiên 3: Ghép cặp Platoon nếu quán còn >= 2 stock
                    if reserved_count.get(cand, 0) > 0 and remaining_stock_today.get(cand, 0) > reserved_count[cand]:
                        bonus -= 3.5

                    # Thưởng tiếp xăng khi sắp cạn
                    if cand in refuel_schedule and cur_fuel - df < self.max_fuel // 2:
                        t_min, t_max = refuel_schedule[cand]
                        if t_min <= cur_step + ds <= t_max:
                            bonus -= 35.0

                    # Chống kẹt xe ở ngõ cụt
                    remaining_steps = max_steps - (cur_step + ds)
                    if remaining_steps < 18 and not allow_top_left_end:
                        if cand in TOP_LEFT_CORNER or cand in SPARSE_CLUSTER:
                            bonus += 50.0
                        elif cand in CORE_CLUSTER:
                            bonus -= 10.0

                    score = bonus + ds
                    if score < best_score:
                        best_score = score
                        best_nxt = cand

                if best_nxt is None:
                    break

                use_eco = cur_fuel < 0.45 * self.max_fuel
                s_table = steps_eco if use_eco else steps_fast
                f_table = fuel_eco if use_eco else fuel_fast

                ds = s_table[(cur_cell, best_nxt)]
                df = f_table[(cur_cell, best_nxt)]
                cur_step += ds
                cur_fuel -= df
                cur_cell = best_nxt
                route.append(best_nxt)
                visited.add(best_nxt)

                reserved_count[best_nxt] = reserved_count.get(best_nxt, 0) + 1
                if remaining_stock_today.get(best_nxt, 0) > 0:
                    remaining_stock_today[best_nxt] -= 1
                    collected_types_today.add(self.cell_types[best_nxt])

                if cur_cell in refuel_schedule:
                    t_min, t_max = refuel_schedule[cur_cell]
                    if t_min <= cur_step <= t_max:
                        cur_fuel = self.max_fuel

            # 2-Opt Local Search
            improved = True
            while improved:
                improved = False
                start_idx = 1 if (route and route[0] == agents_state[aid].pos) else 0
                for i in range(start_idx, len(route) - 2):
                    for j in range(i + 1, len(route)):
                        new_route = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
                        ok, new_steps, _, _ = self._eval_route(
                            new_route, agents_state[aid].pos, agents_state[aid].fuel,
                            max_steps, refuel_schedule, steps_fast, fuel_fast
                        )
                        curr_ok, curr_steps, _, _ = self._eval_route(
                            route, agents_state[aid].pos, agents_state[aid].fuel,
                            max_steps, refuel_schedule, steps_fast, fuel_fast
                        )
                        if ok and new_steps < curr_steps:
                            route = new_route
                            improved = True
                            break
                    if improved:
                        break

            # Greedy Insertion vét cạn bước thừa
            unvisited = list(allowed - set(route))
            inserted = True
            while inserted and unvisited:
                inserted = False
                best_cand = None
                best_test_route = None
                best_steps = 1e9
                start_idx = 1 if (route and route[0] == agents_state[aid].pos) else 0

                for cand in unvisited:
                    if remaining_stock_today.get(cand, 0) <= 0:
                        continue
                    for pos in range(start_idx, len(route) + 1):
                        if pos == len(route) and not allow_top_left_end and cand in TOP_LEFT_CORNER:
                            continue
                        test_route = route[:pos] + [cand] + route[pos:]
                        ok, t_steps, _, _ = self._eval_route(
                            test_route, agents_state[aid].pos, agents_state[aid].fuel,
                            max_steps, refuel_schedule, steps_fast, fuel_fast
                        )
                        if ok and t_steps < best_steps:
                            best_steps = t_steps
                            best_cand = cand
                            best_test_route = test_route

                if best_test_route is not None:
                    route = best_test_route
                    unvisited.remove(best_cand)
                    remaining_stock_today[best_cand] = max(0, remaining_stock_today.get(best_cand, 0) - 1)
                    collected_types_today.add(self.cell_types[best_cand])
                    inserted = True

            agent_routes[aid] = route

        for rid in refuel_ids:
            agent_routes[rid] = [hub]

        plan = self._build_token_plan(
            agents_state, agent_routes, max_steps, steps_fast, paths_fast, hex_map
        )
        return agent_routes, plan

    def _eval_route(
        self, route: List[int], start_cell: int, start_fuel: int, max_steps: int,
        refuel_sched: Dict[int, Tuple[int, int]], s_table, f_table
    ) -> Tuple[bool, int, int, int]:
        cur_cell = start_cell
        cur_fuel = start_fuel
        cur_step = 0
        collected = 0
        idx = 0
        if route and route[0] == start_cell:
            cur_step += 1
            collected += 1
            if start_cell in refuel_sched:
                t_min, t_max = refuel_sched[start_cell]
                if t_min <= cur_step <= t_max:
                    cur_fuel = self.max_fuel
            idx = 1

        for i in range(idx, len(route)):
            nxt = route[i]
            ds = s_table.get((cur_cell, nxt), 999999)
            df = f_table.get((cur_cell, nxt), 999999)
            if cur_fuel < df or cur_step + ds > max_steps:
                return False, cur_step, cur_fuel, collected
            cur_step += ds
            cur_fuel -= df
            cur_cell = nxt
            collected += 1
            if cur_cell in refuel_sched:
                t_min, t_max = refuel_sched[cur_cell]
                if t_min <= cur_step <= t_max:
                    cur_fuel = self.max_fuel
        return True, cur_step, cur_fuel, collected

    def _build_token_plan(
        self, agents_state: Sequence[AgentState], agent_routes: Dict[int, List[int]],
        max_steps: int, dist_steps, paths, hex_map: HexudonMap
    ) -> List[List[int]]:
        dir_index = self.config.dir_index
        plan: List[List[int]] = []

        for i, agent in enumerate(agents_state):
            cur = agent.pos
            route = agent_routes.get(i, [])
            tokens: List[int] = []
            used_steps = 0

            if agent.kind == rules.REFUEL:
                hub = route[0] if route else cur
                if cur != hub:
                    p = paths.get((cur, hub), [])
                    for step_idx in range(len(p) - 1):
                        u, v = p[step_idx], p[step_idx + 1]
                        ds, _ = hex_map.get_step_and_fuel_cost(u)
                        if used_steps + ds > max_steps:
                            break
                        d = dir_index[u].get(v)
                        if d is not None:
                            tokens.append(d)
                            used_steps += ds
                            cur = v
                rem = max_steps - used_steps
                if rem > 0:
                    tokens.append(-rem)
                plan.append(tokens)
                continue

            route_idx = 0
            if route and route[0] == cur:
                route_idx = 1

            for tgt in route[route_idx:]:
                if cur == tgt:
                    continue
                p = paths.get((cur, tgt), [])
                if not p:
                    continue
                for step_idx in range(len(p) - 1):
                    u, v = p[step_idx], p[step_idx + 1]
                    ds, _ = hex_map.get_step_and_fuel_cost(u)
                    if used_steps + ds > max_steps:
                        break
                    d = dir_index[u].get(v)
                    if d is not None:
                        tokens.append(d)
                        used_steps += ds
                        cur = v
                if used_steps >= max_steps:
                    break

            rem = max_steps - used_steps
            if rem > 0:
                tokens.append(-rem)
            plan.append(tokens)

        return compact_plan(plan)

    def to_day_solution(self, agent_routes: Dict[int, List[int]], ctx: DayContext) -> DaySolution:
        spot_idx_at = self.config.spot_index_at
        patrol_slots = {aid: slot for slot, aid in enumerate(ctx.patrols)}
        tours: List[List[int]] = [[] for _ in range(ctx.n_patrols)]
        meets: List[Meet] = []

        refuel_ids = ctx.cars
        car_slot = 0 if refuel_ids else 0
        refuel_hub = agent_routes.get(refuel_ids[0], [None])[0] if refuel_ids else None

        for aid, slot in patrol_slots.items():
            route = agent_routes.get(aid, [])
            for spot_cell in route:
                if spot_cell in spot_idx_at:
                    s_idx = spot_idx_at[spot_cell]
                    tours[slot].append(s_idx)
                    if refuel_hub is not None and spot_cell == refuel_hub:
                        meets.append(Meet(patrol=slot, stop_idx=len(tours[slot]) - 1, car=car_slot))

        return DaySolution(tours=tours, meets=meets, waits={})

    def to_decoded(self, ctx: DayContext, plan: List[List[int]]) -> Decoded:
        res = run_day(
            self.config,
            ctx.steps,
            ctx.agents,
            plan,
            [s.stocks for s in self.config.spots],
            ctx.road_status,
        )
        score = 0.0
        if res.ok:
            from .objective import score_day

            score = score_day(ctx, res)
        return Decoded(plan=plan, result=res, score=score)
