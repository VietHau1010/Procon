import sys
sys.path.insert(0, '.')
import heapq
from typing import Dict, List, Set, Tuple, Optional, Sequence
import json

from hexudon import rules
from hexudon.model import MatchConfig, AgentState, Spot
from hexudon.simulator import run_day, DayResult

# Mã địa hình chuẩn theo đề HEXUDON 2026
TERRAIN_PLAIN = rules.PLAIN        # Đồng bằng: 2 bước, 1 xăng (Quán udon nằm trên ô này)
TERRAIN_MOUNTAIN = rules.MOUNTAIN  # Núi: 3 bước, 2 xăng
TERRAIN_POND = rules.POND          # Ao: Không đi qua được
TERRAIN_ROAD = rules.ROAD          # Đường bộ: Thông thoáng(0)->1b,2x | Đông(1)->2b,2x | Tắc(2)->4b,2x


class HexudonMap:
    """
    Xử lý bản đồ lục giác even-r (hàng chẵn lệch phải) và tính đường đi ngắn nhất
    theo đúng Bảng 1 Thể lệ HEXUDON 2026.
    """
    def __init__(
        self,
        width: int,
        height: int,
        terrain_grid: List[int],
        destinations: Dict[int, Dict],  # cell_id -> {"type": int, "stock": int}
        road_condition: Dict[int, int],  # cell_id -> 0, 1, 2
        neighbors: Optional[List[List[int]]] = None,
        dir_index: Optional[List[Dict[int, int]]] = None,
    ):
        self.width = width
        self.height = height
        self.n_cells = width * height
        self.terrain = terrain_grid
        self.destinations = destinations
        self.road_cond = dict(road_condition)
        self.adj = neighbors if neighbors is not None else self._build_even_r_neighbors()
        self.dir_index = dir_index
        self._patrol_cache: Dict[Tuple[int, float], Tuple[Dict[int, int], Dict[int, int], Dict[int, List[int]]]] = {}
        self._refuel_cache: Dict[int, Tuple[Dict[int, int], Dict[int, List[int]]]] = {}

    def _build_even_r_neighbors(self) -> Dict[int, List[int]]:
        adj: Dict[int, List[int]] = {i: [] for i in range(self.n_cells)}
        for r in range(self.height):
            for c in range(self.width):
                u = r * self.width + c
                if self.terrain[u] == TERRAIN_POND:
                    continue
                if r % 2 == 0:
                    deltas = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]
                else:
                    deltas = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]

                for dr, dc in deltas:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        v = nr * self.width + nc
                        if self.terrain[v] != TERRAIN_POND:
                            adj[u].append(v)
        return adj

    def get_step_and_fuel_cost(self, cell: int) -> Tuple[int, int]:
        t = self.terrain[cell]
        if t == TERRAIN_PLAIN:
            return (2, 1)
        if t == TERRAIN_MOUNTAIN:
            return (3, 2)
        if t == TERRAIN_ROAD:
            cond = self.road_cond.get(cell, 0)
            if cond == 0:
                return (1, 2)
            elif cond == 1:
                return (2, 2)
            else:
                return (4, 2)
        return (999, 999)

    def shortest_path_patrol(self, src: int, fuel_weight: float = 0.05) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, List[int]]]:
        """
        Dijkstra cho xe Patrol từ ô `src`:
        Trả về (steps_to, fuel_to, paths_to).
        Tự động né đường Đông đúc/Ùn tắc sang Đồng bằng để tiết kiệm cả bước lẫn xăng.
        """
        key = (src, round(fuel_weight, 2))
        if key in self._patrol_cache:
            return self._patrol_cache[key]

        steps_to = {src: 0}
        fuel_to = {src: 0}
        best_score = {src: 0.0}
        parent = {src: None}
        pq = [(0.0, 0, 0, src)]

        while pq:
            score, d_step, d_fuel, u = heapq.heappop(pq)
            if score > best_score.get(u, 1e18):
                continue
            step_c, fuel_c = self.get_step_and_fuel_cost(u)
            for v in self.adj[u]:
                if v < 0 or self.terrain[v] == TERRAIN_POND:
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

        # Build paths
        paths_to: Dict[int, List[int]] = {}
        for dst in steps_to:
            cur = dst
            p = []
            while cur is not None:
                p.append(cur)
                cur = parent[cur]
            paths_to[dst] = p[::-1]

        self._patrol_cache[key] = (steps_to, fuel_to, paths_to)
        return steps_to, fuel_to, paths_to

    def shortest_path_refuel(self, src: int) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """Dijkstra cho xe Refuel (không tốn xăng, chỉ tối ưu số bước)."""
        if src in self._refuel_cache:
            return self._refuel_cache[src]

        steps_to = {src: 0}
        parent = {src: None}
        pq = [(0, src)]

        while pq:
            d_step, u = heapq.heappop(pq)
            if d_step > steps_to.get(u, 10**9):
                continue
            step_c, _ = self.get_step_and_fuel_cost(u)
            for v in self.adj[u]:
                if v < 0 or self.terrain[v] == TERRAIN_POND:
                    continue
                ns = d_step + step_c
                if ns < steps_to.get(v, 10**9):
                    steps_to[v] = ns
                    parent[v] = u
                    heapq.heappush(pq, (ns, v))

        paths_to: Dict[int, List[int]] = {}
        for dst in steps_to:
            cur = dst
            p = []
            while cur is not None:
                p.append(cur)
                cur = parent[cur]
            paths_to[dst] = p[::-1]

        self._refuel_cache[src] = (steps_to, paths_to)
        return steps_to, paths_to


def choose_roles_for_match(num_agents: int, max_fuel: int, width: int, height: int) -> List[int]:
    """
    Hàm chọn loại tác nhân đầu trận (0 = Patrol, 1 = Refuel):
    - Với 3..6 xe: luôn chọn (N - 1) Patrol + 1 Refuel.
    - Với 7..8 xe trên bản đồ lớn và bình xăng nhỏ: chọn (N - 2) Patrol + 2 Refuel.
    """
    if num_agents <= 6:
        return [0] * (num_agents - 1) + [1]
    if width * height >= 600 and max_fuel < 120:
        return [0] * (num_agents - 2) + [1, 1]
    return [0] * (num_agents - 1) + [1]


print("HexudonMap & choose_roles_for_match loaded.")
# Append Day Planner
class HexudonDayPlanner:
    def __init__(
        self,
        hex_map: HexudonMap,
        roles: List[int],
        max_fuel: int,
        config: MatchConfig,
    ):
        self.map = hex_map
        self.roles = roles
        self.max_fuel = max_fuel
        self.config = config
        self.patrol_indices = [i for i, r in enumerate(roles) if r == 0]
        self.refuel_indices = [i for i, r in enumerate(roles) if r == 1]
        self.all_types = {info["type"] for info in hex_map.destinations.values()}

    def plan_day(
        self,
        agents_state: Sequence[AgentState],
        max_steps: int,
        collected_types_match: Set[int],
    ) -> Tuple[List[List[int]], Dict]:
        """
        Lập kế hoạch di chuyển toàn ngày cho cả đội dựa trên 5 tầng chiến thuật:
        - Dual-weight Dijkstra (né đường tắc, tiết kiệm xăng khi cạn)
        - Quản lý kho Centralized Reservation
        - Ưu tiên L (-5000) -> D (-1000) -> Q (d_steps, platoon)
        - Zero-Detour Station Intercept: Xe Refuel đón lõng tại quán udon
        - End-of-day Parking: Đỗ xe trên quán udon ở cuối ngày
        """
        n_agents = len(self.roles)
        dir_index = self.config.dir_index
        
        agent_cells = [a.pos for a in agents_state]
        agent_fuels = [a.fuel for a in agents_state]
        agent_time = [0] * n_agents
        
        # State tracking
        visited_by_agent_today: List[Set[int]] = [set() for _ in range(n_agents)]
        remaining_stock_today: Dict[int, int] = {
            st: info["stock"] for st, info in self.map.destinations.items()
        }
        collected_types_today: Set[int] = set()
        reserved_count: Dict[int, int] = {st: 0 for st in self.map.destinations}
        
        # Routes of cells for each agent
        routes: List[List[int]] = [[agent_cells[i]] for i in range(n_agents)]
        plan_tokens: List[List[int]] = [[] for _ in range(n_agents)]
        
        # Check initial positions (Step 1 free collection if starting on a station)
        for p_idx in self.patrol_indices:
            c = agent_cells[p_idx]
            if c in self.map.destinations and remaining_stock_today[c] > 0:
                visited_by_agent_today[p_idx].add(c)
                remaining_stock_today[c] -= 1
                collected_types_today.add(self.map.destinations[c]["type"])

        # Target station assignment for each patrol
        patrol_current_target: Dict[int, Optional[int]] = {p: None for p in self.patrol_indices}
        refuel_rendezvous: Dict[int, Optional[Tuple[int, int]]] = {r: None for r in self.refuel_indices} # {r_idx: (station, arrive_time)}

        # Simulation loop by steps / events
        while any(agent_time[i] < max_steps for i in range(n_agents)):
            # 1. First, check if any refuel car needs a new rendezvous target
            for r_idx in self.refuel_indices:
                r_cell = agent_cells[r_idx]
                r_time = agent_time[r_idx]
                if r_time >= max_steps:
                    continue
                
                # Check if current rendezvous is done or None
                if refuel_rendezvous[r_idx] is None or agent_cells[r_idx] == refuel_rendezvous[r_idx][0]:
                    # Find urgent patrol
                    urgent_patrols = sorted(self.patrol_indices, key=lambda idx: agent_fuels[idx])
                    chosen_intercept = None
                    best_arr = 9999
                    
                    r_steps_map, r_paths = self.map.shortest_path_refuel(r_cell)
                    for p_idx in urgent_patrols:
                        if agent_time[p_idx] >= max_steps:
                            continue
                        p_cell = agent_cells[p_idx]
                        p_fuel = agent_fuels[p_idx]
                        p_steps_map, p_fuel_map, _ = self.map.shortest_path_patrol(p_cell, fuel_weight=0.2)
                        
                        valid = [
                            st for st in self.map.destinations
                            if p_fuel_map.get(st, 999) <= p_fuel
                            and st not in visited_by_agent_today[p_idx]
                            and remaining_stock_today.get(st, 0) > 0
                        ]
                        if valid:
                            target_st = min(valid, key=lambda st: p_steps_map.get(st, 999))
                            chosen_intercept = target_st
                            break
                    
                    if chosen_intercept is not None and chosen_intercept != r_cell:
                        path = r_paths.get(chosen_intercept, [])
                        for step_idx in range(len(path) - 1):
                            u, v = path[step_idx], path[step_idx + 1]
                            ds, _ = self.map.get_step_and_fuel_cost(u)
                            if agent_time[r_idx] + ds > max_steps:
                                break
                            d = dir_index[u].get(v)
                            if d is not None:
                                plan_tokens[r_idx].append(d)
                                agent_time[r_idx] += ds
                                agent_cells[r_idx] = v
                                routes[r_idx].append(v)
                        refuel_rendezvous[r_idx] = (chosen_intercept, agent_time[r_idx])
                    else:
                        # Wait at current spot
                        rem = max_steps - agent_time[r_idx]
                        if rem > 0:
                            plan_tokens[r_idx].append(-rem)
                            agent_time[r_idx] = max_steps

            # 2. Advance patrols to their next targets
            advanced_any = False
            for p_idx in self.patrol_indices:
                p_cell = agent_cells[p_idx]
                p_fuel = agent_fuels[p_idx]
                p_time = agent_time[p_idx]
                if p_time >= max_steps:
                    continue

                # Dual-weight Dijkstra: eco if low fuel, fast if high fuel
                fuel_w = 0.45 if p_fuel < 0.45 * self.max_fuel else 0.05
                steps_to, fuel_to, paths_to = self.map.shortest_path_patrol(p_cell, fuel_weight=fuel_w)

                # Check if refuel rendezvous is active at any station
                r_cells = [agent_cells[r] for r in self.refuel_indices]
                if any(p_cell == rc for rc in r_cells):
                    # Refuel at current station
                    agent_fuels[p_idx] = self.max_fuel
                    p_fuel = self.max_fuel

                # Find candidates
                candidates = []
                for st in self.map.destinations:
                    if st in visited_by_agent_today[p_idx]:
                        continue
                    if remaining_stock_today.get(st, 0) - reserved_count.get(st, 0) <= 0:
                        continue
                    st_step = steps_to.get(st, 9999)
                    st_fuel = fuel_to.get(st, 9999)
                    if p_time + st_step > max_steps:
                        continue
                    if st_fuel > p_fuel:
                        # Check if refuel car is at st
                        if st not in r_cells:
                            continue
                    candidates.append(st)

                if candidates:
                    def score_station(st: int) -> float:
                        u_type = self.map.destinations[st]["type"]
                        d_steps = max(1, steps_to.get(st, 9999))
                        bonus = 0.0
                        if u_type not in collected_types_match:
                            bonus -= 5000.0
                        if u_type not in collected_types_today:
                            bonus -= 1000.0
                        if reserved_count.get(st, 0) > 0 and remaining_stock_today.get(st, 0) > reserved_count[st]:
                            bonus -= 3.5
                        return bonus + d_steps

                    target = min(candidates, key=score_station)
                    reserved_count[target] += 1
                else:
                    # End-of-day Parking: head to nearest station to park for tomorrow
                    target = min(self.map.destinations.keys(), key=lambda st: steps_to.get(st, 9999))

                # Step towards target
                path = paths_to.get(target, [])
                if len(path) >= 2:
                    u, v = path[0], path[1]
                    ds, df = self.map.get_step_and_fuel_cost(u)
                    if p_time + ds <= max_steps and p_fuel >= df:
                        d = dir_index[u].get(v)
                        if d is not None:
                            plan_tokens[p_idx].append(d)
                            agent_time[p_idx] += ds
                            agent_fuels[p_idx] -= df
                            agent_cells[p_idx] = v
                            routes[p_idx].append(v)
                            advanced_any = True

                            # Check if arrived at station
                            if v in self.map.destinations and v not in visited_by_agent_today[p_idx]:
                                if remaining_stock_today[v] > 0:
                                    visited_by_agent_today[p_idx].add(v)
                                    remaining_stock_today[v] -= 1
                                    collected_types_today.add(self.map.destinations[v]["type"])
                                    reserved_count[v] = max(0, reserved_count[v] - 1)
                            
                            # Check if meeting refuel car at v
                            if any(v == agent_cells[r] for r in self.refuel_indices):
                                agent_fuels[p_idx] = self.max_fuel
                    else:
                        # Cannot move further, wait
                        rem = max_steps - p_time
                        if rem > 0:
                            plan_tokens[p_idx].append(-rem)
                            agent_time[p_idx] = max_steps
                        advanced_any = True
                else:
                    # Already at target or no path, wait
                    rem = max_steps - p_time
                    if rem > 0:
                        plan_tokens[p_idx].append(-rem)
                        agent_time[p_idx] = max_steps
                    advanced_any = True

            if not advanced_any:
                # Force finish any remaining times
                for i in range(n_agents):
                    rem = max_steps - agent_time[i]
                    if rem > 0:
                        plan_tokens[i].append(-rem)
                        agent_time[i] = max_steps
                break

        # Post-process: compact plan
        from hexudon.planner.decoder import compact_plan
        final_plan = compact_plan(plan_tokens)
        
        info = {
            "collected_types_today": collected_types_today,
            "end_positions": agent_cells,
            "end_fuels": agent_fuels,
        }
        return final_plan, info


# Run test on Day 0
with open("replays/contest-087d9b9a-t78-20261112-165702.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('path') == '/api/game/config':
            cfg_dict = entry['response']
        if entry.get('path') == '/api/game/state' and entry.get('response', {}).get('day') == 0:
            st0_dict = entry['response']
            break

cfg = MatchConfig.from_json(cfg_dict)
st0_road = {int(k): v for k, v in st0_dict.get('road_condition', {}).items()}
destinations = {s.pos: {"type": s.brand, "stock": s.stocks} for s in cfg.spots}

hex_map = HexudonMap(
    width=cfg.width,
    height=cfg.height,
    terrain_grid=cfg.cells,
    destinations=destinations,
    road_condition=st0_road,
    neighbors=cfg.neighbors,
    dir_index=cfg.dir_index,
)

roles = choose_roles_for_match(cfg.n_agents, cfg.fuel_limits, cfg.width, cfg.height)
print(f"Auto-selected roles: {roles}")

planner = HexudonDayPlanner(hex_map, roles, cfg.fuel_limits, cfg)
agents = [
    AgentState(roles[i], cfg.agent_starts[i], cfg.fuel_limits)
    for i in range(len(roles))
]

plan, info = planner.plan_day(agents, cfg.day_steps[0], set())
print("Plan generated successfully.")
print(f"Collected types today: {len(info['collected_types_today'])} -> {info['collected_types_today']}")
print(f"End positions: {info['end_positions']}")
print(f"End fuels: {info['end_fuels']}")

# Simulate with golden simulator
res = run_day(cfg, cfg.day_steps[0], agents, plan, [s.stocks for s in cfg.spots], st0_road)
print(f"Simulator result: ok={res.ok}, brands={len(res.brands)}, servings={res.servings}")
if not res.ok:
    print(f"Simulator error: {res.error}")
