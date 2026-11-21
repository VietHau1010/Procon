"""Dựng nghiệm ban đầu cho chiến thuật mới:
- Master Cycle TSP + 2-Opt cho cụm quán ăn.
- Bảo đảm 100% nhãn hàng (Brand Coverage) mỗi ngày.
- Đón lõng tiếp nhiên liệu tại trạm trung tâm (Hub Rendezvous).
- Chèn vét cạn các bước trống (Greedy Fill).
- Thích ứng với cả map dồi dào kho (Platoon) lẫn map khan hiếm kho (Central Reservation).
"""

import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .decoder import decode
from ..planner.solution import DayContext, DaySolution, Meet

INF = float("inf")
NEAR_K = 30


class SpotNeighbours:
    """Danh sách spot gần nhất quanh một ô, tính lười và nhớ lại."""

    def __init__(self, ctx: DayContext, k: int = NEAR_K):
        self.ctx = ctx
        self.k = min(k, len(ctx.spot_cells))
        self._cache: Dict[int, Tuple[int, ...]] = {}

    def of(self, cell: int) -> Tuple[int, ...]:
        cached = self._cache.get(cell)
        if cached is not None:
            return cached
        table = self.ctx.dist_step
        scored = []
        for si, spot_cell in enumerate(self.ctx.spot_cells):
            d = table.cost(cell, spot_cell)
            if d != INF:
                scored.append((d, si))
        scored.sort()
        out = tuple(si for _, si in scored[: self.k])
        self._cache[cell] = out
        return out


class InsertionState:
    """Trạng thái kế toán để đánh giá phép chèn trong O(1)."""

    def __init__(self, ctx: DayContext, sol: Optional[DaySolution] = None):
        self.ctx = ctx
        self.sol = sol.copy() if sol is not None else ctx.empty_solution()
        self.near = SpotNeighbours(ctx)
        self.allowance: List[float] = []
        self.tour_steps: List[float] = []
        self.tour_fuel: List[float] = []
        self.targeted: Dict[int, int] = {}
        self.covered: Dict[int, int] = {}
        self._recompute()

    def _cells_of(self, slot: int) -> List[int]:
        return [self.ctx.spot_cells[s] for s in self.sol.tours[slot]]

    def _recompute(self):
        ctx = self.ctx
        has_car = ctx.n_cars > 0
        self.allowance = []
        self.tour_steps = []
        self.tour_fuel = []
        self.targeted = {}
        self.covered = {}
        for slot in range(ctx.n_patrols):
            fuel0 = ctx.fuel_of_patrol(slot)
            self.allowance.append(fuel0 + (ctx.config.fuel_limits if has_car else 0))
            steps = 0.0
            fuel = 0.0
            cur = ctx.start_of_patrol(slot)
            for spot in self.sol.tours[slot]:
                cell = ctx.spot_cells[spot]
                s, f = ctx.dist_step.steps_fuel(cur, cell)
                if s == INF:
                    s, f = 0.0, 0.0
                steps += s
                fuel += f
                cur = cell
                self.targeted[spot] = self.targeted.get(spot, 0) + 1
                brand = ctx.spot_brand[spot]
                self.covered[brand] = self.covered.get(brand, 0) + 1
            self.tour_steps.append(steps)
            self.tour_fuel.append(fuel)

    def gain_of(self, spot: int) -> float:
        ctx = self.ctx
        w = ctx.weights
        brand = ctx.spot_brand[spot]
        if ctx.stocks[spot] <= self.targeted.get(spot, 0):
            return 0.0
        gain = 0.0
        if self.covered.get(brand, 0) == 0:
            gain += w.B
            if brand not in ctx.visited_brands:
                gain += ctx.new_brand_value
        if ctx.stocks[spot] > self.targeted.get(spot, 0):
            gain += w.serving
        return gain

    def delta(self, slot: int, pos: int, spot: int) -> Optional[Tuple[float, float]]:
        """(Δbước, Δxăng) khi chèn spot vào vị trí pos của tour slot."""
        ctx = self.ctx
        cells = self._cells_of(slot)
        cell = ctx.spot_cells[spot]
        prev = cells[pos - 1] if pos > 0 else ctx.start_of_patrol(slot)
        s1, f1 = ctx.dist_step.steps_fuel(prev, cell)
        if s1 == INF:
            return None
        if pos < len(cells):
            nxt = cells[pos]
            s2, f2 = ctx.dist_step.steps_fuel(cell, nxt)
            s0, f0 = ctx.dist_step.steps_fuel(prev, nxt)
            if s2 == INF:
                return None
            return s1 + s2 - s0, f1 + f2 - f0
        return s1, f1

    def feasible(self, slot: int, dsteps: float, dfuel: float) -> bool:
        return (
            self.tour_steps[slot] + dsteps <= self.ctx.steps
            and self.tour_fuel[slot] + dfuel <= self.allowance[slot]
        )

    def apply(self, slot: int, pos: int, spot: int, dsteps: float, dfuel: float):
        self.sol.tours[slot].insert(pos, spot)
        if self.sol.meets:
            self.sol.meets = [
                Meet(m.patrol, m.stop_idx + 1, m.car)
                if m.patrol == slot and m.stop_idx >= pos
                else m
                for m in self.sol.meets
            ]
        if self.sol.waits:
            self.sol.waits = {
                ((s, i + 1) if s == slot and i >= pos else (s, i)): w
                for (s, i), w in self.sol.waits.items()
            }
        self.tour_steps[slot] += dsteps
        self.tour_fuel[slot] += dfuel
        self.targeted[spot] = self.targeted.get(spot, 0) + 1
        brand = self.ctx.spot_brand[spot]
        self.covered[brand] = self.covered.get(brand, 0) + 1

    def candidate_spots(self, slot: int, pos: int) -> Tuple[int, ...]:
        cells = self._cells_of(slot)
        anchor = cells[pos - 1] if pos > 0 else self.ctx.start_of_patrol(slot)
        candidates = set(self.near.of(anchor))
        # Luôn đảm bảo toàn bộ các quán của nhãn hàng chưa phủ có mặt trong ứng viên
        for b in self.ctx.config.brands:
            if self.covered.get(b, 0) == 0:
                for si in self.ctx.config.spots_by_brand.get(b, ()):
                    candidates.add(si)
        return tuple(sorted(candidates))

    def best_insertion(
        self,
        rng: Optional[random.Random] = None,
        coverage_first: bool = False,
        regret: int = 0,
    ):
        ctx = self.ctx
        in_tour = [set(t) for t in self.sol.tours]
        if regret > 0:
            return self._best_by_regret(in_tour, regret)
        best = None
        best_key = 0.0
        for slot in range(ctx.n_patrols):
            for pos in range(len(self.sol.tours[slot]) + 1):
                for spot in self.candidate_spots(slot, pos):
                    if spot in in_tour[slot]:
                        continue
                    gain = self.gain_of(spot)
                    if gain <= 0:
                        continue
                    if coverage_first and self.covered.get(ctx.spot_brand[spot], 0) > 0:
                        continue
                    d = self.delta(slot, pos, spot)
                    if d is None:
                        continue
                    dsteps, dfuel = d
                    if not self.feasible(slot, dsteps, dfuel):
                        continue
                    key = gain / max(1.0, dsteps)
                    if rng is not None:
                        key *= 1.0 + 0.05 * rng.random()
                    if key > best_key:
                        best_key = key
                        best = (slot, pos, spot, dsteps, dfuel)
        return best

    def _best_by_regret(self, in_tour, regret: int):
        ctx = self.ctx
        per_spot: Dict[int, List[Tuple[float, tuple]]] = {}
        for slot in range(ctx.n_patrols):
            for pos in range(len(self.sol.tours[slot]) + 1):
                for spot in self.candidate_spots(slot, pos):
                    if spot in in_tour[slot]:
                        continue
                    gain = self.gain_of(spot)
                    if gain <= 0:
                        continue
                    d = self.delta(slot, pos, spot)
                    if d is None:
                        continue
                    dsteps, dfuel = d
                    if not self.feasible(slot, dsteps, dfuel):
                        continue
                    key = gain / max(1.0, dsteps)
                    per_spot.setdefault(spot, []).append((key, (slot, pos, spot, dsteps, dfuel)))
        best = None
        best_regret = -INF
        for spot, options in per_spot.items():
            options.sort(key=lambda x: -x[0])
            top = options[0]
            others = options[1:regret]
            value = top[0] * len(options[:regret]) - sum(k for k, _ in others)
            if value > best_regret:
                best_regret = value
                best = top[1]
        return best


def trim_excess_stock(ctx: DayContext, sol: DaySolution) -> DaySolution:
    """Propose removing over-targeted stops, preserving refuel and wait anchors.

    Master-cycle construction is intentionally route-first and may duplicate a
    spot across patrols. The simulator correctly rejects the extra collections,
    but carrying those stops into ALNS wastes search time. Remove the most
    expensive redundant occurrence, preserving the cheapest useful visit. This
    is a route-model estimate: incidental collections still require decoding.
    """
    current = sol.copy()
    counts: Dict[int, int] = {}
    for tour in current.tours:
        for spot in tour:
            counts[spot] = counts.get(spot, 0) + 1

    def remove_one(slot: int, index: int) -> None:
        current.tours[slot].pop(index)
        new_meets = []
        for meet in current.meets:
            if meet.patrol != slot or meet.stop_idx < 0:
                new_meets.append(meet)
            elif meet.stop_idx == index:
                continue
            else:
                new_meets.append(
                    Meet(meet.patrol, meet.stop_idx - 1 if meet.stop_idx > index
                         else meet.stop_idx, meet.car)
                )
        current.meets = new_meets
        current.waits = {
            (s, i - 1 if s == slot and i > index else i): wait
            for (s, i), wait in current.waits.items()
            if not (s == slot and i == index)
        }

    for spot, count in list(counts.items()):
        excess = max(0, count - max(0, int(ctx.stocks[spot])))
        for _ in range(excess):
            candidates = []
            target = ctx.spot_cells[spot]
            for slot, tour in enumerate(current.tours):
                for index, value in enumerate(tour):
                    if value != spot:
                        continue
                    if any(m.patrol == slot and m.stop_idx == index for m in current.meets):
                        continue
                    if current.waits.get((slot, index), 0) > 0:
                        continue
                    cells = [ctx.start_of_patrol(slot)] + [ctx.spot_cells[s] for s in tour]
                    prev = cells[index]
                    nxt = cells[index + 2] if index + 2 < len(cells) else None
                    old = ctx.dist_step.cost(prev, target)
                    if nxt is not None:
                        old += ctx.dist_step.cost(target, nxt)
                        new = ctx.dist_step.cost(prev, nxt)
                    else:
                        new = 0
                    if old != INF and new != INF:
                        candidates.append((old - new, slot, index))
            if not candidates:
                break
            _, slot, index = max(candidates)
            remove_one(slot, index)
    return current


def add_fuel_meets(ctx: DayContext, sol: DaySolution) -> DaySolution:
    """Thêm điểm hẹn tiếp nhiên liệu cho xe tuần tra bị cắt vì thiếu xăng."""
    if ctx.n_cars == 0:
        return sol
    from .decoder import walk_patrol

    current = sol.copy()
    base = decode(ctx, current).score
    for slot in range(ctx.n_patrols):
        if not current.tours[slot]:
            continue
        meet_map: Dict[int, int] = {}
        for m in current.meets:
            if m.patrol == slot:
                meet_map[m.stop_idx] = m.car
        walk = walk_patrol(ctx, current, slot, meet_map, current.waits)
        if not walk.truncated and walk.end_fuel > ctx.config.fuel_limits // 4:
            continue
        stops = sorted(i for i in walk.arrivals if i >= 0)
        anchors = [stops[-1]] if stops else [-1]
        if len(stops) >= 2:
            anchors.insert(0, stops[len(stops) // 2])
        anchors.append(-1)
        for stop_idx in anchors:
            if stop_idx in meet_map:
                continue
            cell = walk.stop_cell.get(stop_idx, ctx.start_of_patrol(slot))
            car = min(
                range(ctx.n_cars),
                key=lambda c: ctx.dist_step.cost(ctx.start_of_car(c), cell),
            )
            trial = current.copy()
            trial.meets.append(Meet(patrol=slot, stop_idx=stop_idx, car=car))
            score = decode(ctx, trial).score
            if score > base:
                current = trial
                base = score
                break
    return current


def build_master_cycle(ctx: DayContext) -> List[int]:
    """Tìm chu trình Hamilton khép kín qua tất cả các quán bằng Nearest Neighbor + 2-Opt."""
    table = ctx.dist_step
    n = len(ctx.spot_cells)
    if n <= 1:
        return list(range(n))
    cur = 0
    unvisited = set(range(n)) - {cur}
    cycle = [cur]
    while unvisited:
        cur_c = ctx.spot_cells[cur]
        nxt = min(unvisited, key=lambda s: table.cost(cur_c, ctx.spot_cells[s]))
        cycle.append(nxt)
        unvisited.remove(nxt)
        cur = nxt

    def cost(c: Sequence[int]) -> float:
        return sum(
            table.cost(ctx.spot_cells[c[i]], ctx.spot_cells[c[(i + 1) % n]])
            for i in range(n)
        )

    cur_cost = cost(cycle)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = cycle[:i] + cycle[i : j + 1][::-1] + cycle[j + 1 :]
                c = cost(cand)
                if c < cur_cost:
                    cur_cost = c
                    cycle = cand
                    improved = True
                    break
            if improved:
                break
    return cycle


def construct_master(ctx: DayContext, rng: Optional[random.Random] = None) -> DaySolution:
    """Tạo lộ trình xuất sắc trên chu trình Master Cycle, bảo đảm 100% Brand Coverage và Hub Refuel."""
    table = ctx.dist_step
    cycle = build_master_cycle(ctx)
    n = len(cycle)
    spot_cells = ctx.spot_cells
    if not n or not ctx.n_patrols:
        return ctx.empty_solution()

    cycle_detours: Dict[int, float] = {}
    for i, sp in enumerate(cycle):
        p_c = spot_cells[cycle[(i - 1) % n]]
        s_c = spot_cells[sp]
        n_c = spot_cells[cycle[(i + 1) % n]]
        d = table.cost(p_c, s_c) + table.cost(s_c, n_c) - table.cost(p_c, n_c)
        cycle_detours[sp] = d

    # 1. Điểm ngoại lai dư thừa: nhãn hàng đã có >= 2 quán, độ lệch >= 12
    redundant_outliers = {
        sp
        for sp in cycle
        if len(ctx.config.spots_by_brand.get(ctx.spot_brand[sp], ())) >= 2
        and cycle_detours[sp] >= 12
    }
    # 2. Điểm ngoại lai bắt buộc duy nhất: nhãn hàng chỉ có 1 quán, độ lệch >= 20
    solitary_outliers = {
        sp
        for sp in cycle
        if len(ctx.config.spots_by_brand.get(ctx.spot_brand[sp], ())) == 1
        and cycle_detours[sp] >= 20
    }

    # Bỏ qua điểm ngoại lai dư thừa ở những ngày ngắn/trung bình (<= 160 bước)
    core_cycle = [
        sp for sp in cycle if not (ctx.steps <= 160 and sp in redundant_outliers)
    ] or cycle

    hub_candidates = sorted(
        range(n),
        key=lambda s: sum(table.cost(spot_cells[s], spot_cells[o]) for o in range(n)),
    )
    car_start = ctx.start_of_car(0) if ctx.n_cars > 0 else 0
    best_hub = min(hub_candidates[:5], key=lambda h: table.cost(car_start, spot_cells[h]))

    tours: List[List[int]] = []
    for slot in range(ctx.n_patrols):
        start_c = ctx.start_of_patrol(slot)
        # Đúng 1 xe tuần tra đại diện ghé điểm ngoại lai đơn lẻ ở ngày ngắn (<= 160 bước)
        is_designated = slot == (ctx.day % ctx.n_patrols)
        car_cycle = [
            sp
            for sp in core_cycle
            if (ctx.steps > 160 or is_designated or sp not in solitary_outliers)
        ] or core_cycle
        car_n = len(car_cycle)
        entry_idx = min(range(car_n), key=lambda i: table.cost(start_c, spot_cells[car_cycle[i]]))

        best_t: List[int] = []
        for direction in [1, -1]:
            cur_t = []
            cur_c = start_c
            steps = 0
            idx = entry_idx
            visited_in_tour = set()
            while True:
                sp = car_cycle[idx]
                nxt_c = spot_cells[sp]
                s = table.cost(cur_c, nxt_c)
                if steps + s > ctx.steps:
                    break
                cur_t.append(sp)
                visited_in_tour.add(sp)
                steps += s
                cur_c = nxt_c
                idx = (idx + direction) % car_n
                # Cho phép chạy tiếp vòng thứ 2 nếu còn dư bước
                if len(visited_in_tour) == car_n and steps + 5 > ctx.steps:
                    break
                if len(cur_t) >= car_n + 5:
                    break
            if len(cur_t) > len(best_t):
                best_t = cur_t
        tours.append(best_t)

    # Bảo đảm phủ 100% nhãn hàng (Brand Coverage)
    covered_brands = {ctx.spot_brand[s] for t in tours for s in t}
    missing_brands = set(ctx.config.brands) - covered_brands
    for b in missing_brands:
        b_spots = ctx.config.spots_by_brand.get(b, ())
        if not b_spots:
            continue
        best_cand = None
        best_detour = INF
        for sp in b_spots:
            sp_c = spot_cells[sp]
            for slot in range(ctx.n_patrols):
                cells = [ctx.start_of_patrol(slot)] + [spot_cells[s] for s in tours[slot]]
                for pos in range(len(tours[slot]) + 1):
                    prev_c = cells[pos]
                    nxt_c = cells[pos + 1] if pos + 1 < len(cells) else None
                    detour = table.cost(prev_c, sp_c) + (
                        table.cost(sp_c, nxt_c) - table.cost(prev_c, nxt_c)
                        if nxt_c is not None
                        else 0
                    )
                    if detour < best_detour:
                        best_detour = detour
                        best_cand = (slot, pos, sp)
        if best_cand is not None:
            slot, pos, sp = best_cand
            tours[slot].insert(pos, sp)
            covered_brands.add(b)

    meets: List[Meet] = []
    if ctx.n_cars > 0:
        for slot in range(ctx.n_patrols):
            if best_hub in tours[slot]:
                idx = tours[slot].index(best_hub)
                meets.append(Meet(patrol=slot, stop_idx=idx, car=0))
            elif len(tours[slot]) >= 4:
                idx = len(tours[slot]) // 2
                meets.append(Meet(patrol=slot, stop_idx=idx, car=0))

    sol = trim_excess_stock(ctx, DaySolution(tours=tours, meets=meets, waits={}))
    dec = decode(ctx, sol)
    if not dec.result.ok:
        return construct_standard(ctx, rng)
    return greedy_fill(ctx, sol)


def construct_standard(
    ctx: DayContext,
    rng: Optional[random.Random] = None,
    max_inserts: int = 400,
    coverage_first: bool = True,
) -> DaySolution:
    """Nghiệm khởi tạo tiêu chuẩn: phủ chuỗi trước, rồi tham lam theo lợi ích trên chi phí, và greedy fill."""
    state = InsertionState(ctx)
    inserts = 0
    if coverage_first:
        while inserts < max_inserts:
            best = state.best_insertion(rng=rng, coverage_first=True)
            if best is None:
                break
            state.apply(*best)
            inserts += 1
    while inserts < max_inserts:
        best = state.best_insertion(rng=rng)
        if best is None:
            break
        state.apply(*best)
        inserts += 1
    sol = add_fuel_meets(ctx, state.sol)
    return greedy_fill(ctx, sol)


def construct(
    ctx: DayContext,
    rng: Optional[random.Random] = None,
    max_inserts: int = 400,
    coverage_first: bool = True,
) -> DaySolution:
    """Khởi tạo nghiệm: ưu tiên Master Cycle nếu đủ kho udon, tự động chuyển về standard nếu map khan hiếm."""
    avg_stock = sum(ctx.stocks) / max(1, len(ctx.stocks))
    if avg_stock >= ctx.n_patrols:
        try:
            return construct_master(ctx, rng)
        except Exception:
            pass
    return construct_standard(ctx, rng, max_inserts=max_inserts, coverage_first=coverage_first)


def greedy_fill(ctx: DayContext, sol: DaySolution, max_inserts: int = 100) -> DaySolution:
    """Chèn vét cạn các spot chưa ghé sau khi đã có điểm hẹn tiếp xăng (Greedy Insertion)."""
    state = InsertionState(ctx, sol)
    for _ in range(max_inserts):
        best = state.best_insertion()
        if best is None:
            break
        state.apply(*best)
    return state.sol


def repair(
    ctx: DayContext,
    sol: DaySolution,
    rng: random.Random,
    mode: str = "greedy",
    max_inserts: int = 200,
) -> DaySolution:
    """Sửa nghiệm sau khi phá: chèn lại cho tới khi hết chỗ."""
    state = InsertionState(ctx, sol)
    inserts = 0
    coverage_first = mode == "coverage"
    regret = 3 if mode == "regret" else 0
    if coverage_first:
        while inserts < max_inserts:
            best = state.best_insertion(rng=rng, coverage_first=True)
            if best is None:
                break
            state.apply(*best)
            inserts += 1
    while inserts < max_inserts:
        best = state.best_insertion(rng=rng, regret=regret)
        if best is None:
            break
        state.apply(*best)
        inserts += 1
    return state.sol
