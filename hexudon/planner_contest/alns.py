"""ALNS thích ứng cho chiến thuật contest:
- Sử dụng destroy_chain_dup_safe (chỉ xoá khi kho udon đã cạn, bảo toàn platooning).
- Kết nối trực tiếp với contest construct, repair và decoder.
- Chạy anytime, luôn trả về nghiệm hợp lệ với điểm tối ưu.
"""

import random
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..planner.solution import DayContext, DaySolution, Meet
from .construct import construct, repair
from .decoder import Decoded, decode
from .objective import coverage_key

DESTROY_OPS = ("random", "zero_gain", "related", "chain_dup", "tail", "meet_toggle")
REPAIR_OPS = ("greedy", "regret", "coverage")

REWARD_BEST = 8.0
REWARD_BETTER = 4.0
REWARD_ACCEPT = 1.0
WEIGHT_DECAY = 0.8
WEIGHT_FLOOR = 0.1
ADAPT_EVERY = 50
RRT_BAND = 0.25
RESTART_AFTER = 500


def _remove_stops(sol: DaySolution, victims: Sequence[Tuple[int, int]]) -> DaySolution:
    """Bỏ các điểm dừng (slot, chỉ số); điểm hẹn và lệnh đợi đi theo điểm dừng của chúng."""
    by_patrol: Dict[int, set] = {}
    for slot, idx in victims:
        by_patrol.setdefault(slot, set()).add(idx)

    new_tours: List[List[int]] = []
    remap: Dict[Tuple[int, int], int] = {}
    for slot, tour in enumerate(sol.tours):
        v = by_patrol.get(slot, set())
        cur = []
        for idx, spot in enumerate(tour):
            if idx in v:
                remap[(slot, idx)] = -1
            else:
                remap[(slot, idx)] = len(cur)
                cur.append(spot)
        new_tours.append(cur)

    new_meets = []
    for m in sol.meets:
        if m.stop_idx < 0:
            new_meets.append(m)
            continue
        new_idx = remap.get((m.patrol, m.stop_idx), -1)
        if new_idx >= 0:
            new_meets.append(Meet(m.patrol, new_idx, m.car))

    new_waits = {}
    for (s, idx), w in sol.waits.items():
        if idx < 0:
            # Start-of-day refuel waits have no tour stop to remap. Dropping
            # them can remove the only fuel rescue for a stranded patrol.
            new_waits[(s, idx)] = w
            continue
        new_idx = remap.get((s, idx), -1)
        if new_idx >= 0:
            new_waits[(s, new_idx)] = w
    return DaySolution(tours=new_tours, meets=new_meets, waits=new_waits)


def destroy_random(ctx: DayContext, sol: DaySolution, rng: random.Random, q: int) -> DaySolution:
    stops = sol.stops()
    if not stops:
        return sol
    return _remove_stops(sol, rng.sample(stops, min(q, len(stops))))


def destroy_zero_gain(ctx: DayContext, sol: DaySolution, rng: random.Random, q: int) -> DaySolution:
    counts: Dict[int, int] = {}
    for tour in sol.tours:
        for spot in tour:
            counts[spot] = counts.get(spot, 0) + 1
    zero_gain = []
    for slot, tour in enumerate(sol.tours):
        for idx, spot in enumerate(tour):
            if counts[spot] > ctx.stocks[spot]:
                zero_gain.append((slot, idx))
    if not zero_gain:
        return destroy_random(ctx, sol, rng, q)
    rng.shuffle(zero_gain)
    return _remove_stops(sol, zero_gain[:q])


def destroy_related(ctx: DayContext, sol: DaySolution, rng: random.Random, q: int) -> DaySolution:
    stops = sol.stops()
    if not stops:
        return sol
    seed_slot, seed_idx = rng.choice(stops)
    seed_cell = ctx.spot_cells[sol.tours[seed_slot][seed_idx]]
    scored = []
    for slot, idx in stops:
        cell = ctx.spot_cells[sol.tours[slot][idx]]
        scored.append((ctx.dist_step.cost(seed_cell, cell), slot, idx))
    scored.sort()
    victims = [(slot, idx) for _, slot, idx in scored[:q]]
    return _remove_stops(sol, victims)


def destroy_chain_dup_safe(ctx: DayContext, sol: DaySolution, rng: random.Random, q: int) -> DaySolution:
    """Chỉ bỏ điểm dừng nếu quán đó đã được ghé thăm vượt quá lượng kho dự trữ."""
    counts: Dict[int, int] = {}
    for tour in sol.tours:
        for spot in tour:
            counts[spot] = counts.get(spot, 0) + 1
    victims = []
    for slot, tour in enumerate(sol.tours):
        for idx, spot in enumerate(tour):
            if counts[spot] > ctx.stocks[spot]:
                victims.append((slot, idx))
    if not victims:
        return destroy_random(ctx, sol, rng, q)
    rng.shuffle(victims)
    return _remove_stops(sol, victims[:q])


def destroy_tail(ctx: DayContext, sol: DaySolution, rng: random.Random, q: int) -> DaySolution:
    non_empty = [s for s, t in enumerate(sol.tours) if t]
    if not non_empty:
        return sol
    slot = rng.choice(non_empty)
    k = min(q, len(sol.tours[slot]))
    victims = [(slot, len(sol.tours[slot]) - 1 - i) for i in range(k)]
    return _remove_stops(sol, victims)


def destroy_meet_toggle(ctx: DayContext, sol: DaySolution, rng: random.Random, q: int) -> DaySolution:
    if ctx.n_cars == 0:
        return destroy_random(ctx, sol, rng, q)
    sol = sol.copy()
    if sol.meets and rng.random() < 0.5:
        m = rng.choice(sol.meets)
        sol.meets.remove(m)
        if rng.random() < 0.5 and ctx.n_cars > 1:
            other = rng.randrange(ctx.n_cars)
            sol.meets.append(Meet(m.patrol, m.stop_idx, other))
        return sol
    candidates = []
    for slot, tour in enumerate(sol.tours):
        for idx in range(-1, len(tour)):
            if not any(m.patrol == slot and m.stop_idx == idx for m in sol.meets):
                candidates.append((slot, idx))
    if not candidates:
        return sol
    slot, idx = rng.choice(candidates)
    sol.meets.append(Meet(slot, idx, rng.randrange(ctx.n_cars)))
    return sol


DESTROY_FUNCS = {
    "random": destroy_random,
    "zero_gain": destroy_zero_gain,
    "related": destroy_related,
    "chain_dup": destroy_chain_dup_safe,
    "tail": destroy_tail,
    "meet_toggle": destroy_meet_toggle,
}


class Roulette:
    """Trọng số thích nghi cho các toán tử."""

    def __init__(self, names: Sequence[str]):
        self.names = list(names)
        self.weight = {n: 1.0 for n in self.names}
        self.reward = {n: 0.0 for n in self.names}
        self.uses = {n: 0 for n in self.names}

    def pick(self, rng: random.Random) -> str:
        total = sum(self.weight.values())
        r = rng.random() * total
        acc = 0.0
        for name in self.names:
            acc += self.weight[name]
            if r <= acc:
                return name
        return self.names[-1]

    def award(self, name: str, value: float):
        self.reward[name] += value
        self.uses[name] += 1

    def adapt(self):
        for name in self.names:
            if self.uses[name] == 0:
                continue
            mean = self.reward[name] / self.uses[name]
            self.weight[name] = max(
                WEIGHT_FLOOR, WEIGHT_DECAY * self.weight[name] + (1 - WEIGHT_DECAY) * mean
            )
            self.reward[name] = 0.0
            self.uses[name] = 0


class AlnsResult:
    __slots__ = ("solution", "decoded", "iterations", "seconds")

    def __init__(self, solution: DaySolution, decoded: Decoded, iterations: int, seconds: float):
        self.solution = solution
        self.decoded = decoded
        self.iterations = iterations
        self.seconds = seconds


def improve(
    ctx: DayContext,
    start: Optional[DaySolution] = None,
    deadline: Optional[float] = None,
    max_iterations: int = 1_000_000,
    seed: int = 0,
    on_better: Optional[Callable[[DaySolution, Decoded], None]] = None,
    on_progress: Optional[Callable[[dict], None]] = None,
    progress_every: float = 0.25,
) -> AlnsResult:
    """Chạy ALNS tới hạn chót hoặc số vòng lặp tối đa. Luôn trả về một nghiệm hợp lệ."""
    rng = random.Random((seed << 8) ^ (ctx.day + 1))
    began = time.monotonic()
    cur = start.copy() if start is not None else construct(ctx, rng)
    cur_dec = decode(ctx, cur)
    best, best_dec = cur.copy(), cur_dec
    coverage_sol, coverage_dec = cur.copy(), cur_dec
    if on_better is not None:
        on_better(best, best_dec)

    destroy = Roulette(DESTROY_OPS)
    repair_ops = Roulette(REPAIR_OPS)
    since_best = 0
    it = 0
    max_stops = max(1, ctx.config.n_agents)
    next_progress = began + progress_every

    def report() -> None:
        if on_progress is not None:
            on_progress(
                {
                    "iteration": it,
                    "elapsed": round(time.monotonic() - began, 3),
                    "current": round(cur_dec.score, 3),
                    "best": round(best_dec.score, 3),
                    "destroy": {name: round(w, 3) for name, w in destroy.weight.items()},
                    "repair": {name: round(w, 3) for name, w in repair_ops.weight.items()},
                }
            )

    while it < max_iterations:
        if deadline is not None and time.monotonic() >= deadline:
            break
        it += 1
        d_name = destroy.pick(rng)
        r_name = repair_ops.pick(rng)
        q = rng.randint(1, max(1, min(4, max_stops)))
        cand = DESTROY_FUNCS[d_name](ctx, cur.copy(), rng, q)
        cand = repair(ctx, cand, rng, mode=r_name)
        cand_dec = decode(ctx, cand)
        s = cand_dec.score
        if coverage_key(ctx, cand_dec) > coverage_key(ctx, coverage_dec):
            coverage_sol, coverage_dec = cand.copy(), cand_dec
            if on_better is not None:
                on_better(coverage_sol, coverage_dec)

        if s > best_dec.score:
            best, best_dec = cand.copy(), cand_dec
            cur, cur_dec = cand, cand_dec
            destroy.award(d_name, REWARD_BEST)
            repair_ops.award(r_name, REWARD_BEST)
            since_best = 0
            if on_better is not None:
                on_better(best, best_dec)
        elif s > cur_dec.score:
            cur, cur_dec = cand, cand_dec
            destroy.award(d_name, REWARD_BETTER)
            repair_ops.award(r_name, REWARD_BETTER)
            since_best += 1
        elif s >= best_dec.score - RRT_BAND * ctx.weights.B:
            cur, cur_dec = cand, cand_dec
            destroy.award(d_name, REWARD_ACCEPT)
            repair_ops.award(r_name, REWARD_ACCEPT)
            since_best += 1
        else:
            destroy.award(d_name, 0.0)
            repair_ops.award(r_name, 0.0)
            since_best += 1

        if it % ADAPT_EVERY == 0:
            destroy.adapt()
            repair_ops.adapt()
        if since_best >= RESTART_AFTER:
            cur = destroy_random(ctx, best.copy(), rng, max(2, max_stops // 2))
            cur = repair(ctx, cur, rng, mode="coverage")
            cur_dec = decode(ctx, cur)
            if coverage_key(ctx, cur_dec) > coverage_key(ctx, coverage_dec):
                coverage_sol, coverage_dec = cur.copy(), cur_dec
                if on_better is not None:
                    on_better(coverage_sol, coverage_dec)
            since_best = 0

        if on_progress is not None and time.monotonic() >= next_progress:
            next_progress = time.monotonic() + progress_every
            report()

    if on_progress is not None:
        report()
    return AlnsResult(coverage_sol, coverage_dec, it, time.monotonic() - began)
