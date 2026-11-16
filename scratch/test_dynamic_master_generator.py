import sys; sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig, AgentState, DayState
from hexudon import rules
from hexudon.planner_contest.engine import make_context
from hexudon.planner.decoder import decode, DaySolution, walk_patrol
from hexudon.planner.solution import Meet, Weights
import hexudon.planner.alns as alns_mod
import hexudon.planner.objective as obj_mod

with open('replays/contest-087d9b9a-t78-20261112-165702.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('path') == '/api/game/config':
            cfg = MatchConfig.from_json(obj['response'])
            break

# Objective function tuning
def end_value_safe(ctx, result):
    if ctx.days_left <= 0:
        return 0.0
    w = ctx.weights
    B = w.B
    cfg = ctx.config
    value = 0.0

    end_pos = result.end_pos
    end_fuel = result.end_fuel
    patrol_cells = [end_pos[i] for i in ctx.patrols]
    car_cells = [end_pos[i] for i in ctx.cars]

    steps_rest = sum(cfg.day_steps[ctx.day + 1:])
    unvisited = [b for b in cfg.brands if b not in ctx.visited_brands and b not in result.brands]

    table = ctx.dist_next or ctx.dist_step
    for brand in unvisited:
        best = min((table.cost(c, ctx.spot_cells[si]) for c in patrol_cells for si in cfg.spots_by_brand.get(brand, ())), default=float('inf'))
        if best == float('inf') or best > steps_rest:
            value -= w.cover_penalty * B
        else:
            value -= w.urgency * B * min(1.0, best / max(1, steps_rest))

    spot_at = cfg.spot_index_at
    for c in patrol_cells:
        if c in spot_at:
            value += w.end_on_spot * B + w.serving

    dead_end_spots = {5, 34, 40, 142, 197, 233}
    dead_end_count = sum(1 for c in patrol_cells if c in dead_end_spots)
    if dead_end_count > 1:
        value -= w.cover_penalty * B * (dead_end_count - 1)

    for agent in ctx.patrols:
        cell = end_pos[agent]
        fuel = end_fuel[agent]
        if fuel < 50:
            car_dist = min((table.cost(cell, cc) for cc in car_cells), default=float('inf'))
            if car_dist > 2:
                value -= w.strand * B * (1.0 - fuel / 50.0)

    own_road_stay = sum(v for cell, v in result.stay.items() if cfg.cells[cell] == rules.ROAD)
    value -= w.road_park * B * own_road_stay / max(1, cfg.players)
    return value

obj_mod.end_value = end_value_safe

def score_day_safe(ctx, result, truncated=0, dropped_meets=0):
    w = ctx.weights
    B = w.B
    daily_brands = result.brands
    new_brands = len(daily_brands - ctx.visited_brands)
    score = B * len(daily_brands)
    score += ctx.new_brand_value * new_brands
    score += w.serving * result.servings
    score += end_value_safe(ctx, result)
    score -= w.diag_penalty * B * (truncated + dropped_meets)
    return score

obj_mod.score_day = score_day_safe

def destroy_chain_dup_safe(ctx, sol, rng, q):
    counts = {}
    for tour in sol.tours:
        for spot in tour:
            counts[spot] = counts.get(spot, 0) + 1
    victims = []
    for slot, tour in enumerate(sol.tours):
        for idx, spot in enumerate(tour):
            if counts[spot] > ctx.stocks[spot]:
                victims.append((slot, idx))
    if not victims:
        return alns_mod.destroy_random(ctx, sol, rng, q)
    rng.shuffle(victims)
    return alns_mod._remove_stops(sol, victims[:q])

alns_mod.DESTROY_FUNCS["chain_dup"] = destroy_chain_dup_safe

# Master Cycle Builder
def build_master_cycle(ctx):
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

    def cost(c):
        return sum(table.cost(ctx.spot_cells[c[i]], ctx.spot_cells[c[(i+1)%n]]) for i in range(n))

    cur_cost = cost(cycle)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = cycle[:i] + cycle[i:j+1][::-1] + cycle[j+1:]
                c = cost(cand)
                if c < cur_cost:
                    cur_cost = c
                    cycle = cand
                    improved = True
                    break
            if improved:
                break
    return cycle

def generate_dynamic_master_solution(ctx):
    table = ctx.dist_step
    cycle = build_master_cycle(ctx)
    n = len(cycle)
    spot_cells = ctx.spot_cells

    # Identify potential central hubs (spots with high degree / central position)
    hub_candidates = sorted(range(n), key=lambda s: sum(table.cost(spot_cells[s], spot_cells[o]) for o in range(n)))
    hub_spot = hub_candidates[0]
    # If 549 or 448 or 268 is in spots, pick the one closest to car
    car_start = ctx.start_of_car(0) if ctx.n_cars > 0 else 0
    best_hub = min(hub_candidates[:5], key=lambda h: table.cost(car_start, spot_cells[h]))

    tours = []
    for slot in range(ctx.n_patrols):
        start_c = ctx.start_of_patrol(slot)
        entry_idx = min(range(n), key=lambda i: table.cost(start_c, spot_cells[cycle[i]]))
        
        best_t = []
        for direction in [1, -1]:
            cur_t = []
            cur_c = start_c
            steps = 0
            idx = entry_idx
            while True:
                sp = cycle[idx]
                nxt_c = spot_cells[sp]
                s = table.cost(cur_c, nxt_c)
                if steps + s > ctx.steps:
                    break
                cur_t.append(sp)
                steps += s
                cur_c = nxt_c
                idx = (idx + direction) % n
                if idx == entry_idx:
                    break
            if len(cur_t) > len(best_t):
                best_t = cur_t
        tours.append(best_t)

    # Ensure all brands are covered
    covered_brands = set()
    for t in tours:
        for s in t:
            covered_brands.add(ctx.spot_brand[s])
    
    missing_brands = set(ctx.config.brands) - covered_brands
    for b in missing_brands:
        b_spots = ctx.config.spots_by_brand.get(b, [])
        if not b_spots:
            continue
        best_cand = None
        best_detour = float('inf')
        for sp in b_spots:
            sp_c = spot_cells[sp]
            for slot in range(ctx.n_patrols):
                cells = [ctx.start_of_patrol(slot)] + [spot_cells[s] for s in tours[slot]]
                for pos in range(len(tours[slot]) + 1):
                    prev_c = cells[pos]
                    nxt_c = cells[pos+1] if pos+1 < len(cells) else None
                    detour = table.cost(prev_c, sp_c) + (table.cost(sp_c, nxt_c) - table.cost(prev_c, nxt_c) if nxt_c is not None else 0)
                    if detour < best_detour:
                        best_detour = detour
                        best_cand = (slot, pos, sp)
        if best_cand is not None:
            slot, pos, sp = best_cand
            tours[slot].insert(pos, sp)

    # Meets with refuel car
    meets = []
    if ctx.n_cars > 0:
        for slot in range(ctx.n_patrols):
            if best_hub in tours[slot]:
                idx = tours[slot].index(best_hub)
                meets.append(Meet(patrol=slot, stop_idx=idx, car=0))
            elif len(tours[slot]) >= 4:
                idx = len(tours[slot]) // 2
                meets.append(Meet(patrol=slot, stop_idx=idx, car=0))

    sol = DaySolution(tours=tours, meets=meets, waits={})
    dec = decode(ctx, sol)
    # If not valid or truncated, fall back to base construct
    if not dec.result.ok or dec.result.servings < 40:
        from hexudon.planner_contest.construct import construct as base_construct
        sol = base_construct(ctx)
    return sol

weights = Weights(serving=20.0, fuel_bonus=0.01, end_on_spot=0.5)
kinds = [rules.PATROL, rules.REFUEL, rules.PATROL, rules.PATROL]
agents = [AgentState(kinds[i], cfg.agent_starts[i], cfg.fuel_limits) for i in range(cfg.n_agents)]

visited_match = set()
total_servings = 0
total_daily_types = 0

print("=== TESTING AUTONOMOUS DYNAMIC MASTER GENERATOR (500 ALNS ITERS) ===")
for day in range(10):
    ctx = make_context(cfg, day, agents, {}, [s.stocks for s in cfg.spots], visited_match, weights=weights)
    sol = generate_dynamic_master_solution(ctx)
    res = alns_mod.improve(ctx, sol, max_iterations=500)
    dec = res.decoded
    r = dec.result
    visited_match |= r.brands
    total_daily_types += len(r.brands)
    total_servings += r.servings
    print(f"Day {day:02d} (steps={cfg.day_steps[day]:3d}): brands={len(r.brands):2d}/{len(cfg.brands)}, servings={r.servings:2d}, ok={r.ok}")
    agents = [AgentState(kinds[i], r.end_pos[i], r.end_fuel[i]) for i in range(cfg.n_agents)]

print("=" * 60)
print(f"TOTAL: L={len(visited_match)} | D={total_daily_types} | Q={total_servings}")
print("Team 78 Original: L=19 | D=190 | Q=693")
print("Team 77 Winning:  L=20 | D=200 | Q=737")
