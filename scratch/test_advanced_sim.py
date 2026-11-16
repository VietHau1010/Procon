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

# 1. Custom end_value with Fuel Safety Threshold (>= 50 fuel required)
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
        if fuel < 60:
            car_dist = min((table.cost(cell, cc) for cc in car_cells), default=float('inf'))
            if car_dist > 2:
                value -= w.strand * B * (1.0 - fuel / 60.0)

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

# Improved construct with Multi-Stop Fuel Meets
from hexudon.planner_contest.construct import InsertionState

def two_opt_tour(ctx, slot, tour):
    if len(tour) < 3:
        return tour
    cells = [ctx.start_of_patrol(slot)] + [ctx.spot_cells[s] for s in tour]
    best_tour = list(tour)
    table = ctx.dist_step
    improved = True
    while improved:
        improved = False
        n = len(best_tour)
        for i in range(1, n):
            for j in range(i + 1, n + 1):
                c_prev = cells[i - 1]
                c_i = cells[i]
                c_j_minus_1 = cells[j]
                c_j = cells[j + 1] if j + 1 < len(cells) else None
                cur_dist = table.cost(c_prev, c_i) + (table.cost(c_j_minus_1, c_j) if c_j is not None else 0)
                new_dist = table.cost(c_prev, c_j_minus_1) + (table.cost(c_i, c_j) if c_j is not None else 0)
                if new_dist < cur_dist:
                    best_tour[i-1:j] = reversed(best_tour[i-1:j])
                    cells = [ctx.start_of_patrol(slot)] + [ctx.spot_cells[s] for s in best_tour]
                    improved = True
                    break
            if improved:
                break
    return best_tour

def advanced_fuel_meets(ctx, sol):
    if ctx.n_cars == 0:
        return sol
    current = sol.copy()
    base = decode(ctx, current).score

    for slot in range(ctx.n_patrols):
        if not current.tours[slot]:
            continue
        meet_map = {m.stop_idx: m.car for m in current.meets if m.patrol == slot}
        walk = walk_patrol(ctx, current, slot, meet_map, current.waits)
        
        # Test stops where fuel is between 20 and 120 (sweet spot for refueling)
        stops = sorted(i for i in walk.arrivals if i >= 0)
        # Prioritize stops in the middle of tour
        ranked_stops = sorted(stops, key=lambda idx: abs(idx - len(current.tours[slot]) // 2))
        
        for stop_idx in ranked_stops:
            if stop_idx in meet_map:
                continue
            cell = walk.stop_cell.get(stop_idx, ctx.start_of_patrol(slot))
            for car in range(ctx.n_cars):
                trial = current.copy()
                trial.meets.append(Meet(patrol=slot, stop_idx=stop_idx, car=car))
                dec = decode(ctx, trial)
                if dec.result.ok and dec.score > base:
                    current = trial
                    base = dec.score
                    break
            if len(current.meets) > len(sol.meets):
                break
    return current

def construct_advanced(ctx, rng=None):
    from hexudon.planner_contest.construct import construct as base_construct
    sol = base_construct(ctx, rng=rng)
    
    # 2-Opt on all tours
    for slot in range(ctx.n_patrols):
        sol.tours[slot] = two_opt_tour(ctx, slot, sol.tours[slot])
    
    # Add advanced meets
    sol = advanced_fuel_meets(ctx, sol)
    
    # Greedy fill
    state = InsertionState(ctx, sol)
    for _ in range(60):
        best = state.best_insertion(rng=rng)
        if best is None:
            break
        state.apply(*best)
    
    return advanced_fuel_meets(ctx, state.sol)

weights = Weights(serving=20.0, fuel_bonus=0.01, end_on_spot=0.5)
kinds = [rules.PATROL, rules.REFUEL, rules.PATROL, rules.PATROL]
agents = [AgentState(kinds[i], cfg.agent_starts[i], cfg.fuel_limits) for i in range(cfg.n_agents)]

visited_match = set()
total_servings = 0
total_daily_types = 0

print("=== SIMULATING FULL 10 DAYS WITH ADVANCED CONSTRUCT + ALNS (500 iters) ===")
for day in range(cfg.n_days):
    ctx = make_context(cfg, day, agents, {}, [s.stocks for s in cfg.spots], visited_match, weights=weights)
    sol = construct_advanced(ctx)
    res = alns_mod.improve(ctx, sol, max_iterations=500)
    dec = res.decoded
    r = dec.result
    visited_match |= r.brands
    total_daily_types += len(r.brands)
    total_servings += r.servings
    print(f"Day {day:02d} (steps={cfg.day_steps[day]:3d}): brands={len(r.brands):2d}/{len(cfg.brands)}, servings={r.servings:2d}, ok={r.ok}, end_fuel={[r.end_fuel[i] for i in [0, 2, 3]]}")
    agents = [AgentState(kinds[i], r.end_pos[i], r.end_fuel[i]) for i in range(cfg.n_agents)]

print("=" * 60)
print(f"FINAL RESULT: L={len(visited_match)}/{len(cfg.brands)} | D={total_daily_types} | Q={total_servings}")
print("Team 78 Original: L=19 | D=190 | Q=693")
print("Team 77 Winning:  L=20 | D=200 | Q=737")
