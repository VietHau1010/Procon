import sys; sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig, AgentState, DayState
from hexudon import rules
from hexudon.planner_contest.engine import make_context
from hexudon.planner.decoder import decode, DaySolution
from hexudon.planner.solution import Meet, Weights
import hexudon.planner.alns as alns_mod
import hexudon.planner.objective as obj_mod

with open('replays/contest-087d9b9a-t78-20261112-165702.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('path') == '/api/game/config':
            cfg = MatchConfig.from_json(obj['response'])
            break

# Load T77 reference tours as upper benchmark
with open('C:/Users/LAPTOP/.gemini/antigravity/brain/6cbcc38d-2ad3-4140-b3c5-3d9adeac5e23/.user_uploaded/media_1794716321428_8a2b454c.json') as f:
    replay_t77 = json.load(f)

# Extract day-by-day reference solutions
spot_map = {s.pos: i for i, s in enumerate(cfg.spots)}

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

weights = Weights(serving=20.0, fuel_bonus=0.01, end_on_spot=0.5)

print("Testing T77 seed + ALNS improve across all 10 days:")
total_servings = 0
visited_match = set()

for day in range(10):
    d_obj = replay_t77['days'][day]
    t77 = next(t for t in d_obj['teams'] if str(t['team_id']) == '77')
    patrol_spots = []
    for aid in [0, 2, 3]:
        cells = []
        for fr in t77['frames']:
            colls = fr.get('collected', [])
            c = fr['agents'][aid]['cell']
            if c in colls and (not cells or cells[-1] != c):
                cells.append(c)
        tour = [spot_map[c] for c in cells if c in spot_map]
        patrol_spots.append(tour)
    
    meets = []
    for aid_idx, aid in enumerate([0, 2, 3]):
        for fr_idx, fr in enumerate(t77['frames']):
            if fr_idx > 0 and fr['agents'][aid]['fuel'] > t77['frames'][fr_idx-1]['agents'][aid]['fuel']:
                r_cell = fr['agents'][aid]['cell']
                if r_cell in [cfg.spots[s].pos for s in patrol_spots[aid_idx]]:
                    stop_idx = [cfg.spots[s].pos for s in patrol_spots[aid_idx]].index(r_cell)
                    meets.append(Meet(patrol=aid_idx, stop_idx=stop_idx, car=0))
    
    start_frame = t77['frames'][0]
    agents = [AgentState(rules.PATROL if a['type'] == 'patrol' else rules.REFUEL, a['cell'], a['fuel']) for a in start_frame['agents']]
    ctx = make_context(cfg, day, agents, {}, [s.stocks for s in cfg.spots], visited_match, weights=weights)
    sol_init = DaySolution(tours=patrol_spots, meets=meets, waits={})
    res = alns_mod.improve(ctx, sol_init, max_iterations=200)
    dec = res.decoded
    r = dec.result
    visited_match |= r.brands
    total_servings += r.servings
    print(f"Day {day:02d}: serv={r.servings}/{t77['servings']}, brands={len(r.brands)}/20, ok={r.ok}")

print(f"TOTAL SERVINGS: {total_servings} / 737")
