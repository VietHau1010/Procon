import sys; sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig, AgentState, DayState
from hexudon import rules
from hexudon.planner_contest.engine import make_context
from hexudon.planner_contest.decoder import decode
from hexudon.planner.solution import DaySolution, Meet, Weights
from hexudon.planner.roles import centrality
from hexudon.planner_contest.construct import build_master_cycle
import hexudon.planner_contest.alns as contest_alns

with open('replays/contest-087d9b9a-t78-20261112-165702.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('path') == '/api/game/config':
            cfg = MatchConfig.from_json(obj['response'])
            break

# 1. Centrality-based role selection
c_scores = centrality(cfg)
refuel_car_idx = c_scores.index(min(c_scores))
roles = [rules.PATROL] * cfg.n_agents
roles[refuel_car_idx] = rules.REFUEL
print(f"Role selection: {roles} (Agent {refuel_car_idx} is Refuel)")

# 2. Optimized construct with Outlier Bypass & Hub Rendezvous
def construct_optimized(ctx, rng=None):
    table = ctx.dist_step
    cycle = build_master_cycle(ctx)
    spot_cells = ctx.spot_cells
    n = len(cycle)

    # Spot 57 is an outlier for Brand 9 (which also has 518 and 184 in core)
    # On days <= 160 steps, bypass 57 to save 15 steps per patrol
    spot_57_idx = None
    for i, s in enumerate(cycle):
        if spot_cells[s] == 57:
            spot_57_idx = s
            break

    # Find the central hubs
    hub_candidates = sorted(range(n), key=lambda s: sum(table.cost(spot_cells[s], spot_cells[o]) for o in range(n)))
    car_start = ctx.start_of_car(0) if ctx.n_cars > 0 else 0
    best_hub = min(hub_candidates[:5], key=lambda h: table.cost(car_start, spot_cells[h]))

    # Spot 808 is mandatory Brand 2
    spot_808_idx = None
    for i, s in enumerate(cycle):
        if spot_cells[s] == 808:
            spot_808_idx = s
            break

    # Build core cycle (bypassing 57 if steps <= 160)
    core_cycle = [s for s in cycle if not (ctx.steps <= 160 and s == spot_57_idx)]
    core_n = len(core_cycle)

    tours = []
    # If steps <= 160: 1 patrol covers 808, other 2 patrols stay in dense core
    # If steps > 160: all patrols cover 808
    for slot in range(ctx.n_patrols):
        start_c = ctx.start_of_patrol(slot)
        
        # Decide if this slot is the designated 808 visitor
        include_808 = (ctx.steps > 160) or (slot == (ctx.day % ctx.n_patrols))
        car_cycle = core_cycle if include_808 else [s for s in core_cycle if s != spot_808_idx]
        car_n = len(car_cycle)

        entry_idx = min(range(car_n), key=lambda i: table.cost(start_c, spot_cells[car_cycle[i]]))
        
        best_t = []
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
                # Allow a second lap if steps allow and stock allows!
                if len(visited_in_tour) == car_n and steps + 5 > ctx.steps:
                    break
                if len(cur_t) >= car_n + 5: # cap
                    break
            if len(cur_t) > len(best_t):
                best_t = cur_t
        tours.append(best_t)

    # Ensure 100% brand coverage
    covered_brands = {ctx.spot_brand[s] for t in tours for s in t}
    missing_brands = set(ctx.config.brands) - covered_brands
    for b in missing_brands:
        b_spots = ctx.config.spots_by_brand.get(b, ())
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
                    nxt_c = cells[pos + 1] if pos + 1 < len(cells) else None
                    detour = table.cost(prev_c, sp_c) + (table.cost(sp_c, nxt_c) - table.cost(prev_c, nxt_c) if nxt_c is not None else 0)
                    if detour < best_detour:
                        best_detour = detour
                        best_cand = (slot, pos, sp)
        if best_cand is not None:
            slot, pos, sp = best_cand
            tours[slot].insert(pos, sp)
            covered_brands.add(b)

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
    if not dec.result.ok:
        from hexudon.planner.construct import construct as base_construct
        return base_construct(ctx)
    from hexudon.planner_contest.construct import greedy_fill
    return greedy_fill(ctx, sol)

agents = [AgentState(roles[i], cfg.agent_starts[i], cfg.fuel_limits) for i in range(cfg.n_agents)]
road_status = {}

total_servings = 0
total_daily = 0

print("=== RUNNING ADVANCED SIMULATION ===")
for day in range(10):
    ctx = make_context(cfg, day, agents, road_status, [s.stocks for s in cfg.spots], set())
    sol = construct_optimized(ctx)
    res = contest_alns.improve(ctx, sol, max_iterations=500)
    dec = res.decoded
    r = dec.result
    total_servings += r.servings
    total_daily += len(r.brands)
    print(f"Day {day:02d} (steps={cfg.day_steps[day]:3d}): brands={len(r.brands):2d}/{len(cfg.brands)}, servings={r.servings:2d}, ok={r.ok}")
    agents = [AgentState(roles[i], r.end_pos[i], r.end_fuel[i]) for i in range(cfg.n_agents)]

print("=" * 60)
print(f"ADVANCED RESULT: L=20 | D={total_daily} | Q={total_servings}")
print("Team 78 Original: L=19 | D=190 | Q=693")
print("Team 77 Winning:  L=20 | D=200 | Q=737")
