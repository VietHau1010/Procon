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

def build_hamiltonian_cycle(ctx):
    """Build optimal Hamiltonian cycle of all spots using Nearest Neighbor + 2-Opt."""
    table = ctx.dist_step
    n = len(ctx.spot_cells)
    if n <= 1:
        return list(range(n))
    
    # NN from spot 0
    cur = 0
    unvisited = set(range(n)) - {cur}
    cycle = [cur]
    while unvisited:
        cur_c = ctx.spot_cells[cur]
        nxt = min(unvisited, key=lambda s: table.cost(cur_c, ctx.spot_cells[s]))
        cycle.append(nxt)
        unvisited.remove(nxt)
        cur = nxt

    # 2-Opt
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

def construct_master_tour_solution(ctx):
    table = ctx.dist_step
    cycle = build_hamiltonian_cycle(ctx)
    n = len(cycle)
    spot_cells = ctx.spot_cells

    # Central hub for refuel car (e.g. 549 or spot with min sum dist to all spots)
    hub_spot = next((i for i, c in enumerate(spot_cells) if c == 549), None)
    if hub_spot is None:
        hub_spot = min(range(n), key=lambda s: sum(table.cost(spot_cells[s], spot_cells[o]) for o in range(n)))

    tours = []
    for slot in range(ctx.n_patrols):
        start_c = ctx.start_of_patrol(slot)
        # Entry spot
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

    # Ensure 100% brand coverage
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

    # Meets
    meets = []
    if ctx.n_cars > 0:
        for slot in range(ctx.n_patrols):
            if hub_spot in tours[slot]:
                idx = tours[slot].index(hub_spot)
                meets.append(Meet(patrol=slot, stop_idx=idx, car=0))
            elif len(tours[slot]) >= 4:
                idx = len(tours[slot]) // 2
                meets.append(Meet(patrol=slot, stop_idx=idx, car=0))

    sol = DaySolution(tours=tours, meets=meets, waits={})
    dec = decode(ctx, sol)
    return sol, dec

# Test across all 10 days!
kinds = [rules.PATROL, rules.REFUEL, rules.PATROL, rules.PATROL]
agents = [AgentState(kinds[i], cfg.agent_starts[i], cfg.fuel_limits) for i in range(cfg.n_agents)]

total_servings = 0
total_daily_types = 0
visited_match = set()

print("=== RUNNING MASTER CYCLE CONSTRUCT ACROSS ALL 10 DAYS ===")
for day in range(10):
    ctx = make_context(cfg, day, agents, {}, [s.stocks for s in cfg.spots], visited_match)
    sol, dec = construct_master_tour_solution(ctx)
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
