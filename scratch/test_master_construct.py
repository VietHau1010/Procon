import sys; sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig, AgentState, DayState
from hexudon import rules
from hexudon.planner_contest.engine import make_context
from hexudon.planner.decoder import decode, DaySolution
from hexudon.planner.solution import Meet, Weights
from hexudon.planner.construct import InsertionState

with open('replays/contest-087d9b9a-t78-20261112-165702.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('path') == '/api/game/config':
            cfg = MatchConfig.from_json(obj['response'])
            break

def build_master_cycle(ctx):
    """Build a closed TSP cycle through all spots using Nearest-Neighbor + 2-Opt."""
    table = ctx.dist_step
    all_spots = list(range(len(ctx.spot_cells)))
    if not all_spots:
        return []
    
    # NN from spot 0
    cur = 0
    unvisited = set(all_spots) - {cur}
    cycle = [cur]
    while unvisited:
        cur_cell = ctx.spot_cells[cur]
        nxt = min(unvisited, key=lambda s: table.cost(cur_cell, ctx.spot_cells[s]))
        cycle.append(nxt)
        unvisited.remove(nxt)
        cur = nxt

    # 2-Opt
    n = len(cycle)
    def cycle_cost(t):
        return sum(table.cost(ctx.spot_cells[t[i]], ctx.spot_cells[t[(i+1)%n]]) for i in range(n))
    
    cur_c = cycle_cost(cycle)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = cycle[:i] + cycle[i:j+1][::-1] + cycle[j+1:]
                c = cycle_cost(cand)
                if c < cur_c:
                    cur_c = c
                    cycle = cand
                    improved = True
                    break
            if improved:
                break
    return cycle

def construct_master_platoon(ctx):
    table = ctx.dist_step
    cycle = build_master_cycle(ctx)
    n = len(cycle)
    spot_cells = ctx.spot_cells

    # Central hub for refuel car
    hub_spot = next((i for i, c in enumerate(spot_cells) if c == 549), None)
    if hub_spot is None:
        hub_spot = min(range(len(spot_cells)), key=lambda s: sum(table.cost(spot_cells[s], spot_cells[o]) for o in range(len(spot_cells))))

    tours = []
    for slot in range(ctx.n_patrols):
        start_c = ctx.start_of_patrol(slot)
        # Entry spot
        entry_idx = min(range(n), key=lambda i: table.cost(start_c, spot_cells[cycle[i]]))
        
        # Traverse cycle in both directions, pick the one with most spots within steps & allowance
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
        # Find best patrol and best position to insert a spot of brand b
        best_cand = None
        best_detour = 999999
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

    # Schedule meets
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

agents = [AgentState(rules.PATROL, cfg.agent_starts[i], cfg.fuel_limits) for i in range(cfg.n_agents)]
agents[1] = AgentState(rules.REFUEL, cfg.agent_starts[1], cfg.fuel_limits)
ctx = make_context(cfg, 0, agents, {}, [s.stocks for s in cfg.spots], set())

sol, dec = construct_master_platoon(ctx)
print(f"Master Platoon Day 0: ok={dec.result.ok}, brands={len(dec.result.brands)}, servings={dec.result.servings}, score={dec.score}")
for slot in range(ctx.n_patrols):
    print(f"Patrol {slot}: spots={len(sol.tours[slot])}, cells={[ctx.spot_cells[s] for s in sol.tours[slot]]}")
