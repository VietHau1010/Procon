import sys; sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig, AgentState, DayState
from hexudon import rules
from hexudon.planner_contest.construct import build_master_cycle, trim_excess_stock, greedy_fill
from hexudon.planner_contest.engine import make_context
from hexudon.planner_contest.decoder import decode
from hexudon.planner.solution import DaySolution, Meet
from hexudon.planner.roles import centrality

with open('replays/contest-087d9b9a-t78-20261112-165702.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('path') == '/api/game/config':
            cfg = MatchConfig.from_json(obj['response'])
            break

c_scores = centrality(cfg)
best_refuel_idx = c_scores.index(min(c_scores))
roles = [rules.PATROL] * cfg.n_agents
roles[best_refuel_idx] = rules.REFUEL
agents = [AgentState(roles[i], cfg.agent_starts[i], cfg.fuel_limits) for i in range(cfg.n_agents)]

def build_test_solution(ctx):
    table = ctx.dist_step
    cycle = build_master_cycle(ctx)
    n = len(cycle)
    spot_cells = ctx.spot_cells
    cycle_detours = {}
    for i, sp in enumerate(cycle):
        p_c = spot_cells[cycle[(i - 1) % n]]
        s_c = spot_cells[sp]
        n_c = spot_cells[cycle[(i + 1) % n]]
        d = table.cost(p_c, s_c) + table.cost(s_c, n_c) - table.cost(p_c, n_c)
        cycle_detours[sp] = d

    redundant_outliers = {sp for sp in cycle if len(ctx.config.spots_by_brand.get(ctx.spot_brand[sp], ())) >= 2 and cycle_detours[sp] >= 12}
    solitary_outliers = {sp for sp in cycle if len(ctx.config.spots_by_brand.get(ctx.spot_brand[sp], ())) == 1 and cycle_detours[sp] >= 20}

    core_cycle = [sp for sp in cycle if sp not in solitary_outliers and not (ctx.steps <= 160 and sp in redundant_outliers)] or cycle
    M = len(core_cycle)

    sp_out = next(iter(solitary_outliers)) if solitary_outliers else None
    near_outliers = [s for s in range(ctx.n_patrols) if sp_out is not None and table.cost(ctx.start_of_patrol(s), spot_cells[sp_out]) <= 5]

    if sp_out is not None and near_outliers:
        slot_out = near_outliers[0]
    elif sp_out is not None:
        slot_out = min(range(ctx.n_patrols), key=lambda s: table.cost(ctx.start_of_patrol(s), spot_cells[sp_out]))
    else:
        slot_out = -1

    tours = []
    for slot in range(ctx.n_patrols):
        start_c = ctx.start_of_patrol(slot)
        if sp_out is not None and slot == slot_out:
            if near_outliers:
                # Starts at outlier
                entry_idx = min(range(M), key=lambda i: table.cost(spot_cells[sp_out], spot_cells[core_cycle[i]]))
                best_t = []
                for d in [1, -1]:
                    cur_t = [sp_out]
                    cur_c = spot_cells[sp_out]
                    steps = table.cost(start_c, cur_c)
                    idx = entry_idx
                    for _ in range(M + 5):
                        nxt = core_cycle[idx]
                        nxt_c = spot_cells[nxt]
                        s = table.cost(cur_c, nxt_c)
                        if steps + s > ctx.steps:
                            break
                        cur_t.append(nxt)
                        steps += s
                        cur_c = nxt_c
                        idx = (idx + d) % M
                    if len(cur_t) > len(best_t):
                        best_t = cur_t
                tours.append(best_t)
            else:
                # Ends at outlier
                best_t = []
                best_slack = 9999
                for c_last_idx in range(M):
                    c_last = core_cycle[c_last_idx]
                    dist_to_out = table.cost(spot_cells[c_last], spot_cells[sp_out])
                    for d in [1, -1]:
                        for k in range(1, M + 1):
                            cand_core = [core_cycle[(c_last_idx - (k - 1 - i) * d) % M] for i in range(k)]
                            cost = table.cost(start_c, spot_cells[cand_core[0]])
                            for i in range(k - 1):
                                cost += table.cost(spot_cells[cand_core[i]], spot_cells[cand_core[i+1]])
                            cost += dist_to_out
                            if cost <= ctx.steps:
                                slack = ctx.steps - cost
                                if len(cand_core) + 1 > len(best_t) or (len(cand_core) + 1 == len(best_t) and slack < best_slack):
                                    best_t = cand_core + [sp_out]
                                    best_slack = slack
                tours.append(best_t)
        else:
            # Core cycle traversal
            entry_idx = min(range(M), key=lambda i: table.cost(start_c, spot_cells[core_cycle[i]]))
            best_t = []
            for d in [1, -1]:
                cur_t = []
                cur_c = start_c
                steps = 0
                idx = entry_idx
                for _ in range(M + 5):
                    nxt = core_cycle[idx]
                    nxt_c = spot_cells[nxt]
                    s = table.cost(cur_c, nxt_c)
                    if steps + s > ctx.steps:
                        break
                    cur_t.append(nxt)
                    steps += s
                    cur_c = nxt_c
                    idx = (idx + d) % M
                if len(cur_t) > len(best_t):
                    best_t = cur_t
            tours.append(best_t)

    # Brand Coverage guarantee
    covered_brands = {ctx.spot_brand[s] for t in tours for s in t}
    missing_brands = set(ctx.config.brands) - covered_brands
    for b in missing_brands:
        b_spots = ctx.config.spots_by_brand.get(b, ())
        if not b_spots:
            continue
        best_cand = None
        best_detour = 999999
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

    # Hub Refuel
    hub_candidates = sorted(
        range(n),
        key=lambda s: sum(table.cost(spot_cells[s], spot_cells[o]) for o in range(n)),
    )
    car_start = ctx.start_of_car(0) if ctx.n_cars > 0 else 0
    best_hub = min(hub_candidates[:5], key=lambda h: table.cost(car_start, spot_cells[h]))

    meets = []
    if ctx.n_cars > 0:
        for slot in range(ctx.n_patrols):
            if best_hub in tours[slot]:
                idx = tours[slot].index(best_hub)
                meets.append(Meet(patrol=slot, stop_idx=idx, car=0))
            elif len(tours[slot]) >= 4:
                idx = len(tours[slot]) // 2
                meets.append(Meet(patrol=slot, stop_idx=idx, car=0))

    sol = trim_excess_stock(ctx, DaySolution(tours=tours, meets=meets, waits={}))
    return greedy_fill(ctx, sol)

print("=== 10-DAY SIMULATION WITH OUTLIER SPUR PING-PONG ===")
total_servings = 0
total_daily = 0
for day in range(10):
    ctx = make_context(cfg, day, agents, {}, [s.stocks for s in cfg.spots], set())
    sol = build_test_solution(ctx)
    dec = decode(ctx, sol)
    r = dec.result
    total_servings += r.servings
    total_daily += len(r.brands)
    print(f"Day {day:02d} (steps={cfg.day_steps[day]:3d}): brands={len(r.brands):2d}/{len(cfg.brands)}, servings={r.servings:2d}, ok={r.ok}, end_pos={r.end_pos}")
    agents = [AgentState(roles[i], r.end_pos[i], r.end_fuel[i]) for i in range(cfg.n_agents)]

print("=" * 60)
print(f"TOTAL (Constructive 0-ALNS): L=20 | D={total_daily} | Q={total_servings}")
