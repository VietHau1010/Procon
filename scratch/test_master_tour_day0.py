import sys; sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig, AgentState
from hexudon import rules
from hexudon.planner_contest.engine import make_context
from hexudon.planner.decoder import decode, DaySolution
from hexudon.planner.solution import Meet

with open('replays/contest-087d9b9a-t78-20261112-165702.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('path') == '/api/game/config':
            cfg = MatchConfig.from_json(obj['response'])
            break

agents = [AgentState(rules.PATROL, cfg.agent_starts[i], cfg.fuel_limits) for i in range(cfg.n_agents)]
agents[1] = AgentState(rules.REFUEL, cfg.agent_starts[1], cfg.fuel_limits)
ctx = make_context(cfg, 0, agents, {}, [s.stocks for s in cfg.spots], set())
table = ctx.dist_step

# 1. Build Master Tour on all 30 spots using Nearest Neighbor + 2-Opt
all_spots = list(range(len(cfg.spots)))
# Start NN from spot 0
current = 0
unvisited = set(all_spots) - {current}
master_tour = [current]
while unvisited:
    cur_cell = ctx.spot_cells[current]
    next_spot = min(unvisited, key=lambda s: table.cost(cur_cell, ctx.spot_cells[s]))
    master_tour.append(next_spot)
    unvisited.remove(next_spot)
    current = next_spot

# 2-Opt on master_tour
improved = True
n = len(master_tour)
def cycle_cost(t):
    return sum(table.cost(ctx.spot_cells[t[i]], ctx.spot_cells[t[(i+1)%n]]) for i in range(n))

cur_c = cycle_cost(master_tour)
while improved:
    improved = False
    for i in range(n - 1):
        for j in range(i + 1, n):
            cand = master_tour[:i] + master_tour[i:j+1][::-1] + master_tour[j+1:]
            c = cycle_cost(cand)
            if c < cur_c:
                cur_c = c
                master_tour = cand
                improved = True
                break
        if improved:
            break

print("Master tour length:", n, "Cycle cost:", cur_c)

# 2. For each patrol, enter the master tour at nearest spot, and follow it
tours = []
meets = []
# Find a good central hub for refueling (e.g. spot whose cell is 549 or has lowest sum dist)
spot_cells = ctx.spot_cells
hub_spot = next((i for i, c in enumerate(spot_cells) if c == 549), None)
if hub_spot is None:
    hub_spot = min(all_spots, key=lambda s: sum(table.cost(spot_cells[s], spot_cells[o]) for o in all_spots))

print(f"Hub spot: {hub_spot} (cell {spot_cells[hub_spot]})")

for slot in range(ctx.n_patrols):
    start_c = ctx.start_of_patrol(slot)
    # Find closest entry spot in master tour
    entry_idx = min(range(n), key=lambda i: table.cost(start_c, spot_cells[master_tour[i]]))
    
    # Try forward and backward traversal from entry_idx
    best_t = []
    best_servings = 0
    for direction in [1, -1]:
        cur_t = []
        cur_cell = start_c
        steps = 0
        fuel = ctx.fuel_of_patrol(slot)
        
        idx = entry_idx
        while True:
            sp = master_tour[idx]
            nxt_c = spot_cells[sp]
            s, f = table.steps_fuel(cur_cell, nxt_c)
            if steps + s > ctx.steps:
                break
            cur_t.append(sp)
            steps += s
            cur_cell = nxt_c
            idx = (idx + direction) % n
            if idx == entry_idx:
                break
        if len(cur_t) > len(best_t):
            best_t = cur_t
    tours.append(best_t)

# Add refuel meets: if hub_spot is in patrol's tour, meet at hub_spot
for slot in range(ctx.n_patrols):
    if hub_spot in tours[slot]:
        stop_idx = tours[slot].index(hub_spot)
        meets.append(Meet(patrol=slot, stop_idx=stop_idx, car=0))

sol = DaySolution(tours=tours, meets=meets, waits={})
dec = decode(ctx, sol)
print(f"Master Tour DaySolution: ok={dec.result.ok}, brands={len(dec.result.brands)}, servings={dec.result.servings}, score={dec.score}")
for slot in range(ctx.n_patrols):
    print(f"Patrol {slot}: len={len(tours[slot])}, cells={[spot_cells[s] for s in tours[slot]]}")
