import sys; sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig, AgentState
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

# Read T77 Day 0 to see how close we get
with open('C:/Users/LAPTOP/.gemini/antigravity/brain/6cbcc38d-2ad3-4140-b3c5-3d9adeac5e23/.user_uploaded/media_1794716321428_8a2b454c.json') as f:
    replay = json.load(f)

# Master cycle of all 30 spots
# [448, 446, 326, 268, 57, 294, 321, 290, 288, 286, 257, 76, 131, 250, 245, 184, 336, 363, 546, 518, 549, 692, 637, 703, 643, 522, 468, 471, 532, 808]
spot_by_cell = {s.pos: i for i, s in enumerate(cfg.spots)}
master_cycle_cells = [
    448, 446, 326, 268, 57, 294, 321, 290, 288, 286, 257, 76, 131, 250, 245, 184,
    336, 363, 546, 518, 549, 692, 637, 703, 643, 522, 468, 471, 532, 808
]
# Ensure all 30 spots are present
assert len(master_cycle_cells) == 30
master_cycle_spots = [spot_by_cell[c] for c in master_cycle_cells]

# Test on Day 0:
agents = [AgentState(rules.PATROL, cfg.agent_starts[i], cfg.fuel_limits) for i in range(cfg.n_agents)]
agents[1] = AgentState(rules.REFUEL, cfg.agent_starts[1], cfg.fuel_limits)
ctx = make_context(cfg, 0, agents, {}, [s.stocks for s in cfg.spots], set())
table = ctx.dist_step

# For each patrol: find entry in master cycle, slice K spots that fit in ctx.steps
tours = []
n = 30
for slot in range(ctx.n_patrols):
    start_c = ctx.start_of_patrol(slot)
    entry_idx = min(range(n), key=lambda i: table.cost(start_c, master_cycle_cells[i]))
    
    best_t = []
    for direction in [1, -1]:
        cur_t = []
        cur_c = start_c
        steps = 0
        idx = entry_idx
        while True:
            nxt_c = master_cycle_cells[idx]
            s = table.cost(cur_c, nxt_c)
            if steps + s > ctx.steps:
                break
            cur_t.append(master_cycle_spots[idx])
            steps += s
            cur_c = nxt_c
            idx = (idx + direction) % n
            if idx == entry_idx:
                break
        if len(cur_t) > len(best_t):
            best_t = cur_t
    tours.append(best_t)

print("Sliced tours lengths:", [len(t) for t in tours])
# Meets: central hub cell 549
hub_spot = spot_by_cell[549]
meets = []
for slot in range(ctx.n_patrols):
    if hub_spot in tours[slot]:
        meets.append(Meet(patrol=slot, stop_idx=tours[slot].index(hub_spot), car=0))

sol = DaySolution(tours=tours, meets=meets, waits={})
dec = decode(ctx, sol)
print(f"Master Cycle Sliced Day 0: ok={dec.result.ok}, brands={len(dec.result.brands)}, servings={dec.result.servings}")
