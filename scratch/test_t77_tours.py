import sys; sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig, AgentState
from hexudon import rules
from hexudon.planner_contest.engine import make_context
from hexudon.planner.decoder import decode, DaySolution

with open('replays/contest-087d9b9a-t78-20261112-165702.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('path') == '/api/game/config':
            cfg = MatchConfig.from_json(obj['response'])
            break

agents = [AgentState(rules.PATROL, cfg.agent_starts[i], cfg.fuel_limits) for i in range(cfg.n_agents)]
agents[1] = AgentState(rules.REFUEL, cfg.agent_starts[1], cfg.fuel_limits)
ctx = make_context(cfg, 0, agents, {}, [s.stocks for s in cfg.spots], set())

# Let's inspect the exact spots Team 77 visited per agent:
with open('C:/Users/LAPTOP/.gemini/antigravity/brain/6cbcc38d-2ad3-4140-b3c5-3d9adeac5e23/.user_uploaded/media_1794716321428_8a2b454c.json') as f:
    replay = json.load(f)

d0 = replay['days'][0]
t77 = next(t for t in d0['teams'] if str(t['team_id']) == '77')

t77_agent_spots = [[], [], [], []]
for fr in t77['frames']:
    colls = fr.get('collected', [])
    for aid, a in enumerate(fr['agents']):
        if a['cell'] in colls and a['cell'] not in t77_agent_spots[aid]:
            t77_agent_spots[aid].append(a['cell'])

print("T77 Patrol 0 spots:", len(t77_agent_spots[0]), t77_agent_spots[0])
print("T77 Patrol 2 spots:", len(t77_agent_spots[2]), t77_agent_spots[2])
print("T77 Patrol 3 spots:", len(t77_agent_spots[3]), t77_agent_spots[3])

# Convert cells to spot indices in ctx
spot_map = {s.pos: i for i, s in enumerate(cfg.spots)}
tour0 = [spot_map[c] for c in t77_agent_spots[0] if c in spot_map]
tour1 = [spot_map[c] for c in t77_agent_spots[2] if c in spot_map]
tour2 = [spot_map[c] for c in t77_agent_spots[3] if c in spot_map]

from hexudon.planner.solution import Meet
# Check meets from T77:
# Agent 3 refueled at step 33 at 549 (stop idx ~ 6)
# Agent 2 refueled at step 88 at 549 (stop idx ~ 15)
# Agent 0 refueled at step 99 at 549 (stop idx ~ 17) and step 117 at 692
sol = DaySolution(
    tours=[tour0, tour1, tour2],
    meets=[
        Meet(patrol=2, stop_idx=6, car=0),
        Meet(patrol=1, stop_idx=15, car=0),
        Meet(patrol=0, stop_idx=17, car=0),
    ],
    waits={}
)

dec = decode(ctx, sol)
print("\nT77-style DaySolution decoded:")
print(f"ok={dec.result.ok}, brands={len(dec.result.brands)}, servings={dec.result.servings}, score={dec.score}")
print(f"end_pos={dec.result.end_pos}")
print(f"end_fuel={dec.result.end_fuel}")
