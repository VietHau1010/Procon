import sys; sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig, AgentState
from hexudon import rules
from hexudon.planner_contest.engine import make_context
from hexudon.planner_contest.construct import construct

with open('replays/contest-087d9b9a-t78-20261112-165702.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('path') == '/api/game/config':
            cfg = MatchConfig.from_json(obj['response'])
            break

with open('C:/Users/LAPTOP/.gemini/antigravity/brain/6cbcc38d-2ad3-4140-b3c5-3d9adeac5e23/.user_uploaded/media_1794716321428_8a2b454c.json') as f:
    replay = json.load(f)

d0 = replay['days'][0]
t77 = next(t for t in d0['teams'] if str(t['team_id']) == '77')

agents = [AgentState(rules.PATROL, cfg.agent_starts[i], cfg.fuel_limits) for i in range(cfg.n_agents)]
agents[1] = AgentState(rules.REFUEL, cfg.agent_starts[1], cfg.fuel_limits)
ctx = make_context(cfg, 0, agents, {}, [s.stocks for s in cfg.spots], set())
sol = construct(ctx)

print("T77 Day 0 collections:")
all_t77_colls = []
for fr in t77['frames']:
    all_t77_colls.extend(fr.get('collected', []))
print(f"Total T77 collected: {len(all_t77_colls)} cells, unique: {len(set(all_t77_colls))}")
from collections import Counter
c77 = Counter(all_t77_colls)
print("Most visited spots by T77:", c77.most_common(10))

all_our_colls = []
for slot in range(ctx.n_patrols):
    all_our_colls.extend([ctx.spot_cells[s] for s in sol.tours[slot]])
print(f"\nOur Day 0 collected: {len(all_our_colls)} cells, unique: {len(set(all_our_colls))}")
c_our = Counter(all_our_colls)
print("Most visited spots by Us:", c_our.most_common(10))

# Compare spot locations
print("\nSpots visited by T77 but NOT by us:")
print(set(all_t77_colls) - set(all_our_colls))
print("Spots visited by us but NOT by T77:")
print(set(all_our_colls) - set(all_t77_colls))
