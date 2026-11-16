import sys; sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig, AgentState
from hexudon import rules
from hexudon.planner_contest.engine import make_context
from hexudon.planner.decoder import decode, DaySolution
from hexudon.planner.solution import Meet, Weights
import hexudon.planner.alns as alns_mod

with open('replays/contest-087d9b9a-t78-20261112-165702.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('path') == '/api/game/config':
            cfg = MatchConfig.from_json(obj['response'])
            break

agents = [AgentState(rules.PATROL, cfg.agent_starts[i], cfg.fuel_limits) for i in range(cfg.n_agents)]
agents[1] = AgentState(rules.REFUEL, cfg.agent_starts[1], cfg.fuel_limits)
ctx = make_context(cfg, 0, agents, {}, [s.stocks for s in cfg.spots], set())

with open('C:/Users/LAPTOP/.gemini/antigravity/brain/6cbcc38d-2ad3-4140-b3c5-3d9adeac5e23/.user_uploaded/media_1794716321428_8a2b454c.json') as f:
    replay = json.load(f)

d0 = replay['days'][0]
t77 = next(t for t in d0['teams'] if str(t['team_id']) == '77')
t77_agent_spots = [[], [], [], []]
for fr in t77['frames']:
    colls = fr.get('collected', [])
    for aid, a in enumerate(fr['agents']):
        if a['cell'] in colls and (not t77_agent_spots[aid] or t77_agent_spots[aid][-1] != a['cell']):
            t77_agent_spots[aid].append(a['cell'])

spot_map = {s.pos: i for i, s in enumerate(cfg.spots)}
tour0 = [spot_map[c] for c in t77_agent_spots[0] if c in spot_map]
tour1 = [spot_map[c] for c in t77_agent_spots[2] if c in spot_map]
tour2 = [spot_map[c] for c in t77_agent_spots[3] if c in spot_map]

sol_t77 = DaySolution(
    tours=[tour0, tour1, tour2],
    meets=[
        Meet(patrol=2, stop_idx=6, car=0),
        Meet(patrol=1, stop_idx=15, car=0),
        Meet(patrol=0, stop_idx=17, car=0),
    ],
    waits={}
)

dec0 = decode(ctx, sol_t77)
print("Initial sol_t77: score=", dec0.score, "servings=", dec0.result.servings)

def on_step(info):
    if info.get('accepted'):
        print("Iter %d: op=%s+%s, score=%.1f, best=%.1f" % (info['it'], info['destroy'], info['repair'], info['score'], info['best_score']))

res = alns_mod.improve(ctx, sol_t77, max_iterations=50, on_progress=on_step)
print("Final: score=", res.decoded.score, "servings=", res.decoded.result.servings)
