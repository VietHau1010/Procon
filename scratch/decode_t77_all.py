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

with open('C:/Users/LAPTOP/.gemini/antigravity/brain/6cbcc38d-2ad3-4140-b3c5-3d9adeac5e23/.user_uploaded/media_1794716321428_8a2b454c.json') as f:
    replay_t77 = json.load(f)

spot_map = {s.pos: i for i, s in enumerate(cfg.spots)}

total_servings = 0
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
    
    # Check refuels from frames:
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
    ctx = make_context(cfg, day, agents, {}, [s.stocks for s in cfg.spots], set())
    sol = DaySolution(tours=patrol_spots, meets=meets, waits={})
    dec = decode(ctx, sol)
    total_servings += dec.result.servings
    t77_s = t77['servings']
    print("Day %02d: serv=%d/%d, ok=%s, brands=%d" % (day, dec.result.servings, t77_s, dec.result.ok, len(dec.result.brands)))

print("Total Decoded Servings: %d / 737" % total_servings)
