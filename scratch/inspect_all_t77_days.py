import sys; sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig, AgentState, DayState
from hexudon import rules
from hexudon.planner_contest.engine import make_context
from hexudon.planner.decoder import decode, DaySolution
from hexudon.planner.solution import Meet
from hexudon.planner.alns import improve

with open('replays/contest-087d9b9a-t78-20261112-165702.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('path') == '/api/game/config':
            cfg = MatchConfig.from_json(obj['response'])
            break

with open('C:/Users/LAPTOP/.gemini/antigravity/brain/6cbcc38d-2ad3-4140-b3c5-3d9adeac5e23/.user_uploaded/media_1794716321428_8a2b454c.json') as f:
    replay = json.load(f)

# Let's extract Team 77's tours for all 10 days to see how their pattern works across all days!
for day_idx in range(10):
    d = replay['days'][day_idx]
    t77 = next(t for t in d['teams'] if str(t['team_id']) == '77')
    agent_spots = [[], [], [], []]
    for fr in t77['frames']:
        colls = fr.get('collected', [])
        for aid, a in enumerate(fr['agents']):
            if a['cell'] in colls and (not agent_spots[aid] or agent_spots[aid][-1] != a['cell']):
                agent_spots[aid].append(a['cell'])
    print(f"Day {day_idx} (steps={d['steps']}): T77 serv={t77['servings']}, patrol spots: {[len(agent_spots[i]) for i in [0, 2, 3]]}")
