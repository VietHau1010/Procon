import sys
sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig
from hexudon.simulator import run_day

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
t78 = next(t for t in d0['teams'] if str(t['team_id']) == '78')

print("T77 Day 0 submitted length:", [len(p) for p in t77['submitted']])
print("T78 Day 0 submitted length:", [len(p) for p in t78['submitted']])
