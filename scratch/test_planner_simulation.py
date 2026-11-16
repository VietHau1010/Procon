import sys
sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig, DayState
from hexudon.simulator import run_day
from hexudon import rules

with open("replays/contest-087d9b9a-t78-20261112-165702.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('path') == '/api/game/config':
            cfg_dict = entry['response']
        if entry.get('path') == '/api/game/state' and entry.get('response', {}).get('day') == 0:
            st0_dict = entry['response']
            break

cfg = MatchConfig.from_json(cfg_dict)
print(f"Loaded config: agents={cfg.agent_starts}, fuel={cfg.fuel_limits}, spots={len(cfg.spots)}")
print(f"Target spots positions: {sorted([s.pos for s in cfg.spots])}")
print(f"Day 0 steps: {cfg.day_steps[0]}")
