import sys
sys.path.insert(0, '.')
import json
from hexudon.model import MatchConfig
from hexudon.pathing import DayGraph, DistTable

with open("replays/contest-087d9b9a-t78-20261112-165702.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('path') == '/api/game/config':
            cfg_dict = entry['response']
        if entry.get('path') == '/api/game/state' and entry.get('response', {}).get('day') == 0:
            st0_dict = entry['response']
            break

cfg = MatchConfig.from_json(cfg_dict)
road_status = {int(k): v for k, v in st0_dict.get('road_condition', {}).items()}
graph = DayGraph(cfg, road_status)

spots = cfg.spots
sources = list(cfg.agent_starts) + [s.pos for s in spots]
dt = DistTable(graph, sources, metric="step")

pos_to_idx = {s.pos: i for i, s in enumerate(spots)}
print(f"Index of spot 808: {pos_to_idx.get(808)}")

for a_idx, a_cell in enumerate(cfg.agent_starts):
    scored = []
    for si, s in enumerate(spots):
        d = dt.cost(a_cell, s.pos)
        scored.append((d, si, s.pos, s.brand))
    scored.sort()
    print(f"\nNearest spots to start cell {a_cell} (Agent {a_idx}):")
    for rank, (d, si, spos, sbrand) in enumerate(scored[:15]):
        is_808 = (spos == 808)
        print(f"  Rank {rank+1:2d}: spot_idx={si:2d}, cell={spos:3d}, brand={sbrand:2d}, steps={d} {'<-- 808!' if is_808 else ''}")
    for rank, (d, si, spos, sbrand) in enumerate(scored):
        if spos == 808:
            print(f"  -> Spot 808 is ranked {rank+1} of 30 spots with steps={d}")

# Also check nearest spots to all spots that are close to the south
print("\nWhich spots have 808 in their top 10 nearest?")
for si, s in enumerate(spots):
    scored = []
    for other_si, other_s in enumerate(spots):
        if other_si == si:
            continue
        d = dt.cost(s.pos, other_s.pos)
        scored.append((d, other_si, other_s.pos))
    scored.sort()
    # Is 808 in top 10?
    top10_cells = [x[2] for x in scored[:10]]
    if 808 in top10_cells:
        rank_808 = top10_cells.index(808) + 1
        print(f"Spot {s.pos} (brand {s.brand}) has 808 in top 10! (Rank {rank_808}, dist={scored[rank_808-1][0]})")
