import sys
sys.path.insert(0, '.')
import json

with open("C:/Users/LAPTOP/.gemini/antigravity/brain/6cbcc38d-2ad3-4140-b3c5-3d9adeac5e23/.user_uploaded/media_1794716321428_8a2b454c.json") as f:
    replay = json.load(f)

d0 = replay["days"][0]
t77 = next(t for t in d0["teams"] if str(t["team_id"]) == "77")
frames = t77["frames"]

print("=== TEAM 77 DAY 0 ROUTE ANALYSIS ===")
# Track position sequences
routes = [[], [], [], []]
fuel_seq = [[], [], [], []]
collections = [[], [], [], []]

for step_idx, fr in enumerate(frames):
    step = fr["step"]
    collected = fr.get("collected", [])
    for aid, a in enumerate(fr["agents"]):
        cell = a["cell"]
        fuel = a["fuel"]
        if not routes[aid] or routes[aid][-1] != cell:
            routes[aid].append(cell)
        fuel_seq[aid].append(fuel)
        if cell in collected:
            collections[aid].append((step, cell))

for aid in range(4):
    print(f"\nAgent {aid} (kind={t77['kinds'][aid]}):")
    print(f"  Visited {len(routes[aid])} distinct cell transitions: {routes[aid]}")
    print(f"  Collected {len(collections[aid])} spots: {collections[aid]}")
    # Find refuel moments: when fuel increases from step t-1 to t
    refuels = []
    for t in range(1, len(fuel_seq[aid])):
        if fuel_seq[aid][t] > fuel_seq[aid][t-1]:
            refuels.append((t, fuel_seq[aid][t-1], fuel_seq[aid][t], frames[t]["agents"][aid]["cell"]))
    print(f"  Refuel events: {refuels}")
