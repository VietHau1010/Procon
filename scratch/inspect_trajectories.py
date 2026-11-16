import json

f77_path = "C:/Users/LAPTOP/.gemini/antigravity/brain/6cbcc38d-2ad3-4140-b3c5-3d9adeac5e23/.user_uploaded/media_1794716321428_8a2b454c.json"
f78_path = "replays/contest-087d9b9a-t78-20261112-165702-replay.json"

r77 = json.load(open(f77_path))
r78 = json.load(open(f78_path))

# Check Day 0 for T77
d0_77 = r77['days'][0]
t77 = next(t for t in d0_77['teams'] if str(t['team_id']) == '77')

print("=== DAY 0 - T77 VISITING CELL 808 ===")
for fr_idx, fr in enumerate(t77['frames']):
    step = fr['step']
    collected = fr.get('collected', [])
    if 808 in collected:
        print(f"Step {step}: Cell 808 COLLECTED! Frame info: {fr}")
    for a_idx, a in enumerate(fr['agents']):
        if a['cell'] == 808:
            print(f"Step {step}: Agent {a_idx} at 808! Fuel: {a['fuel']}")

# Check refueling events for T77 across Day 0
print("\n=== DAY 0 - T77 REFUELING EVENTS ===")
for fr_idx in range(1, len(t77['frames'])):
    prev_fr = t77['frames'][fr_idx - 1]
    curr_fr = t77['frames'][fr_idx]
    
    # Check if any agent's fuel increased
    for a_idx in range(len(curr_fr['agents'])):
        prev_fuel = prev_fr['agents'][a_idx]['fuel']
        curr_fuel = curr_fr['agents'][a_idx]['fuel']
        if curr_fuel > prev_fuel:
            print(f"Step {curr_fr['step']}: Agent {a_idx} refueled from {prev_fuel} to {curr_fuel}! Pos: {curr_fr['agents'][a_idx]['cell']}, Tanker pos: {curr_fr['agents'][1]['cell']}")

# Now check T78 Day 0: where did T78 agents go?
print("\n=== DAY 0 - T78 AGENTS TRAJECTORY & CLOSEST TO 808 ===")
d0_78 = r78['days'][0]
t78 = next(t for t in d0_78['teams'] if str(t['team_id']) == '78')

min_dist_to_808 = {0: 999, 1: 999, 2: 999, 3: 999}
closest_cell = {0: None, 1: None, 2: None, 3: None}

def dist(c1, c2, w=30):
    r1, cl1 = c1 // w, c1 % w
    r2, cl2 = c2 // w, c2 % w
    return abs(r1 - r2) + abs(cl1 - cl2) # Manhattan approx or hex dist

for fr in t78['frames']:
    for a_idx, a in enumerate(fr['agents']):
        d = dist(a['cell'], 808)
        if d < min_dist_to_808[a_idx]:
            min_dist_to_808[a_idx] = d
            closest_cell[a_idx] = (a['cell'], fr['step'], a['fuel'])

for a_idx in range(4):
    print(f"T78 Agent {a_idx} closest to 808: dist={min_dist_to_808[a_idx]}, details={closest_cell[a_idx]}")

# Check refueling events for T78 across Day 0
print("\n=== DAY 0 - T78 REFUELING EVENTS ===")
for fr_idx in range(1, len(t78['frames'])):
    prev_fr = t78['frames'][fr_idx - 1]
    curr_fr = t78['frames'][fr_idx]
    for a_idx in range(len(curr_fr['agents'])):
        prev_fuel = prev_fr['agents'][a_idx]['fuel']
        curr_fuel = curr_fr['agents'][a_idx]['fuel']
        if curr_fuel > prev_fuel:
            print(f"Step {curr_fr['step']}: Agent {a_idx} refueled from {prev_fuel} to {curr_fuel}! Pos: {curr_fr['agents'][a_idx]['cell']}, Tanker pos: {curr_fr['agents'][1]['cell']}")
