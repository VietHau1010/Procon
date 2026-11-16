import sys
sys.path.insert(0, '.')
import json

with open("C:/Users/LAPTOP/.gemini/antigravity/brain/6cbcc38d-2ad3-4140-b3c5-3d9adeac5e23/.user_uploaded/media_1794716321428_8a2b454c.json") as f:
    replay = json.load(f)

for day_idx in range(3):
    d = replay["days"][day_idx]
    t77 = next(t for t in d["teams"] if str(t["team_id"]) == "77")
    print(f"=== DAY {day_idx} (steps={d['steps']}) ===")
    print(f"Kinds: {t77['kinds']}, Servings: {t77['servings']}, Types: {t77['types']}")
    # print actions or frames
    frames = t77.get("frames", [])
    if frames:
        # let's see start and end pos, fuel
        start_frame = frames[0]
        end_frame = frames[-1]
        print("Start positions:", [a["cell"] for a in start_frame["agents"]])
        print("Start fuels:    ", [a["fuel"] for a in start_frame["agents"]])
        print("End positions:  ", [a["cell"] for a in end_frame["agents"]])
        print("End fuels:      ", [a["fuel"] for a in end_frame["agents"]])
        # Count collections per agent
        colls = [0] * len(t77["kinds"])
        for fr in frames:
            for cell in fr.get("collected", []):
                pass
        print(f"End servings: {t77['servings']}, types: {t77['types']}")
