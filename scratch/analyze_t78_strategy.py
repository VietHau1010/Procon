import sys
sys.path.insert(0, '.')
import json

with open("C:/Users/LAPTOP/.gemini/antigravity/brain/6cbcc38d-2ad3-4140-b3c5-3d9adeac5e23/.user_uploaded/media_1794716321428_8a2b454c.json") as f:
    replay = json.load(f)

for day_idx in range(3):
    d = replay["days"][day_idx]
    t78 = next(t for t in d["teams"] if str(t["team_id"]) == "78")
    print(f"=== T78 DAY {day_idx} (steps={d['steps']}) ===")
    print(f"Kinds: {t78['kinds']}, Servings: {t78['servings']}, Types: {t78['types']}")
    frames = t78.get("frames", [])
    if frames:
        start_frame = frames[0]
        end_frame = frames[-1]
        print("Start positions:", [a["cell"] for a in start_frame["agents"]])
        print("Start fuels:    ", [a["fuel"] for a in start_frame["agents"]])
        print("End positions:  ", [a["cell"] for a in end_frame["agents"]])
        print("End fuels:      ", [a["fuel"] for a in end_frame["agents"]])
        print(f"End servings: {t78['servings']}, types: {t78['types']}")
