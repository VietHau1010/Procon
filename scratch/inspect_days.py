import json

f77_path = "C:/Users/LAPTOP/.gemini/antigravity/brain/6cbcc38d-2ad3-4140-b3c5-3d9adeac5e23/.user_uploaded/media_1794716321428_8a2b454c.json"
r77 = json.load(open(f77_path))

for day_idx in range(len(r77['days'])):
    d = r77['days'][day_idx]
    teams = {str(t['team_id']): t for t in d['teams']}
    print(f"Day {day_idx} team ids sample: {list(teams.keys())[:5]}")
    t77 = teams.get('77') or teams.get(77)
    t78 = teams.get('78') or teams.get(78)
    print(f"Day {day_idx} found 77? {t77 is not None}, 78? {t78 is not None}")
    if t77 and t78:
        print(f"  T77: kinds={t77.get('kinds')}, servings={t77.get('servings')}, types={t77.get('types')}")
        print(f"  T78: kinds={t78.get('kinds')}, servings={t78.get('servings')}, types={t78.get('types')}")
