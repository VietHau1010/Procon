import json

with open("replays/contest-087d9b9a-t78-20261112-165702.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('path') == '/api/game/config':
            cfg = entry['response']
            break

spots = cfg['spots']
pos_to_brand = {s['pos']: s['brand'] for s in spots}
brand_to_positions = {}
for s in spots:
    brand_to_positions.setdefault(s['brand'], []).append(s['pos'])

f77_path = "C:/Users/LAPTOP/.gemini/antigravity/brain/6cbcc38d-2ad3-4140-b3c5-3d9adeac5e23/.user_uploaded/media_1794716321428_8a2b454c.json"
f78_path = "replays/contest-087d9b9a-t78-20261112-165702-replay.json"

r77 = json.load(open(f77_path))
r78 = json.load(open(f78_path))

print("="*60)
print("REAL SPOT & BRAND COMPARISON (T77 vs T78)")
print("="*60)

all_diffs = []
for day_idx in range(10):
    d77 = r77['days'][day_idx]
    d78 = r78['days'][day_idx]
    
    t77 = next(t for t in d77['teams'] if str(t['team_id']) == '77')
    t78 = next(t for t in d78['teams'] if str(t['team_id']) == '78')
    
    # Spots collected
    spots_77 = set()
    for fr in t77['frames']:
        for sp in fr.get('collected', []):
            spots_77.add(sp)
            
    spots_78 = set()
    for fr in t78['frames']:
        for sp in fr.get('collected', []):
            spots_78.add(sp)
            
    brands_77 = {pos_to_brand[p] for p in spots_77 if p in pos_to_brand}
    brands_78 = {pos_to_brand[p] for p in spots_78 if p in pos_to_brand}
    
    missing_77 = set(range(20)) - brands_77
    missing_78 = set(range(20)) - brands_78
    
    print(f"\n--- Day {day_idx} ---")
    print(f"T77: {len(spots_77)} spots, {len(brands_77)} brands, Servings={t77.get('servings')}. Missing: {missing_77}")
    print(f"T78: {len(spots_78)} spots, {len(brands_78)} brands, Servings={t78.get('servings')}. Missing: {missing_78}")
    
    diff_brands = brands_77 - brands_78
    print(f"Brands collected by T77 but NOT T78: {diff_brands}")
    for b in diff_brands:
        print(f"  Brand {b} spots in map: {brand_to_positions[b]}")
        # check which spot T77 visited
        visited_by_77 = set(brand_to_positions[b]).intersection(spots_77)
        print(f"  T77 visited spot(s): {visited_by_77}")
    
    all_diffs.append((day_idx, missing_78))
