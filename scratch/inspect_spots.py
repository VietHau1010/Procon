import json

with open("replays/contest-087d9b9a-t78-20261112-165702.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('path') == '/api/game/config':
            cfg = entry['response']
            print("Config keys:", cfg.keys())
            spots = cfg.get('spots', [])
            print("Number of spots:", len(spots))
            print("Sample spots:", spots[:3])
            
            spot_to_brand = {}
            brand_to_spots = {}
            for s in spots:
                # s format check
                cell = s.get('cell') or s.get('id') or s.get('index')
                brand = s.get('type') or s.get('brand') or s.get('kind')
                print("Spot detail:", s)
                break
            break
