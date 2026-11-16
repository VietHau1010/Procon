import json

with open("replays/contest-087d9b9a-t78-20261112-165702.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('path') == '/api/game/config':
            cfg = entry['response']
            break

map_info = cfg['map']
print("Map keys:", map_info.keys() if isinstance(map_info, dict) else type(map_info))
if isinstance(map_info, dict):
    width = map_info.get('width')
    height = map_info.get('height')
    print(f"Map width: {width}, height: {height}")
    # Cell 808 coordinate:
    if width:
        print(f"Cell 808 row={808 // width}, col={808 % width}")

# Let's inspect agents config
print("Agents config:", cfg.get('agents'))
print("Fuel limits:", cfg.get('fuelLimits'))
print("Day steps:", cfg.get('daySteps'))
