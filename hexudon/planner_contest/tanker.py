"""Bounded appointment-order alternatives for one tanker.

The original scheduler is always the fallback. Each alternative is rebuilt
with the patrol walks and validated by the whole-team simulator in decoder.
"""

from ..planner.decoder import route_cars


MAX_BEAM_MEETINGS = 8
BEAM_WIDTH = 16
CACHE_SIZE = 128


def _normalise_windows(ctx, ordered, waits):
    events = []
    for start, meet, (cell, earliest, latest) in ordered:
        if meet.stop_idx < 0:
            # No refuelling happens at step 0. At the starting-cell meeting,
            # hold the tanker through at least reflection step 1; arriving at
            # the patrol's departure step is allowed (reflection precedes action).
            earliest = 1
            latest = min(ctx.steps, max(1, waits.get((meet.patrol, -1), 1)))
        events.append((earliest, meet, (cell, earliest, latest)))
    return events


def _beam_indices(ctx, start, events, width=BEAM_WIDTH):
    """Keep earliest-time labels for (visited appointments, last appointment).

    Fixed windows only: waiting requested by the decoder is recalculated in
    the next synchronization round. Rank by appointments served, then finish
    time; this is not the final udon objective, which is checked by decoder.
    """
    count = len(events)
    if count > MAX_BEAM_MEETINGS:
        return tuple(sorted(range(count), key=lambda i: (events[i][2][2], events[i][0], i)))
    # (visited mask, current cell, current time, appointment order)
    frontier = [(0, start, 0, ())]
    best = frontier[0]
    for _ in range(count):
        labels = {}
        for mask, cell, now, order in frontier:
            for index, (_, _, (target, earliest, latest)) in enumerate(events):
                if mask & (1 << index):
                    continue
                travel = 0 if cell == target else ctx.dist_step.cost(cell, target)
                arrival = max(now + travel, earliest)
                if arrival > min(latest, ctx.steps):
                    continue
                nxt = (mask | (1 << index), target, arrival, order + (index,))
                key = (nxt[0], index)
                old = labels.get(key)
                if old is None or (nxt[2], nxt[3]) < (old[2], old[3]):
                    labels[key] = nxt
        if not labels:
            break
        frontier = sorted(labels.values(), key=lambda s: (s[2], s[3]))[:width]
        candidate = frontier[0]
        if (-len(candidate[3]), candidate[2], candidate[3]) < (-len(best[3]), best[2], best[3]):
            best = candidate
    # The existing scheduler still gets every appointment exactly once; it
    # can request a bounded patrol wait for an unserved fixed-window event.
    rest = sorted((i for i in range(count) if not best[0] & (1 << i)),
                  key=lambda i: (events[i][2][2], events[i][0], i))
    return best[3] + tuple(rest)


def ordered_meetings(ctx, car_slot, ordered, waits, mode):
    events = _normalise_windows(ctx, ordered, waits)
    if mode == "deadline":
        return sorted(events, key=lambda row: (row[2][2], row[0], row[1].patrol, row[1].stop_idx))
    if mode != "beam":
        raise ValueError("Unknown tanker ordering policy")
    start = ctx.start_of_car(car_slot)
    key = (start, ctx.steps, tuple((m.patrol, m.stop_idx, *win) for _, m, win in events))
    cache = getattr(ctx, "_tanker_order_cache", None)
    if cache is None:
        cache = {}
        ctx._tanker_order_cache = cache
    order = cache.get(key)
    if order is None:
        order = _beam_indices(ctx, start, events)
        if len(cache) >= CACHE_SIZE:
            cache.clear()
        cache[key] = order
    return [events[i] for i in order]


def scheduler(mode):
    def schedule(ctx, sol, walks, alive, waits, allow_wait):
        return route_cars(ctx, sol, walks, alive, waits, allow_wait,
                          meeting_order=lambda c, car, events, w: ordered_meetings(c, car, events, w, mode))
    return schedule
