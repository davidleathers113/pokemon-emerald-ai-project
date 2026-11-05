"""Rotating tile puzzles (MOSSDEEP CITY GYM, the Trick House's seventh puzzle), modelled
from pret pokeemerald src/rotating_tile_puzzle.c so walks can plan through them.

The colored arrow tiles never move the player: their metatiles are MB_NORMAL. A floor
switch (a coord event whose script runs `moverotatingtileobjects N`) moves every object
whose template stands on an arrow tile of color N one tile the way the arrow points
(MoveRotatingTileObjects). Statues and trainers alike: the arrows are laid in loops, so
a statue that blocks a way can be turned out of it. Nothing blocks such a move: the
template's x/y are changed and a scripted WALK_NORMAL plays (no collision check).
The moved coordinates stay in SaveBlock1's objectEventTemplates, also across the gym's
own pads (pret field_screen_effect.c DoMossdeepGymWarp: SKIP_OBJECT_EVENT_LOAD).

So which tiles join which depends on where the objects stand, and each switch changes
it: plan() searches over (player tile, object places), with the map's warps that lead
to another spot of the same map as edges.

Seen live (2026-11-05): in MOSSDEEP CITY GYM at (3,20) a statue stood at (4,20), where
the map file has none; A there found "no one", and the way on crossed the yellow switch
(2,21), which moved it.
"""
from collections import deque
from collections.abc import Callable, Iterable

# pret include/constants/metatile_labels.h: the first puzzle metatile of each puzzle; the
# puzzle tiles are rows of METATILE_ROW_WIDTH (8), one row per color (5 colors), the
# first four of a row the arrows Right, Down, Left, Up.
MOSSDEEP_GYM_START = 0x250  # METATILE_MossdeepGym_YellowArrow_Right
TRICK_HOUSE_START = 0x298  # METATILE_TrickHousePuzzle_Arrow_YellowOnWhite_Right
METATILE_ROW_WIDTH = 8
COLORS = ("yellow", "blue", "green", "purple", "red")
ARROW_STEPS = ((1, 0), (0, 1), (-1, 0), (0, -1))  # Right, Down, Left, Up

# Bound on the search: (player area, object places) nodes looked at before giving up.
# MOSSDEEP CITY GYM's way from (3,20) to TATE AND LIZA (18 presses, 5 pads) is found
# after about 20,000 (~2 s); its five colors' loops give at most 4^5 object places.
MAX_NODES = 30000

Tile = tuple[int, int]


def arrow_at(metatile: int, trick_house: bool) -> tuple[int, Tile] | None:
    """(color, step) of the arrow a puzzle metatile shows, or None (MoveRotatingTileObjects:
    below METATILE_MossdeepGym_YellowArrow_Right, past the fifth row, or one of a row's
    last four tiles is no arrow)."""
    start = TRICK_HOUSE_START if trick_house else MOSSDEEP_GYM_START
    if metatile < MOSSDEEP_GYM_START or metatile < start:
        return None
    color, number = divmod(metatile - start, METATILE_ROW_WIDTH)
    if color >= len(COLORS) or number >= 4:
        return None
    return color, ARROW_STEPS[number]


def arrows_of(tiles, trick_house: bool) -> dict[Tile, tuple[int, Tile]]:
    """{tile: (color, step)} for every arrow tile of a layout ((metatile id, ...) by row)."""
    out = {}
    for y, row in enumerate(tiles or ()):
        for x, tile in enumerate(row):
            arrow = arrow_at(int(tile[0]), trick_house) if tile else None
            if arrow is not None:
                out[(x, y)] = arrow
    return out


def move_objects(places: tuple[Tile, ...], arrows: dict[Tile, tuple[int, Tile]], color: int) -> tuple[Tile, ...]:
    """Where the objects stand after switch `color`: each one on an arrow of that color
    one tile along it, the rest where they were (MoveRotatingTileObjects reads every
    template's tile once, so an object moved onto another arrow of the color moves no
    further)."""
    out = []
    for place in places:
        arrow = arrows.get(place)
        if arrow is not None and arrow[0] == color:
            out.append((place[0] + arrow[1][0], place[1] + arrow[1][1]))
        else:
            out.append(place)
    return tuple(out)


def plan(
    start: Tile,
    places: tuple[Tile, ...],
    goal: Iterable[Tile],
    steps: Callable[[Tile, frozenset], Iterable[Tile]],
    switches: dict[Tile, int],
    pads: dict[Tile, Tile],
    arrows: dict[Tile, tuple[int, Tile]],
    max_nodes: int = MAX_NODES,
) -> list[tuple[str, Tile]] | None:
    """The fewest switch presses and pad rides from start to a goal tile.

    steps(tile, blocked) gives the tiles a walker on `tile` can step onto, never one in
    blocked (where objects stand). Switch tiles and pads are never walked over: stepping
    on one is an action of its own. Returns [("press", switch tile) | ("warp", pad tile),
    ..., ("walk", goal tile)], [("walk", goal tile)] when a goal is reachable already,
    or None when no way is found within max_nodes.
    """
    goal = frozenset(goal)
    special = frozenset(switches) | frozenset(pads)
    seen: set = set()
    queue: deque = deque([(start, places, [])])
    nodes = 0
    while queue and nodes < max_nodes:
        here, now, actions = queue.popleft()
        nodes += 1
        blocked = frozenset(now)
        area, edges = _flood(here, blocked, steps, special)
        # The whole area, not one tile of it: a switch the player stands on joins the
        # floors on both sides of it, so two areas can share their first tile.
        key = (frozenset(area), now)
        if key in seen:
            continue
        seen.add(key)
        reached = sorted(area & goal, key=lambda t: (abs(t[0] - here[0]) + abs(t[1] - here[1]), t))
        if reached:
            return actions + [("walk", reached[0])]
        for tile in sorted(edges):
            if tile in blocked:
                continue
            if tile in switches:
                moved = move_objects(now, arrows, switches[tile])
                queue.append((tile, moved, actions + [("press", tile)]))
            elif tile in pads:
                queue.append((pads[tile], now, actions + [("warp", tile)]))
    return None


def _flood(start: Tile, blocked: frozenset, steps, special: frozenset) -> tuple[set, set]:
    """(tiles walked to from start without stepping on a special tile, the special tiles
    beside them)."""
    area, edges, stack = {start}, set(), [start]
    while stack:
        tile = stack.pop()
        for nxt in steps(tile, blocked):
            if nxt in special:
                edges.add(nxt)
            elif nxt not in area:
                area.add(nxt)
                stack.append(nxt)
    return area, edges
