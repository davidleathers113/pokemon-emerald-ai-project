"""
Pathfinding utilities for Pokemon Emerald navigation.

Provides A* pathfinding algorithm with collision detection to enable
intelligent navigation around obstacles.
"""

import heapq
import logging
import random
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
import os

# Try to import MetatileBehavior, but don't fail if not available
try:
    from pokemon_env.enums import MetatileBehavior
except ImportError:
    MetatileBehavior = None

logger = logging.getLogger(__name__)

# Mapping from variance to the number of initial moves that must differ
VARIANCE_TO_STEPS = {
    "low": 1,
    "medium": 3,
    "high": 5,
    "extreme": 8,
}

# Pathfinding cost penalties
# Penalty for walking through tall grass. Seen live (2026-11-03): with 1.5, go_to_exit on
# Route 116 cut straight through the grass and a wild battle stopped nearly every walk,
# though a grass-free way existed. A grass step starts a battle ~11% of the time
# (Route 116: rate 20 x 16 / 2880) and a battle costs about a minute of model time, versus
# well under a second per plain step, so a grass step is worth a detour of many tiles.
# 8 detours freely but still walks into grass when it is the only way or the target.
GRASS_TILE_COST_MULTIPLIER = 8.0
# A rock or tree the party can clear (ROCK SMASH, CUT): walkable for the planner, but each
# costs a stop, a question and an animation, so open ground is preferred when there is any.
CLEARABLE_TILE_COST = 10.0
# Map elevation 15 (bridges): pret IsElevationMismatchAt never reports a mismatch on it,
# and ObjectEventUpdateElevation keeps the walker's elevation while stepping on or off
# it, so a bridge joins the floor it was entered from to the same floor beyond. Seen
# live: METEOR FALLS 1F 1R's bridge (19-21,18) between elevation-4 floor counted as a
# wall; the player could neither reach the scene at (14,18) nor walk back to the door.
BRIDGE_ELEVATION = 15
# Grid symbols whose tiles are crossed though their collision bit is set: ledges, doors,
# warps and the cycling road (_get_blocked_positions).
PASSABLE_WITH_COLLISION = frozenset({"D", "S", "&", "←", "→", "↑", "↓", "↗", "↖", "↘", "↙"})
# Grid symbols of ledges (and the slides drawn like them): jumped over, whatever the
# elevations on either side (_elevation_allows).
LEDGE_SYMBOLS = frozenset({"←", "→", "↑", "↓", "↗", "↖", "↘", "↙"})
# Behavior-name words of warp tiles (doors, stairs, ladders, arrow warps): a step onto one
# ends the walk, so no elevation rule applies to it (_elevation_allows).
WARP_BEHAVIOR_WORDS = ("LADDER", "STAIRS", "DOOR", "WARP")

# Surf. Seen live (twice): with SURF on AZUMARILL and the Balance Badge, go_to("Route 119")
# from ROUTE 118 said it "can't be reached on foot" and the AI stalled at the river at
# (17,10); by hand it was A facing the water, YES to "Would you like to SURF?", and walk.
# Water a surfer moves on: pret src/metatile_behavior.c sTileBitAttributes,
# TILE_FLAG_SURFABLE. Left out though surfable there: MB_WATERFALL (going up it needs
# WATERFALL, pret GetInteractedWaterScript), the MB_*_CURRENTs (pret field_player_avatar.c
# ForcedMovement slides the surfer along them, so a walk planned step by step loses its
# place) and MB_WATER_DOOR / MB_WATER_SOUTH_ARROW_WARP, which are exits ('D'/'S').
SURF_WATER_BEHAVIORS = frozenset({
    "POND_WATER", "INTERIOR_DEEP_WATER", "DEEP_WATER", "SOOTOPOLIS_DEEP_WATER", "OCEAN_WATER",
    "NO_SURFACING", "SEAWEED", "SEAWEED_NO_SURFACING",
})
# Water Surf can be started onto: pret MetatileBehavior_IsSurfableFishableWater (less the
# currents, as above), checked by IsPlayerFacingSurfableFishableWater.
SURF_START_BEHAVIORS = frozenset({
    "POND_WATER", "INTERIOR_DEEP_WATER", "DEEP_WATER", "SOOTOPOLIS_DEEP_WATER", "OCEAN_WATER",
})
# pret ELEVATION_DEFAULT: IsPlayerFacingSurfableFishableWater wants the player at it
# (PlayerGetElevation() == 3) and facing an elevation mismatch, and CanStopSurfing lands
# only on a tile at it. Water itself is elevation 1, so the shore is where 3 meets 1.
SHORE_ELEVATION = 3
# The elevation of water in the layouts (and so of a surfer on it).
WATER_ELEVATION = 1
# Planner cost of starting Surf: like a rock to smash, a stop, a question and an
# animation, so a way on land of about as many steps is taken instead.
SURF_START_COST = CLEARABLE_TILE_COST

# Strength boulders (OBJ_EVENT_GFX_PUSHABLE_BOULDER). Seen live (2026-11-04, MAGMA HIDEOUT
# 1F): boulders at (5,22), (7,22) and (6,23) cut (5,21) off from the exit at (10,34), and
# no walk planned a way past them. With FLAG_SYS_USE_STRENGTH set, a step into
# a boulder pushes it one tile and the player stays put (pret src/field_player_avatar.c
# TryPushBoulder, StartStrengthAnim), so the search state holds the boulders' tiles.
# Planner cost of one push: a stop, the push animation (with the field controls locked)
# and, the first time, Strength's question; open ground of a few steps is preferred.
BOULDER_PUSH_COST = 4.0
# Pushes one planned walk may make, and nodes a search with boulders may open: each push
# makes the same tile another search state, so the search is kept bounded. The live puzzle
# needs 2 pushes and a few hundred nodes.
BOULDER_MAX_PUSHES = 8
BOULDER_SEARCH_MAX_NODES = 60000
# Grid symbols a boulder may be pushed onto: floor and tall grass. pret TryPushBoulder wants
# GetCollisionAtCoords == COLLISION_NONE and no MetatileBehavior_IsNonAnimDoor; doors,
# stairs and other warps ('D'/'S'), ledges, water and walls are refused here, the doors and
# warps beyond what the game refuses (a boulder on a warp would cover the way out).
BOULDER_FLOOR = frozenset({".", "~"})
# pret IsMetatileDirectionallyImpassable reads these behaviors (MB_IMPASSABLE_*) and
# MetatileBehavior_IsNonAnimDoor these (MB_NON_ANIMATED_DOOR, MB_WATER_DOOR,
# MB_DEEP_SOUTH_WARP): either stops a boulder.
BOULDER_REFUSED_BEHAVIORS = ("IMPASSABLE", "NON_ANIMATED_DOOR", "WATER_DOOR", "DEEP_SOUTH_WARP")


def tile_behavior_name(tile) -> str:
    """The MetatileBehavior name of a (tile id, behavior, collision, elevation) tile, or ''."""
    if not tile or len(tile) < 2:
        return ""
    behavior = tile[1]
    name = getattr(behavior, "name", None)
    if name is not None:
        return str(name)
    if MetatileBehavior is None:
        return ""
    try:
        return MetatileBehavior(int(behavior)).name
    except (TypeError, ValueError):
        return ""


def is_surf_water(tile) -> bool:
    """Water a surfer can move onto: a SURF_WATER_BEHAVIORS tile without a collision bit."""
    return tile_behavior_name(tile) in SURF_WATER_BEHAVIORS and not (len(tile) > 2 and tile[2])


def is_surf_start(tile) -> bool:
    """Water Surf can be started onto from the shore (SURF_START_BEHAVIORS)."""
    return is_surf_water(tile) and tile_behavior_name(tile) in SURF_START_BEHAVIORS


def is_shore(tile) -> bool:
    """Land a surfer can step off onto, and Surf can start from: elevation 3, no collision."""
    return bool(tile) and len(tile) > 3 and tile[3] == SHORE_ELEVATION and not tile[2] and not is_surf_water(tile)


def elevation_step(walker: Optional[int], there: Optional[int]) -> Optional[int]:
    """The walker's elevation after a step onto a tile at elevation `there`, or None when
    the game refuses the step for an elevation mismatch. None for walker or there means
    unknown: the step is allowed, and an unknown walker stays unknown across a bridge.

    pret src/event_object_movement.c IsElevationMismatchAt(walker's currentElevation):
    no mismatch for a walker at 0 (ELEVATION_TRANSITION), nor onto a tile at 0 or 15
    (ELEVATION_MULTI_LEVEL); any other difference is COLLISION_ELEVATION_MISMATCH in
    GetCollisionAtCoords. ObjectEventUpdateElevation then gives the walker the new tile's
    elevation, except on a tile at 15, where it keeps its own (it also keeps it while the
    tile it came from is 15, but only until the step ends: DoGroundEffects_OnFinishStep
    runs it again with both coords on the new tile). So a tile at 0 joins every floor next
    to it, and a bridge carries each floor across at its own height.
    """
    if there is None:
        return walker
    if walker not in (None, 0) and there not in (0, BRIDGE_ELEVATION) and there != walker:
        return None
    return walker if there == BRIDGE_ELEVATION else there


def elevation_allows(here, there, from_symbol: str, dest_symbol: str, walker: Optional[int]) -> bool:
    """Whether the walker, at elevation `walker` on tile `here` (None: here's own, or
    unknown on a bridge), passes the game's elevation check stepping onto tile `there`
    (layout tuples; from_symbol/dest_symbol their grid symbols).

    elevation_step decides, with two exceptions. A ledge is jumped over whatever the
    elevations: pret CheckForObjectEventCollision tests ShouldJumpLedge after
    GetCollisionAtCoords and returns COLLISION_LEDGE_JUMP even on a mismatch, and the
    walker lands with the landing tile's elevation. A warp tile ends the walk on the
    step: doors fire from the tile before (pret TryDoorWarp; warp_events at elevation 0
    match any walker, GetWarpEventAtPosition), and the walk goes to no other warp
    (find_path blocks them).
    """
    if here is None or there is None or len(here) < 4 or len(there) < 4:
        return True
    if from_symbol in LEDGE_SYMBOLS or dest_symbol in LEDGE_SYMBOLS:
        return True
    if dest_symbol in ("S", "D") or any(word in tile_behavior_name(there) for word in WARP_BEHAVIOR_WORDS):
        return True
    if walker is None and here[3] != BRIDGE_ELEVATION:
        walker = here[3]
    return elevation_step(walker, there[3]) is not None


def walk_steps(tiles, grid, x: int, y: int, walker: Optional[int], surf: bool, walls):
    """(nx, ny, the walker's elevation there) for each tile a walker at elevation `walker`
    on (x, y) can step onto: the tile planner's rules (_can_move_to with the blocked set's
    collision rule) for the floods that don't plan a walk, the route areas
    (game_tools._walk_areas_without) and the ASCII map's reachable floor (porymap_state).

    tiles are the layout tuples, grid the symbols by row; walls are symbols never entered
    (water among them is still entered where water_step allows it). Ledge directions are
    the caller's: a ledge tile is entered from any side here.
    """
    here = tiles[y][x] if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]) else None
    if walker is None and here is not None and len(here) > 3 and here[3] != BRIDGE_ELEVATION:
        walker = here[3]
    from_symbol = grid[y][x]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if not (0 <= ny < len(grid) and 0 <= nx < len(grid[ny])):
            continue
        there = tiles[ny][nx] if ny < len(tiles) and nx < len(tiles[ny]) else None
        symbol = grid[ny][nx]
        wet = water_step(here, there, symbol, walker, surf)
        if wet is False:
            continue
        if wet is None:
            if symbol in walls:
                continue
            # pret MapGridGetCollisionAt (as _get_blocked_positions).
            if there is not None and len(there) > 2 and there[2] and symbol not in PASSABLE_WITH_COLLISION:
                continue
            if not elevation_allows(here, there, from_symbol, symbol, walker):
                continue
        if there is None or len(there) < 4:
            yield nx, ny, walker
        else:
            yield nx, ny, walker if there[3] == BRIDGE_ELEVATION else there[3]


def arrival_elevation(tile_elevation: Optional[int]) -> Optional[int]:
    """The player's elevation on the tile a warp lands on: pret InitPlayerAvatar spawns the
    player at ELEVATION_TRANSITION (0) and ObjectEventUpdateElevation then takes the tile's,
    except on a tile at 15, where it returns early and 0 stays."""
    if tile_elevation is None:
        return None
    return 0 if tile_elevation == BRIDGE_ELEVATION else tile_elevation


def elevations_compatible(a: Optional[int], b: Optional[int]) -> bool:
    """pret AreElevationsCompatible: two objects collide only at the same elevation, or when
    either is at 0 (DoesObjectCollideWithObjectAt). Unknown counts as compatible."""
    return a is None or b is None or a == 0 or b == 0 or a == b


def water_step(here, there, there_symbol: str, walker: Optional[int], surf: bool) -> Optional[bool]:
    """Whether a step from tile `here` to tile `there` (layout tuples) into, on or out of
    Surf water is possible; None for a step that has nothing to do with water (the
    elevation rule decides, elevation_step).

    pret src/field_player_avatar.c: Surf starts only facing surfable water from
    elevation 3 (IsPlayerFacingSurfableFishableWater); a surfer (elevation 1, the
    water's) moves over water and, like any walker, under a bridge (elevation 15 keeps
    its elevation, ObjectEventUpdateElevation); stepping onto an elevation-3 tile
    ends Surf (CanStopSurfing). Any other step between water and land is refused,
    though the game might allow some (an elevation-0 tile next to water): a missing
    way is better than a walk that ends in the water's edge.
    """
    if here is None or there is None or len(here) < 4 or len(there) < 4:
        return None
    from_water, to_water = is_surf_water(here), is_surf_water(there)
    # Under a bridge the surfer keeps the water's elevation (walker). Only a party that can
    # Surf is ever there: ROUTE 110's and 120's elevation-1 bridge ends (BRIDGE_OVER_*)
    # are land at the water's elevation, and from them a walker on foot is no surfer.
    surfing = from_water or (here[3] == BRIDGE_ELEVATION and walker == WATER_ELEVATION and bool(surf))
    if surfing:
        if to_water or there[3] in (BRIDGE_ELEVATION, SHORE_ELEVATION):
            return True
        # A door in the water (MB_WATER_DOOR, MB_WATER_SOUTH_ARROW_WARP: Scorched Slab).
        return there_symbol in ("D", "S") and "WATER" in tile_behavior_name(there)
    if not to_water:
        return None
    return bool(surf) and is_shore(here) and is_surf_start(there)


@dataclass
class Node:
    """Represents a position in the pathfinding grid."""

    x: int
    y: int
    g_cost: float = 0  # Cost from start
    h_cost: float = 0  # Heuristic cost to goal
    f_cost: float = 0  # Total cost (g + h)
    parent: Optional["Node"] = None
    carried: Optional[int] = None  # the walker's elevation (kept across BRIDGE_ELEVATION tiles)
    gates: tuple | None = None  # rotating gates' orientations here (rotating_gates.py)
    boulders: tuple | None = None  # Strength boulders' tiles here, sorted (None: not modelled)
    push: tuple | None = None  # (button, boulder tile, tile it went to) when this step pushed one
    pushes: int = 0  # pushes made on the way here

    def __post_init__(self):
        self.f_cost = self.g_cost + self.h_cost

    def __lt__(self, other):
        """
        Comparison for heapq tie-breaking.
        When f_cost is equal, prefer nodes with higher g_cost (closer to goal),
        or if still equal, prefer nodes closer to goal by h_cost (lower h_cost = closer).
        This ensures deterministic, optimal path selection.
        """
        if self.f_cost != other.f_cost:
            return self.f_cost < other.f_cost
        # Tie-breaking: prefer nodes with higher g_cost (explored more = closer to goal)
        if self.g_cost != other.g_cost:
            return self.g_cost > other.g_cost
        # Secondary tie-breaking: prefer nodes closer to goal (lower h_cost)
        return self.h_cost < other.h_cost

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


class Pathfinder:
    """
    A* pathfinding implementation for Pokemon Emerald navigation.

    Handles collision detection, NPC avoidance, and terrain considerations.
    """

    def __init__(self, collision_map: Optional[Dict] = None, allow_diagonal: bool = False):
        """
        Initialize the pathfinder.

        Args:
            collision_map: Dictionary containing collision data for the current map
            allow_diagonal: Whether to allow diagonal movement (default False for Pokemon)
        """
        self.collision_map = collision_map or {}
        self.allow_diagonal = allow_diagonal
        self.tile_connectivity = {}  # Cache of valid movement directions per tile
        # Cost of a tall-grass step. The server sets 1.0 while a training objective is
        # current (David, 2026-11-03: "make sure it doesn't avoid tall grass now. I want it
        # to battle pokemon") and GRASS_TILE_COST_MULTIPLIER otherwise.
        self.grass_cost = GRASS_TILE_COST_MULTIPLIER
        # The rotating gates' orientations at the end of the last path found (None when
        # the map has none): the walk compares them with the game's afterwards.
        self.last_gate_orientations = None
        # The boulder pushes of the last path found, in walking order: {"index": the
        # push's place in the buttons, "button", "boulder": its tile, "to": where it
        # goes}. The press does not move the player, so callers walk up to the first push
        # and push it themselves (a push locks the field controls, which a walk's
        # movement check would call an interruption).
        self.last_pushes: List[Dict] = []
        self.last_path_buttons: List[str] = []

    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        game_state: Dict,
        max_distance: int = 150,
        variance: Optional[str] = None,
        consider_npcs: bool = False,
        allow_partial: bool = True,
        blocked_coords: Optional[List[Tuple[int, int]]] = None,
    ) -> Optional[List[str]]:
        """
        Find a path from start to goal using A* algorithm.

        Args:
            start: Starting position (x, y) - ROM coordinates (0,0 = top-left of current map)
            goal: Goal position (x, y) - ROM coordinates
            game_state: Current game state with map data, NPCs, etc.
            max_distance: Maximum distance to search (default 150, handles large Pokemon maps)
            variance: Optional variance level ('low', 'medium', 'high') controlling
                how many initial moves must differ when sampling alternative paths.
                When None or invalid, pathfinding remains deterministic.
            consider_npcs: Whether to consider NPC positions as blocked (default False)
            allow_partial: Whether to allow partial paths to closest reachable point (default True)
            blocked_coords: Optional list of additional coordinates to treat as blocked (useful for gym puzzles)

        Returns:
            List of button commands to reach the goal, or None if no path found
        """
        # Extract map data from game state
        map_data = self._extract_map_data(game_state)
        self.last_gate_orientations = None
        self.last_pushes = []
        self.last_path_buttons = []

        # CRITICAL: Require porymap data - no fallback allowed
        if not map_data:
            logger.error(f"Pathfinding: No map data available from {start} to {goal}")
            return None

        # Among rotating gates a part-way walk toward the closest tile by distance turns
        # gates on the way for nothing (the way may lie in the other direction), and the
        # variance alternatives are planned without the gates: only whole planned walks.
        # The same with boulders the walk may push: a push can't be taken back.
        boulders = {tuple(b) for b in map_data.get("boulders") or []}
        if map_data.get("rotating_gates") is not None or (boulders and map_data.get("strength")):
            allow_partial = False
            variance = None

        # Only use porymap data - reject other map types
        if map_data.get("type") != "porymap":
            logger.error(
                f"Pathfinding: Map data type is '{map_data.get('type')}', but only 'porymap' is allowed. Rejecting pathfinding."
            )
            return None

        if "grid" not in map_data or not map_data.get("grid"):
            logger.error(f"Pathfinding: Porymap data missing 'grid' field. Cannot pathfind.")
            return None

        # Get current location for debugging
        location_name = game_state.get("player", {}).get("location", "Unknown")
        game_type = os.environ.get("GAME_TYPE", "emerald").upper()

        # Get blocked positions (walls, NPCs, water, etc.)
        # CRITICAL: Exclude start position - player is there so it must be walkable
        blocked = self._get_blocked_positions(
            game_state, map_data, start_pos=start, goal_pos=goal, consider_npcs=consider_npcs
        )

        # Add additional blocked coordinates (e.g. gate arms discovered by agent)
        if blocked_coords:
            for coord in blocked_coords:
                if coord != start:  # Never block where we are
                    blocked.add(tuple(coord))
                    logger.info(f"🚫 Pathfinding: Manually blocked {coord}")

        # Get warp positions and ensure they're walkable (doors/stairs)
        warps = self._get_warp_positions(game_state, map_data)
        logger.debug(f"Found {len(warps)} warp positions: {list(warps)[:10]}")  # Show first 10
        for warp_pos in warps:
            if game_type == "RED":
                # Red: only unblock the goal warp. Stepping on any other door/stair
                # tile warps the player into an undesired building.
                if warp_pos == goal:
                    was_blocked = warp_pos in blocked
                    blocked.discard(warp_pos)
                    if was_blocked:
                        logger.info(f"🚪 Unblocked goal warp at {warp_pos}")
                continue
            # Emerald: unblock all warps (original behavior)
            was_blocked = warp_pos in blocked
            blocked.discard(warp_pos)  # Warps are always walkable
            if was_blocked:
                logger.info(f"🚪 Unblocked warp at {warp_pos} (was blocked)")
            # Emerald-only: unblock tile ABOVE warp (porymap_json_builder shifts warp coords down by 1)
            if game_type == "EMERALD":
                above_pos = (warp_pos[0], warp_pos[1] - 1)
                if above_pos[1] >= 0:  # Check it's not out of bounds
                    was_above_blocked = above_pos in blocked
                    blocked.discard(above_pos)
                    if was_above_blocked:
                        logger.info(f"🚪 Unblocked position above warp: {above_pos} (warp at {warp_pos}, was blocked)")

        # SAFEGUARD: Explicitly unblock all 'D' (door) and 'S' (stairs) tiles in the grid
        # Red: skip — D/S are intentionally blocked (stepping on them warps the player)
        if game_type != "RED" and "grid" in map_data and map_data.get("type") == "porymap":
            grid = map_data["grid"]
            doors_and_stairs = []
            for y, row in enumerate(grid):
                for x, cell in enumerate((row if isinstance(row, (list, str)) else [])):
                    if cell in ["D", "S"]:
                        pos = (x, y)
                        if pos in blocked:
                            logger.warning(f"Door/stairs at {pos} ('{cell}') was in blocked set - removing!")
                        blocked.discard(pos)
                        doors_and_stairs.append(pos)
            if doors_and_stairs:
                logger.debug(f"Explicitly unblocked {len(doors_and_stairs)} door/stairs tiles: {doors_and_stairs[:5]}")
        # Only the warp being walked to may be stepped on: stepping on any other warps the
        # player away. Seen live: from ROUTE 110 (16,89) the way north ran through the
        # Cycling Road gate's door at (16,88), and every walk north ended inside the gate.
        # The tiles are map.json's warp_events ("raw_warps"): the grid's 'D' marks whole
        # floors in places (MAUVILLE CITY BIKE SHOP), and the builder's "warps" are moved
        # next to their door. When the goal is a warp (raw or the builder's moved copy), the
        # raw warp within one tile of it is the one being walked to. A goal that is not a
        # warp opens no door: seen in a test, train's walk to grass beside ROUTE 117's Day
        # Care door stepped on the door.
        goal_is_warp = goal in {
            (w.get("x"), w.get("y")) for w in map_data.get("warps") or [] if isinstance(w, dict)
        } or goal in {tuple(p) for p in map_data.get("raw_warps") or []}
        for pos in map_data.get("raw_warps") or []:
            pos = tuple(pos)
            if pos == start:
                continue
            if goal_is_warp and abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]) <= 1:
                continue
            blocked.add(pos)

        # Red: unblock cuttable trees ('t') adjacent to the player (they can use Cut)
        if game_type == "RED" and "grid" in map_data:
            grid = map_data["grid"]
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                adj_pos = (start[0] + dx, start[1] + dy)
                ax, ay = adj_pos
                if 0 <= ay < len(grid) and isinstance(grid[ay], (list, str)):
                    row = grid[ay]
                    if 0 <= ax < len(row) and row[ax] == "t":
                        was_blocked = adj_pos in blocked
                        blocked.discard(adj_pos)
                        if was_blocked:
                            logger.info(f"🌳 Unblocked cuttable tree at {adj_pos} (adjacent to player at {start})")

        # Boulders the party can't push (no STRENGTH usable) are walls wherever they stand
        # now, whether or not people are being walked round; pushable ones are _astar's
        # search state instead.
        if boulders and not map_data.get("strength"):
            blocked |= boulders - {start}

        # Ensure start is never blocked
        blocked.discard(start)

        # Ensure door ('D') and stairs ('S') tiles at goal position are always walkable
        if "grid" in map_data and map_data.get("type") == "porymap":
            grid = map_data["grid"]
            goal_x, goal_y = goal
            if 0 <= goal_y < len(grid) and isinstance(grid[goal_y], (list, str)):
                row = grid[goal_y]
                goal_cell = row[goal_x] if 0 <= goal_x < len(row) else "?"
                if goal_cell in ["D", "S"]:
                    was_blocked_before = goal in blocked
                    blocked.discard(goal)
                    logger.info(
                        f"🚪 Goal {goal} is on door/stairs tile '{goal_cell}' - ensuring walkable (was blocked: {was_blocked_before})"
                    )

        # CRITICAL: Always unblock the goal if it's a warp position, regardless of tile type
        # This handles cases where warps were adjusted (e.g., moved down 1 tile)
        if goal in warps:
            was_blocked_before = goal in blocked
            blocked.discard(goal)
            logger.info(f"🚪 Goal {goal} is a warp position - ensuring walkable (was blocked: {was_blocked_before})")
        else:
            logger.debug(f"Goal {goal} is NOT in warp positions list")

        # Check if goal is blocked - if so, find nearest reachable position first
        goal_was_blocked = goal in blocked
        if goal_was_blocked:
            # Log detailed info about why it's blocked
            if "grid" in map_data and map_data.get("type") == "porymap":
                grid = map_data["grid"]
                goal_x, goal_y = goal
                if 0 <= goal_y < len(grid) and isinstance(grid[goal_y], (list, str)):
                    row = grid[goal_y]
                    goal_cell = row[goal_x] if 0 <= goal_x < len(row) else "?"
                    logger.error(f"🚫 Goal {goal} is STILL BLOCKED! Grid cell: '{goal_cell}', Is warp: {goal in warps}")
            else:
                logger.warning(f"Goal {goal} is STILL BLOCKED after unblocking attempts!")
        if goal_was_blocked:
            logger.info(f"Goal {goal} is on a blocked tile, finding nearest reachable position")
            # Temporarily add goal back to blocked for nearest search
            blocked.add(goal)
            nearest = self._find_nearest_reachable(start, goal, blocked, map_data)
            if nearest and nearest != start:
                logger.info(f"Using nearest reachable position {nearest} instead of blocked goal {goal}")
                goal = nearest
            else:
                logger.warning(f"Could not find reachable position near blocked goal {goal}")
            # Remove goal from blocked now that we have a new goal
            blocked.discard(goal)
        else:
            # Goal is not blocked, but ensure it's not in blocked set
            blocked.discard(goal)

        # Log grid cell status for debugging
        if map_data.get("type") == "porymap" and "grid" in map_data:
            grid = map_data["grid"]
            start_x, start_y = start
            goal_x, goal_y = goal
            if 0 <= start_y < len(grid) and isinstance(grid[start_y], (list, str)):
                row = grid[start_y]
                if isinstance(row, str):
                    start_cell = row[start_x] if 0 <= start_x < len(row) else "?"
                else:
                    start_cell = row[start_x] if 0 <= start_x < len(row) else "?"
                logger.debug(
                    f"Start ({start_x}, {start_y}) in grid: '{start_cell}' {'(was blocked, now unblocked)' if start_cell == '#' else ''}"
                )
            if 0 <= goal_y < len(grid) and isinstance(grid[goal_y], (list, str)):
                row = grid[goal_y]
                if isinstance(row, str):
                    goal_cell = row[goal_x] if 0 <= goal_x < len(row) else "?"
                else:
                    goal_cell = row[goal_x] if 0 <= goal_x < len(row) else "?"
                goal_pos = (goal_x, goal_y)
                logger.debug(
                    f"Goal ({goal_x}, {goal_y}) in grid: '{goal_cell}' {'(warp - unblocked)' if goal_pos in warps else ''}"
                )

        logger.debug(f"Total blocked positions: {len(blocked)}, warps: {len(warps)}")

        # Run A* algorithm (returns both path and best_node if path fails). Trigger tiles
        # (floor switches, scene triggers) are walked round when there is another way:
        # seen live in MAUVILLE CITY GYM, the walk to Wattson crossed two floor switches
        # that flipped the barriers back. When the only way is over one (a story scene to
        # walk into), it is walked over.
        # Rocks and trees the party can clear are walkable here (they are objects, so they
        # were blocked above); _astar charges CLEARABLE_TILE_COST for stepping on one.
        blocked -= {tuple(t) for t in map_data.get("clearable") or []}
        triggers = {tuple(t) for t in map_data.get("triggers") or []} - {start, goal}
        result = self._astar(start, goal, blocked | triggers, map_data, max_distance) if triggers else None
        if result is None or isinstance(result, tuple):
            result = self._astar(start, goal, blocked, map_data, max_distance)
        else:
            blocked = blocked | triggers  # the variance candidates below avoid them too

        if isinstance(result, tuple):
            # A* returned (None, best_node) - no complete path but found closest point
            path, best_node = result

            # Only use partial paths if allowed
            if not allow_partial:
                return None

            if best_node and (best_node.x, best_node.y) != start:
                logger.info(
                    f"Using partial path to closest point: {(best_node.x, best_node.y)} (distance {best_node.h_cost:.1f} from goal)"
                )

                # Return partial path (up to 25 steps) toward the goal
                partial_path = self._reconstruct_path(best_node)
                if len(partial_path) > 25:
                    partial_path = partial_path[:15]
                    logger.info(f"Limiting partial path to first 25 steps")

                logger.info(f"Partial path: {len(partial_path)} steps toward goal")
                return self._path_to_buttons(partial_path)
            else:
                logger.warning(f"No progress possible toward {goal}")
                return None
        else:
            path = result

        # A push repeats the player's tile in the path, so its press comes from the nodes.
        base_buttons = list(self.last_path_buttons) if self.last_pushes else self._path_to_buttons(path)

        if variance:
            variance_steps = VARIANCE_TO_STEPS.get(variance)
            if variance_steps:
                variant_buttons = self._generate_variance_candidates(
                    start=start,
                    goal=goal,
                    blocked=blocked,
                    map_data=map_data,
                    max_distance=max_distance,
                    variance_steps=variance_steps,
                    base_path_buttons=base_buttons,
                )
                if variant_buttons:
                    chosen_buttons = random.choice(variant_buttons)
                    logger.info(
                        f"Variance '{variance}' selected alternative path (prefix {chosen_buttons[:variance_steps]})"
                    )
                    return chosen_buttons
                else:
                    logger.debug(f"No alternative paths found for variance '{variance}', using base path")

        return base_buttons

    def _extract_map_data(self, game_state: Dict) -> Optional[Dict]:
        """Extract map data from game state. ONLY returns porymap data - no fallbacks."""
        # ONLY use porymap data (ground truth from JSON) - no fallbacks
        map_data = game_state.get("map", {})
        if map_data and "porymap" in map_data:
            porymap = map_data["porymap"]
            if porymap.get("grid"):
                # Return porymap data structure for pathfinding
                result = {
                    "type": "porymap",
                    "grid": porymap["grid"],  # ASCII grid: [['.', '#', ...], ...]
                    "objects": porymap.get("objects", []),
                    # Live positions (server.game_tools.load_pathfinding_state) win over the
                    # older map-level copy.
                    "runtime_objects": porymap.get("runtime_objects") or map_data.get("object_events", []),
                    "width": porymap.get("dimensions", {}).get("width", 0),
                    "height": porymap.get("dimensions", {}).get("height", 0),
                    "warps": porymap.get("warps", []),  # Include warps for pathfinding
                    # The tiles that really warp (map.json warp_events), see find_path.
                    "raw_warps": porymap.get("raw_warps", []),
                    # Coord-event triggers that would fire now (game_tools.active_triggers).
                    "triggers": porymap.get("triggers", []),
                    # Rocks/trees the party can clear (game_tools.clearable_obstacles).
                    "clearable": porymap.get("clearable", []),
                    # The party can Surf now (game_tools.can_surf): water joins the walk.
                    "surf": bool(porymap.get("surf")),
                    # The map's rotating gate puzzle and the gates' live orientations
                    # (game_tools.load_pathfinding_state): steps push and turn them.
                    "rotating_gates": porymap.get("rotating_gates"),
                    "gate_orientations": porymap.get("gate_orientations"),
                    # Strength boulders' tiles now and whether the party can push them
                    # (game_tools.load_pathfinding_state): pushes join the search.
                    "boulders": porymap.get("boulders") or [],
                    "strength": bool(porymap.get("strength")),
                    # The player's tile and live elevation (game_tools.load_pathfinding_state):
                    # the walker's elevation at the start of a walk from there.
                    "start_position": tuple(porymap["player_position"]) if porymap.get("player_position") else None,
                    "start_elevation": porymap.get("player_elevation"),
                }

                # Include raw tiles with elevation if available
                # Raw tiles are needed for elevation checking
                if "raw_tiles" in porymap:
                    result["raw_tiles"] = porymap["raw_tiles"]
                elif "tiles" in map_data:
                    # Fallback: check if tiles are in map_data directly
                    result["raw_tiles"] = map_data["tiles"]

                return result

        # No porymap data available - return None (no fallback)
        return None

    def _build_tile_connectivity(self, map_data: Dict) -> Dict[Tuple[int, int], Dict[str, bool]]:
        """
        Build a connectivity map showing which directions are valid from each tile.

        Returns dict: {(x, y): {'up': bool, 'down': bool, 'left': bool, 'right': bool}}
        """
        connectivity = {}

        if "grid" not in map_data or "raw_tiles" not in map_data:
            return connectivity

        grid = map_data["grid"]
        raw_tiles = map_data["raw_tiles"]
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        # For each tile, check if movement in each direction is valid
        for y in range(height):
            for x in range(width):
                pos = (x, y)
                connectivity[pos] = {"up": False, "down": False, "left": False, "right": False}

                # Check each direction
                neighbors = [("up", (x, y - 1)), ("down", (x, y + 1)), ("left", (x - 1, y)), ("right", (x + 1, y))]

                for direction, (nx, ny) in neighbors:
                    # Check bounds
                    if 0 <= nx < width and 0 <= ny < height:
                        # Use _can_move_to to check if movement is valid
                        if self._can_move_to(pos, (nx, ny), map_data):
                            connectivity[pos][direction] = True

        return connectivity

    def _get_blocked_positions(
        self,
        game_state: Dict,
        map_data: Dict,
        start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
        consider_npcs: bool = False,
    ) -> Set[Tuple[int, int]]:
        """
        Get all blocked positions on the current map.
        ONLY uses porymap ASCII grid data - no fallbacks.

        Args:
            start_pos: Starting position - will NEVER be marked as blocked (player is there, so it is walkable)
            goal_pos: Goal position - excluded from stationary NPC blocking so we can reach the target NPC.
            consider_npcs: Whether to consider NPC positions as blocked (default False)

        Returns set of (x, y) tuples that are not walkable.
        Note: Ledges are handled separately via directional checks.
        """
        blocked = set()

        # Get map dimensions
        width = map_data.get("width", 50)
        height = map_data.get("height", 50)

        # REQUIRED: Only use porymap ASCII grid (ground truth data)
        if map_data.get("type") != "porymap" or "grid" not in map_data:
            logger.error(
                f"_get_blocked_positions: Expected porymap data with grid, got type='{map_data.get('type')}', has_grid={'grid' in map_data}"
            )
            return blocked

        # Get start elevation if available (for blocking tiles at different elevations)
        start_elevation = None
        if start_pos and "raw_tiles" in map_data:
            raw_tiles = map_data["raw_tiles"]
            sx, sy = start_pos
            if 0 <= sy < len(raw_tiles) and 0 <= sx < len(raw_tiles[sy]):
                start_tile = raw_tiles[sy][sx]
                if start_tile and len(start_tile) >= 4:
                    start_elevation = start_tile[3] if len(start_tile) > 3 else 0

        grid = map_data["grid"]
        raw_tiles = map_data.get("raw_tiles", None)

        # Check if player is in water (surfing)
        player_in_water = False
        if start_pos and 0 <= start_pos[1] < len(grid):
            start_row = grid[start_pos[1]]
            if start_pos[0] < len(start_row):
                player_in_water = start_row[start_pos[0]] == "W"

        def surfable(x: int, y: int) -> bool:
            """Water the party can Surf on (map_data "surf"); where the walk may enter and
            leave it is _water_step's call."""
            if not map_data.get("surf") or not raw_tiles or not (0 <= y < len(raw_tiles)):
                return False
            return 0 <= x < len(raw_tiles[y]) and is_surf_water(raw_tiles[y][x])

        game_type = os.environ.get("GAME_TYPE", "emerald")
        red_block_symbols = {"#", "X", "!", "P", "T", "B", "^", "U", "C", "=", "t", "D", "S", "G"}  # "!": sign (blocked); "?": hidden item (walkable); "t": cuttable tree; "D"/"S": doors/stairs (warp player out of map); "G": Card Key gate
        block_symbols = red_block_symbols if game_type.upper() == "RED" else {"#", "X"}

        for y, row in enumerate(grid):
            if isinstance(row, list):
                for x, cell in enumerate(row):
                    # In porymap ASCII:
                    # Walkable: '.' (normal), '~' (grass), 'S' (stairs/warps), 'D' (doors), '←↓↑→' (ledges), '^' (different elevation)
                    # Blocked: '#' (walls), 'X' (out of bounds), 'W' (water - requires Surf)
                    # Special: '&' (cycling road) - block if at different elevation
                    # CRITICAL: 'D' (door) and 'S' (stairs) tiles are ALWAYS walkable

                    pos = (x, y)

                    # Always block walls and out of bounds
                    if cell in block_symbols:
                        # CRITICAL: Never block the starting position - player is there so it must be walkable
                        if start_pos and pos == start_pos:
                            logger.debug(f"Excluding start position {start_pos} from blocked set (player is there)")
                            continue
                        blocked.add(pos)
                    # Water: block if player is NOT in water (can't surf without Surf ability)
                    # Allow if player IS in water (can navigate water freely while surfing)
                    elif cell == "W":
                        if start_pos and pos == start_pos:
                            # Starting position is always walkable
                            continue
                        if not player_in_water and not surfable(x, y):
                            # Player on ground can't access water
                            blocked.add(pos)
                        # else: player in water, don't block other water tiles
                    # Never block doors or stairs
                    elif cell in ["D", "S"]:
                        continue
                    # Block cycling road tiles if at a different elevation from start
                    # UNLESS there's a walkable path underneath (. & & & . pattern)
                    # OR it's a ladder connecting different elevations (check adjacent tiles)
                    elif cell == "&" and start_elevation is not None and raw_tiles and game_type.upper() == "RED":
                        if 0 <= y < len(raw_tiles) and 0 <= x < len(raw_tiles[y]):
                            tile = raw_tiles[y][x]
                            if tile and len(tile) >= 4:
                                tile_elevation = tile[3] if len(tile) > 3 else 0
                                # If at same elevation, it's walkable (don't block)
                                if tile_elevation == start_elevation:
                                    continue

                                # Check if this is a ladder (has walkable tiles nearby, including other ladders)
                                is_ladder = False
                                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                                    nx, ny = x + dx, y + dy
                                    if 0 <= ny < len(grid) and 0 <= nx < len(row if isinstance(row, str) else row):
                                        neighbor_char = (
                                            (grid[ny][nx] if isinstance(grid[ny], str) else grid[ny][nx])
                                            if ny < len(grid)
                                            else "#"
                                        )
                                        # Check for walkable tiles OR other ladder tiles (for chaining)
                                        if neighbor_char in [".", "~", "S", "D", "&"]:
                                            # Found walkable/ladder neighbor - check its elevation
                                            if 0 <= ny < len(raw_tiles) and 0 <= nx < len(raw_tiles[ny]):
                                                neighbor_tile = raw_tiles[ny][nx]
                                                if neighbor_tile and len(neighbor_tile) >= 4:
                                                    neighbor_elev = neighbor_tile[3] if len(neighbor_tile) > 3 else 0
                                                    # If neighbor is at start elevation OR is another ladder, this is part of a ladder chain
                                                    if neighbor_elev == start_elevation or neighbor_char == "&":
                                                        is_ladder = True
                                                        break

                                # If it's a ladder, don't block it
                                if is_ladder:
                                    continue

                                # If bridge is at HIGHER elevation, check for ground path underneath
                                if tile_elevation > start_elevation:
                                    # Check for . & & & . pattern (ground path under bridge)
                                    has_ground_path = False

                                    # Search left through consecutive bridge tiles to find ground
                                    left_walkable = False
                                    search_x = x - 1
                                    while search_x >= 0 and search_x < len(row):
                                        search_char = row[search_x]
                                        if search_char == "&":
                                            search_x -= 1
                                        elif search_char in [".", "~"]:
                                            # Found walkable ground - check elevation
                                            if search_x < len(raw_tiles[y]):
                                                search_tile = raw_tiles[y][search_x]
                                                if search_tile and len(search_tile) >= 4:
                                                    search_elev = search_tile[3] if len(search_tile) > 3 else 0
                                                    if search_elev == start_elevation:
                                                        left_walkable = True
                                            break
                                        else:
                                            break

                                    # Search right through consecutive bridge tiles to find ground
                                    right_walkable = False
                                    search_x = x + 1
                                    while search_x < len(row):
                                        search_char = row[search_x]
                                        if search_char == "&":
                                            search_x += 1
                                        elif search_char in [".", "~"]:
                                            # Found walkable ground - check elevation
                                            if search_x < len(raw_tiles[y]):
                                                search_tile = raw_tiles[y][search_x]
                                                if search_tile and len(search_tile) >= 4:
                                                    search_elev = search_tile[3] if len(search_tile) > 3 else 0
                                                    if search_elev == start_elevation:
                                                        right_walkable = True
                                            break
                                        else:
                                            break

                                    has_ground_path = left_walkable and right_walkable

                                    # Only block if NO ground path underneath
                                    if not has_ground_path:
                                        blocked.add(pos)
                                # Block cycling road if at different elevation (not ladder, not bridge with underpass)
                                else:
                                    blocked.add(pos)

            elif isinstance(row, str):
                # Handle string rows (each character is a cell)
                for x, cell in enumerate(row):
                    pos = (x, y)

                    # Always block walls and out of bounds
                    if cell in block_symbols:
                        if start_pos and pos == start_pos:
                            logger.debug(f"Excluding start position {start_pos} from blocked set (player is there)")
                            continue
                        blocked.add(pos)
                    # Water: block if player is NOT in water (can't surf without Surf ability)
                    # Allow if player IS in water (can navigate water freely while surfing)
                    elif cell == "W":
                        if start_pos and pos == start_pos:
                            # Starting position is always walkable
                            continue
                        if not player_in_water and not surfable(x, y):
                            # Player on ground can't access water
                            blocked.add(pos)
                        # else: player in water, don't block other water tiles
                    # Never block doors or stairs
                    elif cell in ["D", "S"]:
                        continue
                    # Block cycling road tiles if at a different elevation from start
                    # UNLESS there's a walkable path underneath (. & & & . pattern)
                    # OR it's a ladder connecting different elevations (check adjacent tiles)
                    elif cell == "&" and start_elevation is not None and raw_tiles and game_type.upper() == "RED":
                        if 0 <= y < len(raw_tiles) and 0 <= x < len(raw_tiles[y]):
                            tile = raw_tiles[y][x]
                            if tile and len(tile) >= 4:
                                tile_elevation = tile[3] if len(tile) > 3 else 0
                                # If at same elevation, it's walkable (don't block)
                                if tile_elevation == start_elevation:
                                    continue

                                # Check if this is a ladder (has walkable tiles nearby, including other ladders)
                                is_ladder = False
                                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                                    nx, ny = x + dx, y + dy
                                    if 0 <= ny < len(grid) and 0 <= nx < len(grid[ny] if ny < len(grid) else ""):
                                        neighbor_char = grid[ny][nx] if isinstance(grid[ny], str) else grid[ny][nx]
                                        # Check for walkable tiles OR other ladder tiles (for chaining)
                                        if neighbor_char in [".", "~", "S", "D", "&"]:
                                            # Found walkable/ladder neighbor - check its elevation
                                            if 0 <= ny < len(raw_tiles) and 0 <= nx < len(raw_tiles[ny]):
                                                neighbor_tile = raw_tiles[ny][nx]
                                                if neighbor_tile and len(neighbor_tile) >= 4:
                                                    neighbor_elev = neighbor_tile[3] if len(neighbor_tile) > 3 else 0
                                                    # If neighbor is at start elevation OR is another ladder, this is part of a ladder chain
                                                    if neighbor_elev == start_elevation or neighbor_char == "&":
                                                        is_ladder = True
                                                        break

                                # If it's a ladder, don't block it
                                if is_ladder:
                                    continue

                                # If bridge is at HIGHER elevation, check for ground path underneath
                                if tile_elevation > start_elevation:
                                    # Check for . & & & . pattern (ground path under bridge)
                                    has_ground_path = False

                                    # Search left through consecutive bridge tiles to find ground
                                    left_walkable = False
                                    search_x = x - 1
                                    while search_x >= 0 and search_x < len(row):
                                        search_char = row[search_x] if isinstance(row, str) else row[search_x]
                                        if search_char == "&":
                                            search_x -= 1
                                        elif search_char in [".", "~"]:
                                            # Found walkable ground - check elevation
                                            if search_x < len(raw_tiles[y]):
                                                search_tile = raw_tiles[y][search_x]
                                                if search_tile and len(search_tile) >= 4:
                                                    search_elev = search_tile[3] if len(search_tile) > 3 else 0
                                                    if search_elev == start_elevation:
                                                        left_walkable = True
                                            break
                                        else:
                                            break

                                    # Search right through consecutive bridge tiles to find ground
                                    right_walkable = False
                                    search_x = x + 1
                                    while search_x < len(row):
                                        search_char = row[search_x] if isinstance(row, str) else row[search_x]
                                        if search_char == "&":
                                            search_x += 1
                                        elif search_char in [".", "~"]:
                                            # Found walkable ground - check elevation
                                            if search_x < len(raw_tiles[y]):
                                                search_tile = raw_tiles[y][search_x]
                                                if search_tile and len(search_tile) >= 4:
                                                    search_elev = search_tile[3] if len(search_tile) > 3 else 0
                                                    if search_elev == start_elevation:
                                                        right_walkable = True
                                            break
                                        else:
                                            break

                                    has_ground_path = left_walkable and right_walkable

                                    # Only block if NO ground path underneath
                                    if not has_ground_path:
                                        blocked.add(pos)
                                # Block cycling road if at different elevation (not ladder, not bridge with underpass)
                                else:
                                    blocked.add(pos)

        # pret GetCollisionAtCoords: a tile whose collision bits are set stops the walker
        # whatever it is drawn as. Ledges (the jump goes over them), doors and warps (they
        # fire) and cycling-road tiles (handled above) are the exceptions. Seen building
        # R30: once elevation 0 stopped counting as a mismatch, the walk up BRENDAN'S
        # HOUSE 1F pressed into the TV at (4,4) (drawn 'T', elevation 0, collision 1).
        if raw_tiles and game_type.upper() != "RED":
            for y, row in enumerate(grid):
                for x, cell in enumerate(row):
                    if cell in PASSABLE_WITH_COLLISION or (start_pos and (x, y) == start_pos):
                        continue
                    tile = raw_tiles[y][x] if y < len(raw_tiles) and x < len(raw_tiles[y]) else None
                    if tile and len(tile) > 2 and tile[2]:
                        blocked.add((x, y))

        # People on a bridge (a tile at BRIDGE_ELEVATION) stand at their own elevation there,
        # and pret DoesObjectCollideWithObjectAt stops only a walker whose elevation is
        # compatible with theirs (AreElevationsCompatible): someone on the bridge does not
        # stop a surfer under it. Those are kept per tile for _astar, which knows the
        # walker's elevation; everywhere else a person's elevation is their tile's, and any
        # step the elevation check lets onto that tile meets them.
        bridge_people: Dict[Tuple[int, int], Set[int]] = {}
        map_data["bridge_people"] = bridge_people

        def block_person(pos, elevation):
            x, y = pos
            tile = raw_tiles[y][x] if raw_tiles and 0 <= y < len(raw_tiles) and 0 <= x < len(raw_tiles[y]) else None
            on_bridge = tile is not None and len(tile) > 3 and tile[3] == BRIDGE_ELEVATION
            if on_bridge and elevation is not None and game_type.upper() != "RED":
                bridge_people.setdefault(pos, set()).add(int(elevation))
            else:
                blocked.add(pos)

        # Add NPC/object positions as blocked.
        # Prefer runtime object_events (live positions); fall back to static porymap objects.
        if consider_npcs:
            runtime_objects = map_data.get("runtime_objects") or []
            if runtime_objects:
                for obj in runtime_objects:
                    obj_x = obj.get("current_x", obj.get("x", 0))
                    obj_y = obj.get("current_y", obj.get("y", 0))
                    if obj_x is None or obj_y is None:
                        continue
                    obj_pos = (obj_x, obj_y)
                    if goal_pos and obj_pos == goal_pos:
                        continue
                    block_person(obj_pos, obj.get("elevation"))
            # The map file's objects too: live positions only cover objects near the
            # camera, so a far-off trainer would otherwise vanish from the plan.
            if "objects" in map_data:
                objects = map_data["objects"]
                for obj in objects:
                    obj_x = obj.get("x", 0)
                    obj_y = obj.get("y", 0)
                    # Block objects that are stationary or have limited movement
                    # Block: NONE, FACE_*, WANDER_AROUND (they occupy their position)
                    # Allow: WALK_*, RUN_*, JUMP_* (they actively move around)
                    movement_type = obj.get("movement_type", "")
                    movement_type_upper = movement_type.upper()

                    if (
                        "NONE" in movement_type_upper
                        or "STATIC" in movement_type_upper
                        or "FACE_" in movement_type_upper  # FACE_DOWN, FACE_UP, etc.
                        or "WANDER" in movement_type_upper  # WANDER_AROUND
                        or "LOOK" in movement_type_upper  # LOOK_AROUND, LOOK_AROUND_EX
                        or "BERRY" in movement_type_upper  # BERRY_TREE_GROWTH
                        or not movement_type
                    ):
                        obj_pos = (obj_x, obj_y)
                        if goal_pos and obj_pos == goal_pos:
                            continue
                        block_person(obj_pos, obj.get("elevation"))

        # Add out-of-bounds positions
        for x in range(-1, width + 1):
            blocked.add((x, -1))
            blocked.add((x, height))
        for y in range(-1, height + 1):
            blocked.add((-1, y))
            blocked.add((width, y))

        return blocked

    def _is_tile_blocked(self, tile) -> bool:
        """
        Check if a tile is blocked based on its properties.
        Uses the shared walkability function from map_formatter for consistency.

        Note: Ledges (JUMP_*) are NOT blocked here - they are handled
        via directional validation in _can_move_to().
        """
        if isinstance(tile, str):
            # String representation - check for wall symbols
            return tile in ["#", "X", "█", "▓"]

        # Use shared walkability function (inverted)
        from utils.mapping.map_formatter import is_tile_walkable

        return not is_tile_walkable(tile)

    def _get_npc_positions(self, game_state: Dict) -> Set[Tuple[int, int]]:
        """Get positions of all NPCs on the current map."""
        npcs = set()

        # Check various possible NPC data locations
        npc_data = None
        if "npcs" in game_state:
            npc_data = game_state["npcs"]
        elif "game_state" in game_state and "npcs" in game_state["game_state"]:
            npc_data = game_state["game_state"]["npcs"]

        if npc_data:
            for npc in npc_data:
                if isinstance(npc, dict) and "x" in npc and "y" in npc:
                    npcs.add((npc["x"], npc["y"]))

        return npcs

    def _get_warp_positions(self, game_state: Dict, map_data: Dict) -> Set[Tuple[int, int]]:
        """
        Get positions of warps (doors, stairs, transitions) which are always walkable.

        Args:
            game_state: Current game state
            map_data: Map data structure

        Returns:
            Set of (x, y) positions that are warps (always walkable even if marked as blocked)
        """
        warps = set()

        # Get warps from porymap data
        if map_data.get("type") == "porymap":
            # Check if warps are in a separate field
            # Porymap JSON structure should have warps array
            if "warps" in map_data:
                for warp in map_data["warps"]:
                    if isinstance(warp, dict) and "x" in warp and "y" in warp:
                        warps.add((warp["x"], warp["y"]))

        # Also check game_state for warp information
        map_info = game_state.get("map", {})
        if "warps" in map_info:
            for warp in map_info["warps"]:
                if isinstance(warp, dict) and "x" in warp and "y" in warp:
                    warps.add((warp["x"], warp["y"]))

        # Check porymap JSON data if available
        porymap_data = map_info.get("porymap", {})
        if "warps" in porymap_data:
            for warp in porymap_data["warps"]:
                if isinstance(warp, dict) and "x" in warp and "y" in warp:
                    warps.add((warp["x"], warp["y"]))

        return warps

    def _water_step(
        self,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        map_data: Dict,
        carried_elevation: Optional[int] = None,
    ) -> Optional[bool]:
        """water_step for two tiles of map_data (the walker's elevation is carried_elevation)."""
        raw_tiles = map_data.get("raw_tiles")
        if not raw_tiles:
            return None

        def tile(pos):
            x, y = pos
            if 0 <= y < len(raw_tiles) and 0 <= x < len(raw_tiles[y]):
                found = raw_tiles[y][x]
                return found if found and len(found) > 3 else None
            return None

        here, there = tile(from_pos), tile(to_pos)
        if here is None or there is None:
            return None
        return water_step(here, there, map_data["grid"][to_pos[1]][to_pos[0]], carried_elevation, map_data.get("surf"))

    def _elevation_allows(
        self,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        map_data: Dict,
        carried_elevation: Optional[int] = None,
    ) -> bool:
        """elevation_allows for two tiles of map_data, the walker at carried_elevation."""
        raw_tiles = map_data.get("raw_tiles") or []
        grid = map_data["grid"]

        def tile(pos):
            x, y = pos
            if 0 <= y < len(raw_tiles) and 0 <= x < len(raw_tiles[y]):
                found = raw_tiles[y][x]
                return found if found and len(found) > 3 else None
            return None

        here, there = tile(from_pos), tile(to_pos)
        if here is None or there is None:
            return True
        from_symbol, dest_symbol = grid[from_pos[1]][from_pos[0]], grid[to_pos[1]][to_pos[0]]
        return elevation_allows(here, there, from_symbol, dest_symbol, carried_elevation)

    def _can_move_to(
        self,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        map_data: Dict,
        carried_elevation: Optional[int] = None,
    ) -> bool:
        """
        Check if movement from from_pos to to_pos is valid.

        Handles:
        1. One-way ledges: can only move in the direction of the ledge
        2. Elevation differences: cannot move between tiles at different elevations
           unless there's a special behavior (ledge, slide, stairs, etc.) connecting them

        Args:
            from_pos: Source position (x, y)
            to_pos: Destination position (x, y)
            map_data: Map data dictionary with grid and optionally raw_tiles
            carried_elevation: the walker's elevation when from_pos is a BRIDGE_ELEVATION
                tile (it keeps the one it had before stepping on); None if unknown

        Returns:
            True if movement is valid, False otherwise
        """
        if "grid" not in map_data:
            return True

        grid = map_data["grid"]

        # Check bounds
        to_y, to_x = to_pos[1], to_pos[0]  # grid is [y][x]
        from_y, from_x = from_pos[1], from_pos[0]

        if to_y < 0 or to_y >= len(grid) or to_x < 0 or to_x >= len(grid[0]):
            return False
        if from_y < 0 or from_y >= len(grid) or from_x < 0 or from_x >= len(grid[0]):
            return False

        # Get symbol at destination
        dest_symbol = grid[to_y][to_x]
        from_symbol = grid[from_y][from_x]

        # Calculate movement direction
        dx = to_pos[0] - from_pos[0]  # positive = moving east, negative = moving west
        dy = to_pos[1] - from_pos[1]  # positive = moving south, negative = moving north

        water_step = self._water_step(from_pos, to_pos, map_data, carried_elevation)
        if water_step is not None:
            return water_step

        # Emerald: the game's own elevation rule (elevation_step) with the walker's
        # elevation carried in the search state. The connector guesses below stay for Red.
        # Seen live: on MOSSDEEP CITY go_to walked DOWN from the west shore (16,6) (elevation
        # 3) into (16,7) (elevation 5) again and again; FORTREE CITY's heal walk stepped
        # from (20,13) (3) onto the treetop walkway (20,14) (4). Both refused by the game.
        game_is_red = os.environ.get("GAME_TYPE", "emerald").upper() == "RED"
        if not game_is_red and "raw_tiles" in map_data:
            if not self._elevation_allows(from_pos, to_pos, map_data, carried_elevation):
                return False

        # ========================================================================
        # Elevation Checking Logic
        # ========================================================================
        # Always check elevation via raw_tiles. The grid filter marks unreachable
        # tiles as '#', but adjacent reachable tiles at different elevations still
        # need per-edge validation (e.g. E3 next to E0 ramp).
        # ========================================================================
        walkable_symbols = [".", "~", "S", "D", "←", "→", "↑", "↓", "&", "*"] \
            if os.environ.get("GAME_TYPE", "emerald").upper() == "RED" \
                else [".", "~", "^", "S", "D", "←", "→", "↑", "↓", "&"]
        both_walkable = from_symbol in walkable_symbols and dest_symbol in walkable_symbols

        # Check elevation differences and only allow transitions through valid
        # connectors (E0 tiles, stairs, doors, ladders, ledges).
        # ========================================================================
        if "raw_tiles" in map_data and game_is_red:
            raw_tiles = map_data["raw_tiles"]
            try:
                # Get elevation from raw tiles
                # Raw tiles format: (tile_id, behavior, collision, elevation, ...)
                from_tile = (
                    raw_tiles[from_y][from_x] if from_y < len(raw_tiles) and from_x < len(raw_tiles[from_y]) else None
                )
                to_tile = raw_tiles[to_y][to_x] if to_y < len(raw_tiles) and to_x < len(raw_tiles[to_y]) else None

                if from_tile and to_tile and len(from_tile) >= 4 and len(to_tile) >= 4:
                    from_elevation = from_tile[3] if len(from_tile) > 3 else 0
                    to_elevation = to_tile[3] if len(to_tile) > 3 else 0
                    if from_symbol != "&" and dest_symbol != "&":
                        if to_elevation == BRIDGE_ELEVATION:
                            to_elevation = from_elevation  # never a mismatch
                        elif from_elevation == BRIDGE_ELEVATION:
                            # Stepping off: the walker's own elevation decides.
                            known = carried_elevation not in (None, BRIDGE_ELEVATION)
                            from_elevation = carried_elevation if known else to_elevation
                        # pret IsElevationMismatchAt: a walker at elevation 0, or a tile
                        # at 0, never mismatches. Seen live on JAGGED PASS: the south
                        # ledges are elevation 0 between elevation-3 floor, and the step
                        # off a ledge (the jump's landing) counted as a wall, so no route
                        # led down the pass ("no walkable path" from the top).
                        if from_elevation == 0 or to_elevation == 0:
                            to_elevation = from_elevation

                    # Calculate elevation difference
                    elev_diff = abs(from_elevation - to_elevation)

                    # Check if elevations differ (ANY difference, not just >= 3)
                    if from_elevation != to_elevation:
                        # Skip elevation check for cycling road tiles - they have special handling later
                        if from_symbol == "&" or dest_symbol == "&":
                            pass  # Let cycling road special handling deal with it
                        # Only allow elevation changes through valid connectors:
                        # 1. Destination is stairs/door/warp (check both symbol AND behavior)
                        # 2. Source is stairs/door (exiting to different elevation)
                        # 3. Destination is a ledge (one-way jump down)
                        else:
                            # Check if destination tile has stair/ladder/door behavior
                            # Tiles can be walkable '.' but still be stairs based on behavior
                            to_behavior = to_tile[1] if len(to_tile) > 1 else 0
                            from_behavior = from_tile[1] if len(from_tile) > 1 else 0

                            # Get behavior names — use Red's enum for Red maps so that
                            # RedMetatileBehavior integer codes resolve to the correct names.
                            # RedMetatileBehavior values deliberately match Emerald's for the
                            # connector behaviours (LADDER, NON_ANIMATED_DOOR, WARP_*), so
                            # the substring checks below work identically for both games.
                            to_behavior_name = ""
                            from_behavior_name = ""
                            try:
                                if os.environ.get("GAME_TYPE", "emerald").upper() == "RED":
                                    from pokemon_red_env.utils.red_metatile_behavior import (
                                        RedMetatileBehavior as _BehaviorEnum,
                                    )
                                else:
                                    from pokemon_env.enums import MetatileBehavior as _BehaviorEnum

                                if isinstance(to_behavior, int):
                                    to_behavior_name = _BehaviorEnum(to_behavior).name
                                if isinstance(from_behavior, int):
                                    from_behavior_name = _BehaviorEnum(from_behavior).name
                            except Exception:
                                pass

                            # Check if destination has stair/ladder/door behavior
                            is_dest_connector = (
                                dest_symbol in ["S", "D"]
                                or "LADDER" in to_behavior_name
                                or "STAIRS" in to_behavior_name
                                or "DOOR" in to_behavior_name
                                or "WARP" in to_behavior_name
                            )

                            # Check if source has stair/ladder/door behavior
                            is_source_connector = (
                                from_symbol in ["S", "D"]
                                or "LADDER" in from_behavior_name
                                or "STAIRS" in from_behavior_name
                                or "DOOR" in from_behavior_name
                                or "WARP" in from_behavior_name
                            )

                            # Allow movement if destination is a warp/door/stairs (symbol or behavior)
                            if is_dest_connector:
                                logger.debug(
                                    f"Allowing movement to stairs/door/ladder: ({from_x}, {from_y}) E{from_elevation} -> ({to_x}, {to_y}) E{to_elevation} (behavior: {to_behavior_name})"
                                )
                                return True  # Warps/doors/stairs/ladders explicitly connect elevations

                            # Allow movement if SOURCE is stairs/door/ladder (moving away from connector to different elevation)
                            if is_source_connector:
                                logger.debug(
                                    f"Allowing movement from stairs/door/ladder: ({from_x}, {from_y}) E{from_elevation} -> ({to_x}, {to_y}) E{to_elevation} (behavior: {from_behavior_name})"
                                )
                                return True  # Can exit stairs/doors/ladders to any elevation

                            # Allow movement if it's a directional ledge (one-way jump down)
                            if dest_symbol in ["→", "←", "↑", "↓", "↗", "↖", "↘", "↙"]:
                                logger.debug(
                                    f"Allowing movement to ledge: ({from_x}, {from_y}) E{from_elevation} -> ({to_x}, {to_y}) E{to_elevation}"
                                )
                                return True  # Ledges explicitly allow elevation changes

                            # Allow movement between adjacent walkable tiles ONLY through E0 connector tiles
                            # Pokemon uses E0 tiles as invisible stairs between elevation areas
                            # ALL elevation changes between non-E0 tiles must be blocked
                            walkable_behaviors = [
                                "NORMAL",
                                "MOUNTAIN_TOP",
                                "INDOOR",
                                "CAVE",
                                "TALL_GRASS",
                                "LONG_GRASS",
                                "SHORT_GRASS",
                            ]
                            both_walkable = any(b in from_behavior_name for b in walkable_behaviors) and any(
                                b in to_behavior_name for b in walkable_behaviors
                            )
                            both_walkable_symbols = dest_symbol in [".", "~"] and from_symbol in [".", "~"]

                            if both_walkable and both_walkable_symbols:
                                elev_diff = to_elevation - from_elevation  # Positive = going up, negative = going down

                                # ONLY allow elevation changes if one tile is E0 (connector/stair)
                                if from_elevation == 0 or to_elevation == 0:
                                    logger.debug(
                                        f"Allowing E0 connector transition: ({from_x}, {from_y}) E{from_elevation} -> ({to_x}, {to_y}) E{to_elevation} (Δ{elev_diff:+d})"
                                    )
                                    return True

                                # Block ALL other elevation changes between non-E0 tiles
                                # This includes E3->E4, E4->E3, E3->E1, etc.
                                # Must use E0 connector tiles as stairs
                                logger.info(
                                    f"🚫 Blocking non-E0 elevation change: ({from_x}, {from_y}) E{from_elevation} -> ({to_x}, {to_y}) E{to_elevation} (Δ{elev_diff:+d}) - must use E0 stairs"
                                )
                                return False

                            # Block all other movement between different elevations
                            logger.info(
                                f"🚫 Blocking elevation change: ({from_x}, {from_y}) '{from_symbol}' E{from_elevation} -> ({to_x}, {to_y}) E{to_elevation}"
                            )
                            return False
            except (IndexError, TypeError, AttributeError) as e:
                # If we can't get elevation data, fall through to symbol-based checks
                logger.debug(f"Could not check elevation: {e}")
        # ========================================================================

        # Cycling road (&) - special handling for cross-elevation movement
        # When moving between different elevations (ground <-> bridge):
        #   - Allow horizontal movement (crossing under/over the bridge)
        #   - Allow vertical movement ONLY if there's a ground path pattern (. & & & .)
        #     (indicating you're on the ground path underneath, not trying to climb onto the bridge)
        # When on the same elevation (e.g., both on bridge), allow all directions
        if (from_symbol == "&" or dest_symbol == "&") and "raw_tiles" in map_data and game_is_red:
            raw_tiles = map_data["raw_tiles"]
            try:
                from_tile = (
                    raw_tiles[from_y][from_x] if from_y < len(raw_tiles) and from_x < len(raw_tiles[from_y]) else None
                )
                to_tile = raw_tiles[to_y][to_x] if to_y < len(raw_tiles) and to_x < len(raw_tiles[to_y]) else None

                if from_tile and to_tile and len(from_tile) >= 4 and len(to_tile) >= 4:
                    from_elev = from_tile[3] if len(from_tile) > 3 else 0
                    to_elev = to_tile[3] if len(to_tile) > 3 else 0

                    # If moving between different elevations involving cycling road
                    if from_elev != to_elev:
                        # Horizontal movement always allowed (crossing under/over the bridge)
                        if dx != 0:  # Moving horizontally
                            pass  # Allow
                        # Vertical movement only allowed if there's a ground path pattern
                        elif dx == 0:  # Trying to move vertically
                            # Check which tile is the & tile
                            if from_symbol == "&":
                                check_x, check_y = from_x, from_y
                                player_elev = to_elev  # Moving from bridge, player at destination elevation
                            else:  # dest_symbol == '&'
                                check_x, check_y = to_x, to_y
                                player_elev = from_elev  # Moving to bridge, player at source elevation

                            # Check for . & & & . pattern (ground path underneath bridge)
                            # Search left through consecutive bridge tiles to find ground
                            left_walkable = False
                            search_x = check_x - 1
                            while search_x >= 0 and search_x < len(grid[check_y]):
                                search_char = grid[check_y][search_x]
                                if search_char == "&":
                                    search_x -= 1
                                elif search_char in [".", "~"]:
                                    # Found walkable ground - check elevation
                                    if search_x < len(raw_tiles[check_y]):
                                        search_tile = raw_tiles[check_y][search_x]
                                        if search_tile and len(search_tile) >= 4:
                                            search_elev = search_tile[3] if len(search_tile) > 3 else 0
                                            if search_elev == player_elev:
                                                left_walkable = True
                                    break
                                else:
                                    break

                            # Search right through consecutive bridge tiles to find ground
                            right_walkable = False
                            search_x = check_x + 1
                            while search_x < len(grid[check_y]):
                                search_char = grid[check_y][search_x]
                                if search_char == "&":
                                    search_x += 1
                                elif search_char in [".", "~"]:
                                    # Found walkable ground - check elevation
                                    if search_x < len(raw_tiles[check_y]):
                                        search_tile = raw_tiles[check_y][search_x]
                                        if search_tile and len(search_tile) >= 4:
                                            search_elev = search_tile[3] if len(search_tile) > 3 else 0
                                            if search_elev == player_elev:
                                                right_walkable = True
                                    break
                                else:
                                    break

                            has_ground_path = left_walkable and right_walkable

                            # Block vertical movement if no ground path underneath
                            if not has_ground_path:
                                return False
            except (IndexError, TypeError, AttributeError):
                pass  # Fall through if we can't get elevation data

        # Check if source is a directional tile (slide/ledge) - can only move in allowed direction
        # Slides and ledges are one-way: you can only exit in the direction they point
        if from_symbol == "→":  # SLIDE/JUMP_EAST
            # Can only move east (dx > 0) from this tile
            if dx <= 0:
                return False
        elif from_symbol == "←":  # SLIDE/JUMP_WEST
            # Can only move west (dx < 0) from this tile
            if dx >= 0:
                return False
        elif from_symbol == "↑":  # SLIDE/JUMP_NORTH
            # Can only move north (dy < 0) from this tile
            if dy >= 0:
                return False
        elif from_symbol == "↓":  # SLIDE/JUMP_SOUTH (mudslides in caves)
            # Can only move south (dy > 0) from this tile
            if dy <= 0:
                return False
        elif from_symbol in ["↗", "↖", "↘", "↙"]:  # Diagonal slides/ledges
            if from_symbol == "↗":  # NORTHEAST
                if dx <= 0 or dy >= 0:
                    return False
            elif from_symbol == "↖":  # NORTHWEST
                if dx >= 0 or dy >= 0:
                    return False
            elif from_symbol == "↘":  # SOUTHEAST
                if dx <= 0 or dy <= 0:
                    return False
            elif from_symbol == "↙":  # SOUTHWEST
                if dx >= 0 or dy <= 0:
                    return False

        # Check if destination is a ledge - if so, validate direction
        # Ledge direction rules:
        # - Can ONLY move TO a ledge from the correct direction
        # - Moving away from a ledge is always OK
        if dest_symbol == "→":  # JUMP_EAST ledge
            # Can only jump east (dx > 0)
            if dx <= 0:  # Trying to move west, north, south, or diagonal
                return False
        elif dest_symbol == "←":  # JUMP_WEST ledge
            # Can only jump west (dx < 0)
            if dx >= 0:  # Trying to move east, north, south, or diagonal
                return False
        elif dest_symbol == "↑":  # JUMP_NORTH ledge
            # Can only jump north (dy < 0)
            if dy >= 0:  # Trying to move south, east, west, or diagonal
                return False
        elif dest_symbol == "↓":  # JUMP_SOUTH ledge
            # Can only jump south (dy > 0)
            if dy <= 0:  # Trying to move north, east, west, or diagonal
                return False
        elif dest_symbol in ["↗", "↖", "↘", "↙"]:  # Diagonal ledges
            # Diagonal ledges require both components to match
            if dest_symbol == "↗":  # JUMP_NORTHEAST
                if dx <= 0 or dy >= 0:  # Must move east AND north
                    return False
            elif dest_symbol == "↖":  # JUMP_NORTHWEST
                if dx >= 0 or dy >= 0:  # Must move west AND north
                    return False
            elif dest_symbol == "↘":  # JUMP_SOUTHEAST
                if dx <= 0 or dy <= 0:  # Must move east AND south
                    return False
            elif dest_symbol == "↙":  # JUMP_SOUTHWEST
                if dx >= 0 or dy <= 0:  # Must move west AND south
                    return False

        return True

    def _astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        blocked: Set[Tuple[int, int]],
        map_data: Dict,
        max_distance: int,
    ):
        """
        A* pathfinding algorithm implementation.

        Returns:
            - List of (x, y) positions from start to goal if path found
            - Tuple of (None, best_node) if no complete path found (for partial paths)
        """
        raw_tiles = map_data.get("raw_tiles") or []
        surf = bool(map_data.get("surf"))

        def raw_tile(pos):
            x, y = pos
            return raw_tiles[y][x] if 0 <= y < len(raw_tiles) and 0 <= x < len(raw_tiles[y]) else None

        def elevation(pos):
            tile = raw_tile(pos)
            return tile[3] if tile and len(tile) > 3 else None

        def carried_after(carried, pos):
            """The walker's elevation on pos: a bridge keeps the one it came with."""
            here = elevation(pos)
            return carried if here == BRIDGE_ELEVATION else here

        # People standing on bridges, with their elevations (_get_blocked_positions).
        bridge_people = map_data.get("bridge_people") or {}

        def person_stops(carried, pos):
            return any(elevations_compatible(carried, theirs) for theirs in bridge_people.get(pos, ()))

        # Rotating gates: which tiles join which depends on the gates' orientations and
        # each push turns a gate, so the same tile with the gates turned otherwise is
        # another search state (rotating_gates.py). Their check runs only for a step onto a
        # tile with collision 0, as pret CheckForObjectEventCollision does it only when
        # GetCollisionAtCoords found nothing (a ledge jump or Surf step never pushes one).
        puzzle = map_data.get("rotating_gates")
        start_gates = tuple(map_data.get("gate_orientations") or ()) if puzzle is not None else None
        if puzzle is not None and len(start_gates) != len(puzzle.gates):
            puzzle, start_gates = None, None

        def gates_after(gates, pos, to):
            if puzzle is None:
                return gates
            tile = raw_tile(to)
            if tile is not None and len(tile) > 2 and tile[2] != 0:
                return gates
            dx, dy = to[0] - pos[0], to[1] - pos[1]
            button = "RIGHT" if dx > 0 else "LEFT" if dx < 0 else "DOWN" if dy > 0 else "UP"
            return puzzle.step(gates, button, to)

        # Strength boulders the party can push: their tiles are part of the search state
        # (the same tile with a boulder moved is another state), and a step into one is a
        # push when the tile behind it takes the boulder (_push_target).
        start_boulders = None
        if map_data.get("strength") and map_data.get("boulders"):
            start_boulders = tuple(sorted({tuple(b) for b in map_data["boulders"]} - {start}))

        def key(pos, carried, gates=None, boulders=None):
            # On a bridge, the same tile walked on from different floors leads on
            # differently, so it is a different search state.
            return (pos, carried if elevation(pos) == BRIDGE_ELEVATION else None, gates, boulders)

        # The player's own elevation where the walk starts from their tile (pret
        # gObjectEvents[player].currentElevation, game_tools.load_pathfinding_state): on a
        # bridge it is the floor they came from, which the tile can't tell.
        live = map_data.get("start_elevation")
        start_carried = carried_after(live if start == map_data.get("start_position") else None, start)
        start_node = Node(
            start[0], start[1], 0, self._heuristic(start, goal), carried=start_carried, gates=start_gates,
            boulders=start_boulders,
        )
        goal_node = Node(goal[0], goal[1])
        clearable = {tuple(t) for t in map_data.get("clearable") or []}
        self.last_pushes = []
        self.last_path_buttons = []

        open_list = []
        heapq.heappush(open_list, start_node)
        closed_set = set()
        node_map = {key(start, start_carried, start_gates, start_boulders): start_node}

        # Track the best node (closest to goal) in case we can't reach it
        best_node = start_node
        opened = 0

        while open_list:
            current = heapq.heappop(open_list)

            # Check if we reached the goal
            if (current.x, current.y) == goal:
                # How the walk leaves the gates, for the caller to check after walking.
                self.last_gate_orientations = current.gates
                if start_boulders is not None:
                    self.last_path_buttons, self.last_pushes = self._node_buttons(current)
                return self._reconstruct_path(current)

            if start_boulders is not None:
                opened += 1
                if opened > BOULDER_SEARCH_MAX_NODES:
                    logger.warning(f"Pathfinding: boulder search from {start} to {goal} gave up after {opened - 1} nodes")
                    break

            # Check if we've searched too far
            # max_distance is a step budget, but g_cost counts a grass step as grass_cost.
            # Seen live (2026-11-03): with grass at 8x, go_to_exit("west") from Route 116
            # (44,9) reported "no walkable path" for a ~45-step route that crossed grass.
            if current.g_cost > max_distance * max(1.0, self.grass_cost):
                continue

            closed_set.add(key((current.x, current.y), current.carried, current.gates, current.boulders))

            # Update best node if this one is closer to goal
            if current.h_cost < best_node.h_cost:
                best_node = current

            # Check all neighbors
            neighbors = self._get_neighbors(current, blocked)
            # Sort neighbors by direction preference for deterministic tie-breaking:
            # Prefer moving toward goal (lower h_cost) when f_cost is equal
            # This helps find optimal paths faster
            neighbors_with_cost = []
            here = (current.x, current.y)
            for neighbor_pos in neighbors:
                if current.boulders is not None and neighbor_pos in current.boulders:
                    # A step into a boulder pushes it (pret TryPushBoulder) when the walk to
                    # it would be a legal step; the player stays where they are.
                    delta = (neighbor_pos[0] - current.x, neighbor_pos[1] - current.y)
                    if current.pushes >= BOULDER_MAX_PUSHES or not self._can_move_to(
                        here, neighbor_pos, map_data, current.carried
                    ):
                        continue
                    to = self._push_target(neighbor_pos, delta, current.boulders, blocked, map_data)
                    if to is None:
                        continue
                    moved = tuple(sorted((set(current.boulders) - {neighbor_pos}) | {to}))
                    if key(here, current.carried, current.gates, moved) in closed_set:
                        continue
                    button = self._path_to_buttons([here, neighbor_pos])[0]
                    neighbors_with_cost.append(
                        (here, current.g_cost + BOULDER_PUSH_COST, current.h_cost, current.carried, current.gates,
                         moved, (button, neighbor_pos, to))
                    )
                    continue
                carried = carried_after(current.carried, neighbor_pos)
                # Check if this move is valid (handles ledge directions)
                if not self._can_move_to((current.x, current.y), neighbor_pos, map_data, current.carried):
                    continue
                if bridge_people and person_stops(current.carried, neighbor_pos):
                    continue
                gates = gates_after(current.gates, (current.x, current.y), neighbor_pos)
                if puzzle is not None and gates is None:
                    continue  # a gate's arm is in the way and can't turn
                if key(neighbor_pos, carried, gates, current.boulders) in closed_set:
                    continue

                # Calculate movement cost - add penalty for grass tiles to avoid wild encounters
                move_cost = 1.0  # Base cost

                # Check if moving onto grass tile (represented as '~' in porymap grid)
                if "grid" in map_data and map_data.get("type") == "porymap":
                    grid = map_data["grid"]
                    nx, ny = neighbor_pos
                    if 0 <= ny < len(grid) and isinstance(grid[ny], (list, str)):
                        row = grid[ny]
                        if 0 <= nx < len(row):
                            cell = row[nx]
                            if cell == "~":  # Grass tile
                                move_cost = self.grass_cost
                if neighbor_pos in clearable:
                    move_cost = CLEARABLE_TILE_COST
                elif surf and is_surf_water(raw_tile(neighbor_pos)) and not is_surf_water(raw_tile((current.x, current.y))):
                    move_cost = SURF_START_COST

                g_cost = current.g_cost + move_cost
                h_cost = self._heuristic(neighbor_pos, goal)
                neighbors_with_cost.append((neighbor_pos, g_cost, h_cost, carried, gates, current.boulders, None))

            # Sort by h_cost (distance to goal) so we explore promising directions first
            # This helps find optimal paths when multiple neighbors have same f_cost
            neighbors_with_cost.sort(key=lambda x: x[2])  # Sort by h_cost

            for neighbor_pos, g_cost, h_cost, carried, gates, boulders, push in neighbors_with_cost:
                state = key(neighbor_pos, carried, gates, boulders)
                pushes = current.pushes + (1 if push else 0)
                if state not in node_map:
                    neighbor = Node(
                        neighbor_pos[0], neighbor_pos[1], g_cost, h_cost, carried=carried, gates=gates,
                        boulders=boulders, push=push, pushes=pushes,
                    )
                    neighbor.parent = current
                    node_map[state] = neighbor
                    heapq.heappush(open_list, neighbor)
                else:
                    neighbor = node_map[state]
                    if g_cost < neighbor.g_cost:
                        neighbor.g_cost = g_cost
                        neighbor.f_cost = g_cost + neighbor.h_cost
                        neighbor.parent = current
                        neighbor.push, neighbor.pushes = push, pushes

        # No path found - return the best node we explored (for partial paths)
        return (None, best_node)

    def _get_neighbors(self, node: Node, blocked: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Get valid neighbor positions for a node using connectivity map."""
        neighbors = []
        current_pos = (node.x, node.y)

        # Use pre-computed connectivity map if available
        if current_pos in self.tile_connectivity:
            connectivity = self.tile_connectivity[current_pos]

            # Check each direction based on connectivity
            direction_map = {
                "up": (node.x, node.y - 1),
                "down": (node.x, node.y + 1),
                "left": (node.x - 1, node.y),
                "right": (node.x + 1, node.y),
            }

            for direction, neighbor_pos in direction_map.items():
                # Only add if connectivity allows AND not blocked
                if connectivity.get(direction, False) and neighbor_pos not in blocked:
                    neighbors.append(neighbor_pos)
        else:
            # Fallback to old behavior if no connectivity map
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

            if self.allow_diagonal:
                directions.extend([(-1, -1), (1, -1), (-1, 1), (1, 1)])

            for dx, dy in directions:
                new_x, new_y = node.x + dx, node.y + dy
                if (new_x, new_y) not in blocked:
                    neighbors.append((new_x, new_y))

        return neighbors

    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate heuristic distance (Manhattan distance for grid movement)."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """Reconstruct the path from start to goal."""
        path = []
        current = node
        while current:
            path.append((current.x, current.y))
            current = current.parent
        path.reverse()
        return path

    def _node_buttons(self, node: Node) -> Tuple[List[str], List[Dict]]:
        """(buttons, pushes) of the walk ending at node: a push's press comes from the
        node (the player's tile does not change), every other press from the step."""
        chain = []
        while node is not None:
            chain.append(node)
            node = node.parent
        chain.reverse()
        buttons: List[str] = []
        pushes: List[Dict] = []
        for before, after in zip(chain, chain[1:]):
            if after.push is not None:
                button, boulder, to = after.push
                pushes.append({"index": len(buttons), "button": button, "boulder": boulder, "to": to})
                buttons.append(button)
            else:
                buttons.extend(self._path_to_buttons([(before.x, before.y), (after.x, after.y)]))
        return buttons, pushes

    def _push_target(
        self,
        boulder: Tuple[int, int],
        delta: Tuple[int, int],
        boulders,
        blocked: Set[Tuple[int, int]],
        map_data: Dict,
    ) -> Optional[Tuple[int, int]]:
        """Where a push in direction delta moves the boulder on `boulder`, or None when the
        game would not move it (or this planner won't).

        pret src/field_player_avatar.c TryPushBoulder: the tile behind must give
        GetCollisionAtCoords(boulder, ...) == COLLISION_NONE (src/event_object_movement.c:
        no collision bits, on the map, not MB_IMPASSABLE_* that way, no elevation mismatch
        with the boulder, no object there) and must not be MetatileBehavior_IsNonAnimDoor.
        Beyond that, warps, ledges and water are refused (BOULDER_FLOOR): a missing push
        is better than one that buries an exit."""
        tx, ty = boulder[0] + delta[0], boulder[1] + delta[1]
        target = (tx, ty)
        grid = map_data.get("grid") or []
        if target in boulders or target in blocked or not (0 <= ty < len(grid) and 0 <= tx < len(grid[ty])):
            return None
        if grid[ty][tx] not in BOULDER_FLOOR:
            return None
        if target in {tuple(p) for p in map_data.get("raw_warps") or []}:
            return None
        if target in self._object_tiles(map_data):
            return None  # pret DoesObjectCollideWithObjectAt: a person stands there
        raw_tiles = map_data.get("raw_tiles") or []

        def tile(pos):
            x, y = pos
            return raw_tiles[y][x] if 0 <= y < len(raw_tiles) and 0 <= x < len(raw_tiles[y]) else None

        there = tile(target)
        if there is not None:
            if len(there) > 2 and there[2]:
                return None  # MapGridGetCollisionAt
            name = tile_behavior_name(there)
            if any(word in name for word in BOULDER_REFUSED_BEHAVIORS):
                return None
            # pret IsElevationMismatchAt: elevations 0 and 15 match anything.
            here = tile(boulder)
            mine = here[3] if here is not None and len(here) > 3 else 0
            theirs = there[3] if len(there) > 3 else 0
            if mine not in (0, BRIDGE_ELEVATION) and theirs not in (0, BRIDGE_ELEVATION) and mine != theirs:
                return None
        return target

    def _object_tiles(self, map_data: Dict) -> Set[Tuple[int, int]]:
        """Tiles people stand on: their live tiles and their map-file tiles (the live list
        covers only those loaded near the camera)."""
        tiles = set()
        for obj in map_data.get("runtime_objects") or []:
            x, y = obj.get("current_x", obj.get("x")), obj.get("current_y", obj.get("y"))
            if x is not None and y is not None:
                tiles.add((x, y))
        for obj in map_data.get("objects") or []:
            if obj.get("x") is not None and obj.get("y") is not None:
                tiles.add((obj["x"], obj["y"]))
        return tiles

    def _find_nearest_reachable(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        blocked: Set[Tuple[int, int]],
        map_data: Dict,
        max_distance: int = 10,
    ) -> Optional[Tuple[int, int]]:
        """
        Find the nearest reachable position to the goal.

        Uses BFS to explore positions near the goal and find the closest
        reachable one.

        Args:
            start: Starting position
            goal: Goal position (may be blocked)
            blocked: Set of blocked positions
            map_data: Map data for pathfinding
            max_distance: Maximum distance to search from goal (default 10)

        Returns:
            Nearest reachable position, or None if none found
        """
        if goal not in blocked:
            return goal

        visited = set()
        queue = [(goal, 0)]
        visited.add(goal)
        max_search_distance = max_distance

        while queue:
            pos, dist = queue.pop(0)

            if dist > max_search_distance:
                break

            # Check neighbors in priority order:
            # 1. Cardinal directions (especially directly below/south)
            # 2. Diagonals
            # This ensures we prefer positions directly adjacent to the goal
            # For objects like clocks, directly below is usually the best position

            # Priority order: (dx, dy) pairs
            # First: cardinal directions (south/down has highest priority for objects)
            cardinal_directions = [
                (0, 1),  # South (directly below) - highest priority for objects
                (0, -1),  # North (directly above)
                (-1, 0),  # West (left)
                (1, 0),  # East (right)
            ]

            # Then: diagonal directions
            diagonal_directions = [
                (-1, -1),  # Northwest
                (1, -1),  # Northeast
                (-1, 1),  # Southwest
                (1, 1),  # Southeast
            ]

            # Check cardinal directions first
            for dx, dy in cardinal_directions:
                new_pos = (pos[0] + dx, pos[1] + dy)

                if new_pos not in visited:
                    visited.add(new_pos)

                    if new_pos not in blocked:
                        # Found a reachable position
                        # Check if we can path from start to here
                        test_result = self._astar(start, new_pos, blocked, map_data, 50)
                        # Handle both tuple (None, best_node) and list returns
                        if isinstance(test_result, list):
                            # Found a complete path
                            return new_pos

                    queue.append((new_pos, dist + 1))

            # Then check diagonal directions
            for dx, dy in diagonal_directions:
                new_pos = (pos[0] + dx, pos[1] + dy)

                if new_pos not in visited:
                    visited.add(new_pos)

                    if new_pos not in blocked:
                        # Found a reachable position
                        # Check if we can path from start to here
                        test_result = self._astar(start, new_pos, blocked, map_data, 50)
                        # Handle both tuple (None, best_node) and list returns
                        if isinstance(test_result, list):
                            # Found a complete path
                            return new_pos

                    queue.append((new_pos, dist + 1))

        return None

    def _generate_variance_candidates(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        blocked: Set[Tuple[int, int]],
        map_data: Dict,
        max_distance: int,
        variance_steps: int,
        base_path_buttons: List[str],
        max_variants: int = 8,
    ) -> List[List[str]]:
        """
        Generate alternative button sequences that reach the goal but differ in their initial moves.
        Each path is guaranteed to have no cycles (never revisits a tile).
        """
        if variance_steps <= 0 or not base_path_buttons or len(base_path_buttons) < variance_steps:
            return []

        # Track seen prefixes to ensure we generate diverse candidates
        base_prefix_key = tuple(base_path_buttons[:variance_steps])
        seen_prefixes = {base_prefix_key}
        candidates: List[List[str]] = []

        prefix_paths = self._enumerate_prefix_paths(
            start=start,
            blocked=blocked,
            map_data=map_data,
            steps_required=variance_steps,
            max_prefixes=max_variants * 4,  # oversample to increase odds of unique prefixes
        )

        for prefix_positions in prefix_paths:
            prefix_buttons = self._path_to_buttons(prefix_positions)
            if len(prefix_buttons) < variance_steps:
                continue

            prefix_key = tuple(prefix_buttons[:variance_steps])
            if prefix_key in seen_prefixes:
                continue

            # Block all tiles in the prefix path (except the last one, which is the start of remainder)
            # This prevents the remainder path from looping back through already-visited tiles
            blocked_with_prefix = blocked.copy()
            for pos in prefix_positions[:-1]:  # Exclude last position (start of remainder)
                blocked_with_prefix.add(pos)

            remainder = self._astar(prefix_positions[-1], goal, blocked_with_prefix, map_data, max_distance)
            if not isinstance(remainder, list) or len(remainder) < 1:
                continue

            full_positions = prefix_positions + remainder[1:]

            # Validate that the full path has no cycles (no position appears twice)
            if len(set(full_positions)) != len(full_positions):
                logger.debug(
                    f"Rejecting path with cycle: {len(full_positions)} steps but only {len(set(full_positions))} unique positions"
                )
                continue

            buttons = self._path_to_buttons(full_positions)
            if len(buttons) < variance_steps:
                continue

            initial_key = tuple(buttons[:variance_steps])
            if initial_key in seen_prefixes:
                continue

            seen_prefixes.add(initial_key)
            candidates.append(buttons)

            if len(candidates) >= max_variants:
                break

        return candidates

    def _enumerate_prefix_paths(
        self,
        start: Tuple[int, int],
        blocked: Set[Tuple[int, int]],
        map_data: Dict,
        steps_required: int,
        max_prefixes: int = 256,
    ) -> List[List[Tuple[int, int]]]:
        """
        Enumerate walkable position sequences originating from start with a fixed number of steps.
        """
        if steps_required <= 0:
            return []

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        results: List[List[Tuple[int, int]]] = []
        seen_paths: Set[Tuple[Tuple[int, int], ...]] = set()
        raw_tiles = map_data.get("raw_tiles") or []

        def tile_elevation(pos):
            x, y = pos
            tile = raw_tiles[y][x] if 0 <= y < len(raw_tiles) and 0 <= x < len(raw_tiles[y]) else None
            return tile[3] if tile and len(tile) > 3 else None

        # The walker's elevation goes along each prefix, as in _astar (elevation_step).
        live = map_data.get("start_elevation") if start == map_data.get("start_position") else None
        start_walker = live if tile_elevation(start) == BRIDGE_ELEVATION else tile_elevation(start)
        stack = [(start, [start], start_walker)]

        while stack and len(results) < max_prefixes:
            current_pos, positions, walker = stack.pop()
            steps_taken = len(positions) - 1

            if steps_taken == steps_required:
                path_tuple = tuple(positions)
                if path_tuple not in seen_paths:
                    seen_paths.add(path_tuple)
                    results.append(positions)
                continue

            for dx, dy in directions:
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)

                if next_pos in blocked:
                    continue

                if next_pos in positions:
                    continue

                if not self._can_move_to(current_pos, next_pos, map_data, walker):
                    continue

                there = tile_elevation(next_pos)
                stack.append((next_pos, positions + [next_pos], walker if there == BRIDGE_ELEVATION else there))

        return results

    def _path_to_buttons(self, path: List[Tuple[int, int]]) -> List[str]:
        """Convert a path of positions to button commands."""
        if len(path) < 2:
            return []

        buttons = []
        for i in range(1, len(path)):
            prev = path[i - 1]
            curr = path[i]

            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]

            if dx > 0:
                buttons.append("RIGHT")
            elif dx < 0:
                buttons.append("LEFT")
            elif dy > 0:
                buttons.append("DOWN")
            elif dy < 0:
                buttons.append("UP")

        return buttons

    def _simple_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[str]:
        """Simple straight-line path when no map data is available."""
        buttons = []

        dx = goal[0] - start[0]
        dy = goal[1] - start[1]

        # Move horizontally first
        if dx > 0:
            buttons.extend(["RIGHT"] * min(abs(dx), 10))
        elif dx < 0:
            buttons.extend(["LEFT"] * min(abs(dx), 10))

        # Then vertically
        if dy > 0:
            buttons.extend(["DOWN"] * min(abs(dy), 10))
        elif dy < 0:
            buttons.extend(["UP"] * min(abs(dy), 10))

        return buttons if buttons else None


# Convenience function
def find_path(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    game_state: Dict,
    max_distance: int = 150,
    variance: Optional[str] = None,
    consider_npcs: bool = False,
    allow_partial: bool = True,
) -> Optional[List[str]]:
    """
    Find a path from start to goal position.

    Args:
        start: Starting position (x, y)
        goal: Goal position (x, y)
        game_state: Current game state with map data
        max_distance: Maximum distance to search (default 150)
        variance: Optional variance level ('low', 'medium', 'high')
        consider_npcs: Whether to consider NPC positions as blocked (default False)
        allow_partial: Whether to allow partial paths to closest reachable point (default True)

    Returns:
        List of button commands to reach the goal, or None if no path found
    """
    pathfinder = Pathfinder()
    return pathfinder.find_path(
        start,
        goal,
        game_state,
        max_distance=max_distance,
        variance=variance,
        consider_npcs=consider_npcs,
        allow_partial=allow_partial,
    )