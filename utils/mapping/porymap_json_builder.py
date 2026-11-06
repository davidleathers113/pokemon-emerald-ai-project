#!/usr/bin/env python3
"""
Porymap JSON Map Builder

Builds a complete JSON map structure from porymap data (map.json + map.bin)
that can be passed to the LLM for navigation.

The JSON structure includes:
- Full grid with walkability (from map.bin)
- Warp points (from map.json)
- Object events/NPCs (from map.json)
- Connections to adjacent maps (from map.json)
- Dimensions and metadata
"""

import sys
import json
import copy
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.mapping.map_formatter import format_tile_to_symbol
from utils.mapping.pokeemerald_parser import PokeemeraldMapLoader, PokeemeraldLayoutParser
from pokemon_env.enums import MetatileBehavior


def tile_to_symbol(tile_tuple: Tuple[int, Any, int, int], location_name: str = "") -> str:
    """Proxy to shared formatter so behaviour-based symbols stay in sync."""
    if not tile_tuple:
        return '?'
    return format_tile_to_symbol(tile_tuple, location_name=location_name)


def override_walk_tiles(override_tiles, override_ascii: str, layout_tiles) -> Optional[List[List[tuple]]]:
    """The tiles to plan walks on for a map drawn by an ASCII override
    (ascii_map_loader.MAP_OVERRIDES), or None when the decomp's layout isn't the same size.

    An override draws walls and floor only: every tile it makes is elevation 3 with no
    collision. The game checks the layout's tiles (pret GetCollisionAtCoords:
    MapGridGetCollisionAt and IsElevationMismatchAt), so walks use those: the layout tile
    everywhere, except where the override draws a door or warp or the layout has one,
    which is what the overrides were made to fix (a gym door drawn a tile off). There the
    override's tile is kept with the layout's elevation.

    Seen live: on MOSSDEEP CITY the override's elevation 3 everywhere joined the west shore
    (16,6) (elevation 3) to (16,7) (elevation 5), and go_to walked into the cliff again and
    again; its water was floor ('W' becomes '.' in ascii_to_metatiles) and its cliff edges,
    collision 1 in the layout, were open. FORTREE CITY's treetop walkway (elevation 4) was
    joined to the ground (3) beside it.
    """
    lines = [line.strip() for line in override_ascii.strip().split("\n") if line.strip()]
    if (
        not layout_tiles
        or len(layout_tiles) != len(override_tiles)
        or len(lines) != len(override_tiles)
        or any(len(a) != len(b) or len(a) != len(c) for a, b, c in zip(layout_tiles, override_tiles, lines))
    ):
        return None
    from utils.mapping.pathfinding import WARP_BEHAVIOR_WORDS, tile_behavior_name

    tiles = []
    for game_row, drawn_row, line in zip(layout_tiles, override_tiles, lines):
        row = []
        for game, drawn, symbol in zip(game_row, drawn_row, line):
            game = tuple(game)
            if len(game) < 4:
                row.append(tuple(drawn))
            elif symbol in ("D", "S") or any(word in tile_behavior_name(game) for word in WARP_BEHAVIOR_WORDS):
                row.append((drawn[0], drawn[1], drawn[2], game[3]))
            else:
                row.append(game)
        tiles.append(row)
    return tiles


def underwater_floor(metatiles) -> Optional[List[List[tuple]]]:
    """An underwater map's tiles with their water behaviors read as plain floor (MB_NORMAL).

    Underwater the player is no surfer: pret src/field_player_avatar.c sets
    PLAYER_AVATAR_FLAG_UNDERWATER, not _SURFING (PlayerAvatarTransition_Underwater), and
    MovePlayerNotOnBike walks them (PlayerWalkNormal) where GetCollisionAtCoords allows,
    by collision bit and elevation only. UNDERWATER SEAFLOOR CAVERN's floor is
    MB_DEEP_WATER at elevation 3, and the planner's Surf rules (pathfinding.water_step)
    took it for water to Surf onto from its warp, so no walk crossed it. Which tile B
    surfaces from (MB_NO_SURFACING) is read from the layout itself
    (porymap_state.layout_tiles), never from these."""
    from utils.mapping.pathfinding import SURF_WATER_BEHAVIORS, tile_behavior_name

    if not metatiles:
        return metatiles
    return [
        [
            (tile[0], MetatileBehavior.NORMAL, *tile[2:])
            if len(tile) > 1 and tile_behavior_name(tile) in SURF_WATER_BEHAVIORS else tile
            for tile in row
        ]
        for row in metatiles
    ]


def build_json_map(map_name: str, pokeemerald_root: Path,
                   include_grid: bool = True,
                   include_ascii: bool = True,
                   badge_count: int = 0,
                   layout: str = "") -> Optional[Dict[str, Any]]:
    """
    Build a complete JSON map structure from porymap data.

    Args:
        map_name: Map name (e.g., "OldaleTown", "LittlerootTown")
        pokeemerald_root: Root directory of pokeemerald project
        include_grid: Include full grid data as JSON (True) or just metadata (False)
        include_ascii: Include ASCII representation for visualization
        badge_count: Number of badges player has (for game-state-aware map selection)
        layout: a LAYOUT_* id to draw the map from instead of the map file's: the one its
            transition script swaps in (porymap_state.map_layout_now)

    Returns:
        Dictionary with complete map data in JSON-serializable format
        {
            "name": "OldaleTown",
            "id": "MAP_OLDALE_TOWN",
            "dimensions": {"width": 20, "height": 20},
            "grid": [[".", "#", ...], ...],  # Optional: full grid
            "ascii": "...",  # Optional: ASCII visualization
            "warps": [...],
            "objects": [...],
            "connections": [...],
            "metadata": {...}
        }
    """
    pokeemerald_root = Path(pokeemerald_root)
    
    # Load map data
    map_loader = PokeemeraldMapLoader(pokeemerald_root)
    map_data = map_loader.load_map(map_name)
    
    if not map_data:
        return None
    
    # Get layout parser
    layout_parser = PokeemeraldLayoutParser(pokeemerald_root)
    
    # Get layout name
    layout_name = layout or map_loader.get_layout_name_from_map(map_name)

    # Parse tile data if layout available
    grid = None
    ascii_map = None
    dimensions = {"width": 0, "height": 0}

    # Check for map overrides (game-state-aware selection + partial/full overrides)
    from utils.mapping.ascii_map_loader import get_effective_map_name, get_override, ascii_to_metatiles
    
    effective_map_name = get_effective_map_name(map_name, badge_count=badge_count)
    override = get_override(effective_map_name)
    
    # Load base metatiles from porymap (or override ASCII if provided)
    metatiles = None
    # The tiles walks are planned on ("raw_tiles"); the override's own unless replaced below.
    walk_tiles = None
    if override and 'ascii' in override:
        # Use override ASCII - convert to metatiles
        metatiles = ascii_to_metatiles(override['ascii'], effective_map_name)
        if metatiles:
            dimensions = {"width": len(metatiles[0]), "height": len(metatiles)}
            if layout_name:
                walk_tiles = override_walk_tiles(
                    metatiles, override['ascii'], layout_parser.get_metatiles_with_behavior(layout_name)
                )
    
    # Fall back to binary map.bin files if no ASCII override
    if metatiles is None and layout_name:
        # Get metatiles with behavior
        metatiles = layout_parser.get_metatiles_with_behavior(layout_name)
        if map_data.get("map_type") == "MAP_TYPE_UNDERWATER":
            metatiles = underwater_floor(metatiles)

        if metatiles:
            height = len(metatiles)
            width = len(metatiles[0]) if height > 0 else 0
            dimensions = {"width": width, "height": height}

    # Build grid and ASCII for any metatiles (corrected or binary)
    ascii_from_override = False
    if metatiles:
        height = len(metatiles)
        width = len(metatiles[0]) if height > 0 else 0

        # Build grid
        if include_grid:
            grid = []
            for row in metatiles:
                grid_row = []
                for tile in row:
                    symbol = tile_to_symbol(tile, map_name)
                    grid_row.append(symbol)
                grid.append(grid_row)

        # Store raw tiles with elevation for pathfinding
        raw_tiles = metatiles  # Already in format (tile_id, behavior, collision, elevation)

        # Build ASCII representation
        if include_ascii:
            ascii_lines = []
            for row in metatiles:
                line = "".join(tile_to_symbol(tile, map_name) for tile in row)
                ascii_lines.append(line)

            # Overlay special object markers (TV, Clock, etc.)
            # This must be done BEFORE joining ascii_lines into ascii_map
            ascii_grid = [list(line) for line in ascii_lines]

            # Define special object graphics IDs and their symbols
            special_object_markers = {
                # TVs - common graphics_id for TV objects
                "OBJ_EVENT_GFX_ITEM_BALL": "I",  # Item balls
                "OBJ_EVENT_GFX_CUTTABLE_TREE": "T",  # Tree
                # Add more as needed
            }

            # Mark background events (clocks, PCs, etc.) on the ASCII map
            for bg_event in map_data.get("bg_events", []):
                bg_x = bg_event.get("x", 0)
                bg_y = bg_event.get("y", 0)
                script = bg_event.get("script", "")

                # Check if position is valid
                if 0 <= bg_y < len(ascii_grid) and 0 <= bg_x < len(ascii_grid[bg_y]):
                    # Clock events
                    if "WallClock" in script or "Clock" in script:
                        ascii_grid[bg_y][bg_x] = 'K'  # Clock marker
                    # PC events (script often ends with _PC or contains EventScript_PC)
                    elif "PC" in script:
                        ascii_grid[bg_y][bg_x] = 'P'  # PC marker
                    # TV/GameCube/Notebook events
                    elif "TV" in script or "GameCube" in script or "Notebook" in script:
                        ascii_grid[bg_y][bg_x] = 'V'  # TV/notebook marker

            # Convert back to strings
            ascii_lines = [''.join(row) for row in ascii_grid]
            ascii_map = "\n".join(ascii_lines)

            # When an override provides ASCII, use it verbatim so P/V/K/S/I/D match exactly
            # (rebuilding from metatiles can change symbols, and binary fallback can differ)
            if override and "ascii" in override:
                override_ascii = "\n".join(
                    line.strip() for line in override["ascii"].strip().split("\n") if line.strip()
                )
                if override_ascii:
                    ascii_map = override_ascii
                    ascii_from_override = True
    
    # Extract warps, each on its own tile (pret warp_events): the game warps from the
    # tile the event is on (src/field_control_avatar.c GetWarpEventAtMapPosition), so
    # the listed tile is that one. Which tile to step in from and the press that fires it
    # is the tile's behavior's business (porymap_state.warp_entry), not a move of the
    # warp. Seen live: the builder used to move a warp one tile toward any wall row
    # beside it, and off the map's last row into added rows: AQUA HIDEOUT B2F's pad
    # (5,8) was listed at (6,8), a wall the planner then tried to walk into, and every
    # MOSSDEEP CITY GYM pad was a tile or two off ((7,18) listed at (6,17)). The added
    # rows also left the grid a row or column bigger than the map's tiles.
    warps = []
    warp_positions = set()  # Track warp positions for grid overlay

    for warp in map_data.get("warp_events", []):
        warp_x = warp.get("x", 0)
        warp_y = warp.get("y", 0)
        warps.append({
            "x": warp_x,
            "y": warp_y,
            "elevation": warp.get("elevation", 0),
            "dest_map": warp.get("dest_map", "?"),
            "dest_warp_id": warp.get("dest_warp_id", 0)
        })

        # Track warp position for grid overlay
        warp_positions.add((warp_x, warp_y))

    # CRITICAL: Overlay warp symbols on grid and ASCII after extracting warp positions
    # This ensures warps are always marked as walkable, even if the underlying tile
    # is NORMAL with collision (like Petalburg Gym doors)
    if layout_name and metatiles:
        # Update grid (JSON format)
        if include_grid and grid:
            for warp_x, warp_y in warp_positions:
                if 0 <= warp_y < len(grid) and 0 <= warp_x < len(grid[0]):
                    # Get current symbol
                    current_symbol = grid[warp_y][warp_x]
                    # Only override if it's not already a door/stairs
                    # Use 'D' for exit doors (to Petalburg City), 'S' for internal warps
                    warp_dest = None
                    for warp in warps:
                        if warp['x'] == warp_x and warp['y'] == warp_y:
                            warp_dest = warp.get('dest_map', '')
                            break

                    if current_symbol not in ['D', 'S']:
                        # External doors (exits) use 'D', internal warps use 'S'
                        if warp_dest and 'PETALBURG_CITY_GYM' not in warp_dest:
                            grid[warp_y][warp_x] = 'D'
                        else:
                            grid[warp_y][warp_x] = 'S'

        # Update ASCII representation
        if include_ascii and ascii_map:
            ascii_grid = [list(line) for line in ascii_map.split('\n')]
            for warp_x, warp_y in warp_positions:
                if 0 <= warp_y < len(ascii_grid) and 0 <= warp_x < len(ascii_grid[0]):
                    current_symbol = ascii_grid[warp_y][warp_x]

                    # Get warp destination
                    warp_dest = None
                    for warp in warps:
                        if warp['x'] == warp_x and warp['y'] == warp_y:
                            warp_dest = warp.get('dest_map', '')
                            break

                    if current_symbol not in ['D', 'S']:
                        # External doors use 'D', internal warps use 'S'
                        if warp_dest and 'PETALBURG_CITY_GYM' not in warp_dest:
                            ascii_grid[warp_y][warp_x] = 'D'
                        else:
                            ascii_grid[warp_y][warp_x] = 'S'

            # Rebuild ASCII map from grid
            ascii_map = '\n'.join(''.join(row) for row in ascii_grid)
    
    # Extract objects - use override if provided, else porymap data
    if override and 'objects' in override:
        objects = override['objects']
    else:
        objects = []
        for number, obj in enumerate(map_data.get("object_events", []), start=1):
            objects.append({
                "local_id": obj.get("local_id", ""),
                # The game's localId: the object's place in object_events, from 1 (pret
                # tools/mapjson). Live positions are matched by it (field_control.object_places).
                "event_id": number,
                "x": obj.get("x", 0),
                "y": obj.get("y", 0),
                "elevation": obj.get("elevation", 0),
                "graphics_id": obj.get("graphics_id", "?"),
                "movement_type": obj.get("movement_type", "?"),
                "movement_range_x": obj.get("movement_range_x", 0),
                "movement_range_y": obj.get("movement_range_y", 0),
                "trainer_type": obj.get("trainer_type", "?"),
                "trainer_sight_or_berry_tree_id": obj.get("trainer_sight_or_berry_tree_id", "?"),
                "flag": obj.get("flag", "0"),
            })
    
    # Extract connections - use override if provided, else porymap data
    if override and 'connections' in override:
        connections = override['connections']
    else:
        connections = []
        connections_data = map_data.get("connections")
        if connections_data is not None:
            for conn in connections_data:
                connections.append({
                    "direction": conn.get("direction", "?"),
                    "offset": conn.get("offset", 0),
                    "map": conn.get("map", "?")
                })
    
    # Extract bg_events (PC, clock, TV, notebook, etc.) - use override if provided, else porymap data
    if override and 'bg_events' in override:
        bg_events = override['bg_events']
    else:
        bg_events = []
        for bg in map_data.get("bg_events", []):
            bg_events.append({
                "type": bg.get("type", "?"),
                "x": bg.get("x", 0),
                "y": bg.get("y", 0),
                "elevation": bg.get("elevation", 0),
                "player_facing_dir": bg.get("player_facing_dir", "?"),
                "script": bg.get("script", "?")
            })
    
    # Use override warps if provided (already extracted above, but override takes precedence)
    if override and 'warps' in override:
        warps = override['warps']
        # Rebuild warp_positions for grid overlay
        warp_positions = {(w['x'], w['y']) for w in warps}
    
    # Build complete JSON structure
    json_map = {
        "name": effective_map_name if override else map_data.get("name", map_name),
        "id": map_data.get("id", f"MAP_{map_name.upper()}"),
        "dimensions": dimensions,
        "warps": warps,
        "objects": objects,
        "bg_events": bg_events,
        "connections": connections,
    }
    
    # Add grid and ASCII if requested
    if include_grid and grid:
        json_map["grid"] = grid
    
    if include_ascii and ascii_map:
        json_map["ascii"] = ascii_map
        if ascii_from_override:
            json_map["ascii_from_override"] = True
    
    # Include raw tiles with elevation for pathfinding elevation checks
    if metatiles:
        json_map["raw_tiles"] = walk_tiles or metatiles
    
    return json_map


@lru_cache(maxsize=3)
def _build_json_map_for_llm_cached(
    map_name: str,
    pokeemerald_root_str: str,
    badge_count: int,
    layout: str = "",
) -> Optional[Dict[str, Any]]:
    """Cache static porymap builds to avoid repeated filesystem parsing."""
    return build_json_map(
        map_name=map_name,
        pokeemerald_root=Path(pokeemerald_root_str),
        include_grid=True,
        include_ascii=True,
        badge_count=badge_count,
        layout=layout,
    )


def build_json_map_for_llm(map_name: str, pokeemerald_root: Path, badge_count: int = 0) -> Optional[Dict[str, Any]]:
    """
    Build a JSON map optimized for LLM consumption.
    
    This version includes:
    - Full grid as nested arrays (for coordinate-based queries)
    - ASCII visualization (for human readability)
    - All important navigation features (warps, objects, connections)
    - Game-state-aware map selection (e.g., Petalburg Gym lobby vs full gym)
    
    Args:
        map_name: Map name (e.g., "OldaleTown")
        pokeemerald_root: Root directory of pokeemerald project
        badge_count: Number of badges player has (for game-state-aware map selection)
        
    Returns:
        JSON-serializable dictionary optimized for LLM
    """
    from utils.mapping.porymap_state import map_layout_now

    # The layout the map's transition script swaps in now (ROUTE 131 is always drawn from
    # LAYOUT_ROUTE131_SKY_PILLAR), so a map read from the decomp matches the game's.
    cached_map = _build_json_map_for_llm_cached(
        map_name=map_name,
        pokeemerald_root_str=str(Path(pokeemerald_root).resolve()),
        badge_count=badge_count,
        layout=map_layout_now(map_name),
    )
    if cached_map is None:
        return None
    # Callers mutate map data (overlay/elevation filtering), so return a deep copy.
    return copy.deepcopy(cached_map)


def save_json_map(map_name: str, pokeemerald_root: Path, output_path: Path):
    """
    Build and save a JSON map to file.
    
    Args:
        map_name: Map name to build
        pokeemerald_root: Root directory of pokeemerald project
        output_path: Where to save the JSON file
    """
    json_map = build_json_map_for_llm(map_name, pokeemerald_root)
    
    if not json_map:
        raise ValueError(f"Could not build map for: {map_name}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(json_map, f, indent=2)
    
    print(f"✅ Saved JSON map to: {output_path}")
    print(f"   Map: {map_name}")
    print(f"   Dimensions: {json_map['dimensions']['width']}x{json_map['dimensions']['height']}")
    print(f"   Warps: {len(json_map['warps'])}")
    print(f"   Objects: {len(json_map['objects'])}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build JSON map from porymap data")
    parser.add_argument("--map", type=str, required=True, help="Map name (e.g., OldaleTown)")
    parser.add_argument("--root", type=str, default="../pokeemerald", help="Pokeemerald root directory")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--print", action="store_true", help="Print JSON to stdout")
    
    args = parser.parse_args()
    
    pokeemerald_root = Path(args.root).resolve()
    
    json_map = build_json_map_for_llm(args.map, pokeemerald_root)
    
    if not json_map:
        print(f"Error: Could not build map for {args.map}")
        exit(1)
    
    if args.print:
        print(json.dumps(json_map, indent=2))
    elif args.output:
        save_json_map(args.map, pokeemerald_root, Path(args.output))
    else:
        # Default: save to maps_json/{map_name}.json
        output_dir = Path("maps_json")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{args.map}.json"
        save_json_map(args.map, pokeemerald_root, output_path)
