"""
Server-side game tool implementations (no HTTP calls).

These helpers operate directly on the EmeraldEmulator instance and are called
by the /mcp/* endpoint handlers in server/app.py.  The MCP server
(server/cli/pokemon_mcp_server.py) does NOT import this module; it routes
tool calls to app.py over HTTP instead.
"""

import logging
import os
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.json_utils import serialize_for_json
from utils.mapping.pathfinding import Pathfinder
from utils.stores.memory import get_memory_store
from utils.stores.skills import get_skill_store
from utils.stores.subagents import get_subagent_store

logger = logging.getLogger(__name__)

# Module-level singletons (lazy on first import of this module)
pathfinder = Pathfinder()
memory_store = get_memory_store()
skill_store = get_skill_store()
subagent_store = get_subagent_store()


# ---------------------------------------------------------------------------
# Porymap helpers
# ---------------------------------------------------------------------------

def load_porymap_for_pathfinding(state: dict) -> tuple:
    """Load porymap data into *state* for pathfinding, with game-state-aware override support.

    Returns:
        (coord_offset, state) - coord_offset is (offset_x, offset_y) if using override map, else None
    """
    coord_offset = None
    location_name = state.get("player", {}).get("location", "Unknown")

    if not location_name or location_name in ("Unknown", "TITLE_SEQUENCE"):
        return coord_offset, state

    try:
        from utils.mapping.porymap_json_builder import build_json_map_for_llm
        from utils.state_formatter import ROM_TO_PORYMAP_MAP
        from utils.mapping.ascii_map_loader import get_effective_map_name, get_override

        badge_count = 0
        badges = state.get("game", {}).get("badges", [])
        if isinstance(badges, list):
            badge_count = len(badges)
        elif isinstance(badges, int):
            badge_count = badges

        from pokemon_env.porymap_paths import get_porymap_root
        pokeemerald_root = get_porymap_root()
        if not pokeemerald_root:
            return coord_offset, state

        porymap_map_name = ROM_TO_PORYMAP_MAP.get(location_name)
        if not porymap_map_name:
            return coord_offset, state

        effective_map_name = get_effective_map_name(porymap_map_name, badge_count=badge_count)
        override = get_override(effective_map_name)
        if override and ("offset_x" in override or "offset_y" in override):
            offset_x = override.get("offset_x", 0)
            offset_y = override.get("offset_y", 0)
            coord_offset = (offset_x, offset_y)
            logger.info(
                f"Porymap pathfinding: Using override map '{effective_map_name}' "
                f"with coord offset ({offset_x}, {offset_y})"
            )

        try:
            json_map = build_json_map_for_llm(porymap_map_name, pokeemerald_root, badge_count=badge_count)
        except ValueError as e:
            logger.error(f"Failed to build porymap for '{porymap_map_name}': {e}")
            return coord_offset, state

        if json_map and "grid" in json_map:
            raw_tiles = json_map.get("raw_tiles")

            if "map" not in state:
                state["map"] = {}
            if "porymap" not in state["map"]:
                state["map"]["porymap"] = {}

            state["map"]["porymap"]["grid"] = json_map["grid"]
            state["map"]["porymap"]["objects"] = json_map.get("objects", [])
            state["map"]["porymap"]["dimensions"] = json_map.get("dimensions", {})
            state["map"]["porymap"]["warps"] = json_map.get("warps", [])
            state["map"]["porymap"]["raw_tiles"] = raw_tiles

            grid_dims = (
                f"{len(json_map['grid'][0])}x{len(json_map['grid'])}" if json_map.get("grid") else "None"
            )
            logger.info(
                f"Loaded porymap for pathfinding: '{effective_map_name}' "
                f"(ROM: '{location_name}'), grid: {grid_dims}, badges: {badge_count}"
            )

    except Exception as e:
        logger.warning(f"Failed to load porymap data for pathfinding: {e}")

    return coord_offset, state


# ---------------------------------------------------------------------------
# _direct helpers — called by app.py /mcp/* endpoint handlers
# ---------------------------------------------------------------------------

def get_game_state_direct(env, state_formatter, action_history=None, current_obs=None) -> dict:
    """Get game state without HTTP calls.

    Args:
        env: EmeraldEmulator instance
        state_formatter: format_state_for_llm function
        action_history: Optional list of recent actions with start/end positions
        current_obs: Optional numpy array of current frame (from game loop)

    Returns:
        Dictionary with success status and state data including screenshot
    """
    try:
        import base64
        import io
        from PIL import Image
        import numpy as np

        screenshot = None
        if current_obs is not None:
            screenshot = Image.fromarray(current_obs)
            logger.debug("Using current_obs (latest frame from game loop)")
        elif hasattr(env, "current_frame") and env.current_frame is not None:
            screenshot = Image.fromarray(env.current_frame)
            logger.debug("Using env.current_frame (background-polled)")
        elif hasattr(env, "get_screenshot"):
            screenshot = env.get_screenshot()
            logger.debug("Using env.get_screenshot() (direct video buffer - may be stale)")

        state = env.get_comprehensive_state(screenshot=screenshot)

        # For Red: inject red_whole_map so _format_red_map_info() can build the ASCII map
        # and write porymap back into state for movement preview. porymap is NOT pre-injected
        # here because _format_map_info() constructs it from red_whole_map automatically.
        game_type = os.environ.get("GAME_TYPE", "emerald")
        if game_type == "red":
            try:
                if hasattr(env, 'memory_reader') and hasattr(env.memory_reader, 'map_reader'):
                    whole_map = env.memory_reader.map_reader.get_whole_map_data()
                    if whole_map and whole_map.get("grid"):
                        state.setdefault("map", {})["red_whole_map"] = whole_map
                        # add back porymap field for movement preview
                        state["map"]["porymap"] = {
                            "grid": whole_map["grid"],
                            "objects": whole_map.get("objects", []),
                            "dimensions": whole_map.get("dimensions", {}),
                            "warps": whole_map.get("warp_events", []),
                            "raw_tiles": whole_map.get("raw_tiles"),
                        }
            except Exception as e:
                logger.warning(f"Failed to inject red_whole_map in get_game_state_direct: {e}")

        state["_memory_reader"] = getattr(env, "memory_reader", None)
        try:
            state_text = state_formatter(state, action_history=action_history)
        except Exception as formatter_err:
            logger.exception("State formatter failed; returning fallback text with screenshot preserved")
            game_state_name = state.get("game", {}).get("game_state") or "unknown"
            location = state.get("player", {}).get("location") or "Unknown"
            position = state.get("player", {}).get("position") or {}
            state_text = (
                "State text formatter unavailable for this screen.\n"
                f"Game State: {game_state_name}\n"
                f"Location: {location}\n"
                f"Position: X={position.get('x', 'unknown')}, Y={position.get('y', 'unknown')}\n"
                "Use the attached screenshot as the source of truth for the current UI."
            )

        state.pop("_memory_reader", None)

        screenshot_b64 = None
        if screenshot is not None:
            buffered = io.BytesIO()
            screenshot.save(buffered, format="PNG")
            screenshot_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            logger.debug("Encoded fresh screenshot (caching disabled)")

        result = {
            "success": True,
            "state_text": state_text,
            "screenshot_base64": screenshot_b64,
            "player_position": state.get("player", {}).get("position", {}),
            "location": state.get("player", {}).get("location", "Unknown"),
            "raw_state": state,
        }

        return serialize_for_json(result)

    except Exception as e:
        logger.error(f"Failed to get game state: {e}")
        return {"success": False, "error": str(e)}


def load_pathfinding_state(env) -> tuple:
    """Comprehensive state with the current map's porymap data loaded for the pathfinder.

    Returns (state, coord_offset, map_source). coord_offset is (offset_x, offset_y) when
    an override map is used (local = ROM - offset), else None. map_source is
    {"json_map", "porymap_map_name"} for Emerald (json_map carries connections and the
    porymap warps), or {} when no porymap data could be loaded.
    """
    state = env.get_comprehensive_state()

    location_name = state.get("player", {}).get("location", "Unknown")
    coord_offset = None
    map_source: dict = {}

    # --- Red: load native map data as porymap-compatible format ---
    _game_type = os.environ.get("GAME_TYPE", "emerald")
    if _game_type == "red":
        try:
            if hasattr(env, "memory_reader") and hasattr(env.memory_reader, "map_reader"):
                whole_map = env.memory_reader.map_reader.get_whole_map_data()
                if whole_map and whole_map.get("grid"):
                    if "map" not in state:
                        state["map"] = {}
                    state["map"]["porymap"] = {
                        "grid": whole_map["grid"],
                        "objects": whole_map.get("objects", []),
                        "dimensions": whole_map.get("dimensions", {}),
                        "warps": whole_map.get("warp_events", []),
                        "raw_tiles": whole_map.get("raw_tiles"),
                    }
                    logger.info(
                        f"Loaded Red map for pathfinding: '{whole_map.get('location')}', "
                        f"grid: {whole_map['dimensions'].get('width')}x{whole_map['dimensions'].get('height')}"
                    )
        except Exception as e:
            logger.warning(f"Failed to load Red map data for pathfinding: {e}")

    elif location_name and location_name not in ("Unknown", "TITLE_SEQUENCE"):
        try:
            from utils.mapping.porymap_json_builder import build_json_map_for_llm
            from utils.mapping.ascii_map_loader import get_effective_map_name, get_override
            from utils.mapping.dynamic_map_overlay import apply_live_metatile_overlay
            from utils.state_formatter import ROM_TO_PORYMAP_MAP

            badge_count = 0
            badges = state.get("game", {}).get("badges", [])
            if isinstance(badges, list):
                badge_count = len(badges)
            elif isinstance(badges, int):
                badge_count = badges

            from pokemon_env.porymap_paths import get_porymap_root
            pokeemerald_root = get_porymap_root()

            if pokeemerald_root:
                porymap_map_name = ROM_TO_PORYMAP_MAP.get(location_name)

                if porymap_map_name:
                    effective_map_name = get_effective_map_name(porymap_map_name, badge_count=badge_count)
                    override = get_override(effective_map_name)
                    if override and ("offset_x" in override or "offset_y" in override):
                        offset_x = override.get("offset_x", 0)
                        offset_y = override.get("offset_y", 0)
                        coord_offset = (offset_x, offset_y)
                        logger.info(
                            f"Using override map '{effective_map_name}' with coord offset ({offset_x}, {offset_y})"
                        )

                    try:
                        json_map = build_json_map_for_llm(
                            porymap_map_name, pokeemerald_root, badge_count=badge_count
                        )
                    except ValueError as e:
                        logger.error(
                            f"Failed to build porymap for '{porymap_map_name}' due to corrupted tileset: {e}"
                        )
                        json_map = None

                    if json_map and "grid" in json_map:
                        raw_tiles = json_map.get("raw_tiles")
                        player_pos = state.get("player", {}).get("position", {})
                        px = player_pos.get("x", 0)
                        py = player_pos.get("y", 0)

                        if raw_tiles and 0 <= py < len(raw_tiles) and 0 <= px < len(raw_tiles[py]):
                            player_tile = raw_tiles[py][px]
                            if len(player_tile) >= 4:
                                player_elevation = player_tile[3]
                                logger.info(
                                    f"Skipping elevation filtering - pathfinding handles elevation "
                                    f"via E0 connectors (player at elevation {player_elevation})"
                                )

                        if "map" not in state:
                            state["map"] = {}
                        if "porymap" not in state["map"]:
                            state["map"]["porymap"] = {}

                        state["map"]["porymap"]["grid"] = json_map["grid"]
                        state["map"]["porymap"]["objects"] = _shown_objects(env, json_map.get("objects", []))
                        # Where people stand now (they wander off their map-file tile).
                        # Seen live: on ROUTE 117 someone stood at (47,4) on the way to
                        # the grass, and every train call was stopped there in 1 s.
                        try:
                            from server.field_control import npc_positions

                            state["map"]["porymap"]["runtime_objects"] = [
                                {"current_x": px, "current_y": py} for px, py in npc_positions(env.memory_reader)
                            ]
                        except Exception as e:  # positions are a refinement; the static list remains
                            logger.debug(f"live object positions unavailable: {e}")
                        state["map"]["porymap"]["dimensions"] = json_map.get("dimensions", {})
                        state["map"]["porymap"]["warps"] = json_map.get("warps", [])
                        # The tiles that really warp (map.json warp_events): the builder
                        # moves its "warps" next to the door, so these are kept apart.
                        # Spots warps only land on are plain floor (arrival_only_warps).
                        from utils.mapping.porymap_state import arrival_only_warps

                        landing = arrival_only_warps(porymap_map_name)
                        state["map"]["porymap"]["raw_warps"] = [
                            (int(w.get("x", -1)), int(w.get("y", -1)))
                            for i, w in enumerate(_raw_map_json(porymap_map_name).get("warp_events") or [])
                            if i not in landing
                        ]
                        # Floor switches and scene triggers that would fire if stepped on:
                        # walks go round them unless that is the only way (find_path).
                        state["map"]["porymap"]["triggers"] = active_triggers(env, porymap_map_name)
                        # Rocks and trees the party can clear: walkable for the planner
                        # (at a cost), cleared by the walk when it bumps into one.
                        state["map"]["porymap"]["clearable"] = list(clearable_obstacles(porymap_map_name))
                        state["map"]["porymap"]["raw_tiles"] = raw_tiles

                        # For selected dynamic maps (e.g. Mauville Gym), replace static
                        # porymap tiles with the current live map buffer state.
                        apply_live_metatile_overlay(state, env, location_name)
                        map_source = {"json_map": json_map, "porymap_map_name": porymap_map_name}

                        grid_dims = (
                            f"{len(json_map['grid'][0])}x{len(json_map['grid'])}"
                            if json_map.get("grid")
                            else "None"
                        )
                        logger.info(
                            f"Loaded UNFILTERED porymap for pathfinding: '{porymap_map_name}' "
                            f"(ROM: '{location_name}'), grid: {grid_dims}"
                        )
                    else:
                        logger.warning(f"Porymap data for '{porymap_map_name}' missing grid data")
                else:
                    logger.warning(f"Could not map ROM location '{location_name}' to porymap map name")
            else:
                logger.warning("Could not find pokeemerald root for porymap data")
        except Exception as porymap_err:
            logger.warning(f"Failed to load porymap data for pathfinding: {porymap_err}")

    return state, coord_offset, map_source


def _shown_objects(env, objects: list) -> list:
    """The map's objects minus those whose hide flag the game has set (caller holds the lock).

    Seen live: on ROUTE 110 five TEAM AQUA grunts and PROF BIRCH from an earlier scene
    (FLAG_HIDE_ROUTE_110_TEAM_AQUA, FLAG_HIDE_ROUTE_110_BIRCH, both set) were still in the
    pathfinder's object list. They stand in a row across the only walk north, so every
    navigate_to toward the rival found "no complete path". The prompt's map already
    filtered them (porymap_state, npc_source "flag_filtered"); the pathfinder did not.
    """
    from utils.mapping.flag_hide_constants import FLAG_HIDE_IDS

    reader = getattr(env, "memory_reader", None)
    if reader is None:
        return list(objects)
    shown = []
    for obj in objects:
        flag_id = FLAG_HIDE_IDS.get(str(obj.get("flag", "0")))
        try:
            hidden = flag_id is not None and reader._is_event_flag_set(flag_id)
        except Exception:  # unreadable: keep the object (the old behaviour)
            hidden = False
        if not hidden:
            shown.append(obj)
    return shown


def navigate_to_direct(
    env,
    x,
    y,
    reason: str = "",
    variance: Optional[str] = None,
    consider_npcs: bool = True,
    blocked_coords: Optional[List[Tuple[int, int]]] = None,
) -> dict:
    """Calculate path to coordinates without HTTP calls.

    Returns buttons to be queued via take_action.
    """
    try:
        x = int(x)
        y = int(y)

        variance_level = None
        nav_reason = "" if reason is None else str(reason)

        if variance is not None:
            variance_str = str(variance).strip()
            variance_lower = variance_str.lower()
            if variance_lower in {"low", "medium", "high"}:
                variance_level = variance_lower
            elif variance_lower in {"", "none"}:
                variance_level = None
            elif not reason:
                nav_reason = variance_str
        elif nav_reason.lower() in {"low", "medium", "high"}:
            variance_level = nav_reason.lower()
            nav_reason = ""

        state, coord_offset, _map_source = load_pathfinding_state(env)

        player_pos = state.get("player", {}).get("position", {})
        rom_start_x = player_pos.get("x", 0)
        rom_start_y = player_pos.get("y", 0)

        start_x = rom_start_x
        start_y = rom_start_y
        goal_x = x
        goal_y = y
        if coord_offset:
            offset_x, offset_y = coord_offset
            start_x = rom_start_x - offset_x
            start_y = rom_start_y - offset_y
            logger.info(
                f"Coordinate translation: ROM ({rom_start_x}, {rom_start_y}) -> "
                f"local ({start_x}, {start_y}), goal ({goal_x}, {goal_y})"
            )

        start = (start_x, start_y)
        goal = (goal_x, goal_y)

        goal_was_blocked = False
        goal_cell = None
        location_name = state.get("player", {}).get("location", "")
        width = 0
        map_data = state.get("map", {}).get("porymap", {})
        if map_data.get("grid"):
            grid = map_data["grid"]
            width = max(len(r) for r in grid) if grid else 0
            # A target off the map used to be clamped silently to the edge; the model then
            # thought it had left the map. Say so and point at the tool that does leave it.
            if not (0 <= goal_y < len(grid) and 0 <= goal_x < len(grid[goal_y])):
                side = "up" if goal_y < 0 else "down" if goal_y >= len(grid) else "left" if goal_x < 0 else "right"
                hint = exit_hint(location_name, side) or "To leave the map use go_to_exit with an entry of the EXITS list."
                return {
                    "success": False,
                    "error": f"({x}, {y}) is outside this map (x 0-{width - 1}, y 0-{len(grid) - 1}). {hint}",
                    "target": f"({x}, {y})",
                    "reason": "out_of_map",
                }
            if 0 <= goal_y < len(grid):
                row = grid[goal_y]
                if isinstance(row, (list, str)) and 0 <= goal_x < len(row):
                    cell = row[goal_x] if isinstance(row, str) else row[goal_x]
                    goal_cell = cell
                    if cell in ("#", "W", "X"):
                        goal_was_blocked = True

        buttons = pathfinder.find_path(
            start, goal, state, variance=variance_level, consider_npcs=consider_npcs, blocked_coords=blocked_coords
        )

        if not buttons and goal_was_blocked and start != goal:
            grid = map_data["grid"]
            near_edge = (
                "up" if goal_y <= 1 else "down" if goal_y >= len(grid) - 2
                else "left" if goal_x <= 1 else "right" if goal_x >= width - 2 else ""
            )
            hint = exit_hint(location_name, near_edge) if near_edge else ""
            return {
                "success": False,
                "error": (
                    f"({x}, {y}) is not walkable (map tile '{goal_cell}') and you already stand on the "
                    f"nearest reachable tile ({start_x}, {start_y}). Pick a walkable tile "
                    "('.' or '~' on the ASCII map), or use go_to_exit to leave the map."
                    + (f" {hint}" if hint else "")
                ),
                "target": f"({x}, {y})",
                "reason": "unwalkable",
            }

        if not buttons:
            error_msg = f"No path found from ({start_x}, {start_y}) to ({x}, {y})"

            map_data = state.get("map", {}).get("porymap", {})
            if map_data.get("grid"):
                grid = map_data["grid"]
                if 0 <= y < len(grid):
                    row = grid[y]
                    if isinstance(row, (list, str)) and 0 <= x < len(row):
                        cell = row[x] if isinstance(row, str) else row[x]
                        if cell == "#":
                            error_msg += " - Target is blocked by a wall or obstacle"
                        elif cell == "W":
                            error_msg += " - Target requires Surf (water)"
                        elif cell in ["X", "?"]:
                            error_msg += " - Target is out of bounds or unexplored"
                        else:
                            error_msg += f" - Target tile '{cell}' may be unreachable from current position"

            manhattan_dist = abs(x - start_x) + abs(y - start_y)
            if manhattan_dist > 100:
                error_msg += f" (distance: {manhattan_dist} tiles - may be too far)"

            return {"success": False, "error": error_msg, "target": f"({x}, {y})", "reason": "unreachable"}

        nav_reason_text = f"Navigating from ({start_x}, {start_y}) to ({x}, {y})"
        if nav_reason:
            nav_reason_text += f": {nav_reason}"
        if variance_level:
            nav_reason_text += f" (variance={variance_level})"

        logger.info(f"🗺️ {nav_reason_text} - Buttons calculated: {len(buttons)}")

        from server.field_control import end_of_path

        # Same coordinate frame as the requested target (map-local on override maps).
        local_end = end_of_path(start, buttons)
        path_end = local_end

        result = {
            "success": True,
            "buttons": buttons,
            "target": f"({x}, {y})",
            "path_end": list(path_end),
            "coord_offset": list(coord_offset) if coord_offset else None,
            "path_length": len(buttons),
            "reason": nav_reason,
            "variance": variance_level or "none",
        }

        # The pathfinder quietly retargets an unwalkable goal and returns partial paths;
        # without this note the model believes it reached the tile it asked for.
        if local_end != goal:
            if goal_was_blocked:
                result["note"] = (
                    f"({x}, {y}) is not walkable (map tile '{goal_cell}'); walking to the nearest "
                    f"reachable tile ({path_end[0]}, {path_end[1]}) instead"
                )
            else:
                result["note"] = (
                    f"No complete path to ({x}, {y}); walking to ({path_end[0]}, {path_end[1]}), "
                    "the closest point the pathfinder could reach"
                )

        return result
    except Exception as e:
        logger.error(f"Failed to navigate: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


def list_map_exits(json_map: dict, porymap_map_name: str, coord_offset=None, live_tile=None) -> list:
    """Every exit the EXITS prompt line lists, with the coordinates go_to_exit needs.

    Warp coordinates come from the map's raw warp_events: the json_map copies are
    nudged toward the map edge for display (a house's exit mats at y=8 are listed at
    y=9), and walking to the listed tile would miss the warp.

    Warps that only mark a landing spot (porymap_state.arrival_only_warps) are left out,
    unless live_tile(x, y), the tile in the game's map buffer now, shows a warp behavior
    there: scripts can lay one down (pret PetalburgCity_Gym_OnLoad opens the doors of
    rooms whose trainer is beaten).
    """
    import json as _json

    from pokemon_env.porymap_paths import get_porymap_root
    from server.field_control import EDGE_NAMES
    from utils.mapping.porymap_state import (
        _readable_map_name,
        arrival_only_warps,
        connection_open_span,
        sign_door_warps,
    )

    exits = []
    for conn in json_map.get("connections", []) or []:
        direction = conn.get("direction")
        if direction not in EDGE_NAMES:
            continue  # dive/emerge connections are not walkable exits
        name = _readable_map_name(conn.get("map", "?"))
        exits.append({
            "kind": "edge",
            "direction": direction,
            "open_span": connection_open_span(conn),
            "dead_ends": _edge_dead_ends(
                porymap_map_name or "", str(conn.get("map", "")), direction, int(conn.get("offset", 0) or 0)
            ),
            "name": name,
            "label": f"{EDGE_NAMES[direction]} edge -> {name}",
        })

    raw_warps = []
    root = get_porymap_root()
    if root and porymap_map_name:
        map_file = Path(root) / "data" / "maps" / porymap_map_name / "map.json"
        try:
            with open(map_file, encoding="utf-8") as fh:
                raw_warps = _json.load(fh).get("warp_events", []) or []
        except (OSError, ValueError) as e:
            logger.warning(f"go_to_exit: could not read warps from {map_file}: {e}")
    listed = json_map.get("warps", []) or []
    ox, oy = coord_offset or (0, 0)
    arrival_only = arrival_only_warps(porymap_map_name or "")
    sign_doors = sign_door_warps(porymap_map_name or "")
    for i, warp in enumerate(raw_warps):
        if i in arrival_only and not _fires_now(live_tile, warp):
            continue  # a spot warps land on: stepping on it goes nowhere
        name = _readable_map_name(warp.get("dest_map", "?"))
        shown = listed[i] if i < len(listed) else warp
        listed_xy = (shown.get("x"), shown.get("y"))
        exits.append({
            "kind": "warp",
            "x": int(warp.get("x", 0)) - ox,
            "y": int(warp.get("y", 0)) - oy,
            "listed_xy": listed_xy,
            "name": name,
            "label": f"door/warp at ({listed_xy[0]}, {listed_xy[1]}) -> {name}",
            "dest_map": str(warp.get("dest_map", "")),
            "arrival": warp_arrival(warp),
            # Read from the tile in front, not stepped on (porymap_state.sign_door_warps).
            "sign_door": i in sign_doors,
        })
    return exits


def _fires_now(live_tile, warp: dict) -> bool:
    """list_map_exits: the game's map buffer shows a warp behavior on the warp's tile."""
    from utils.mapping.porymap_state import lands_only

    if live_tile is None:
        return False
    try:
        tile = live_tile(int(warp.get("x", -1)), int(warp.get("y", -1)))
    except Exception as e:  # the live read is a refinement; the static data stands
        logger.debug(f"live tile unreadable: {e}")
        return False
    return tile is not None and len(tile) > 2 and not tile[2] and not lands_only(tile)


def warp_arrival(warp: dict) -> Optional[Tuple[int, int]]:
    """The tile a warp puts the player on, in its destination map's coordinates: the
    destination's warp_events[dest_warp_id] (pret src/overworld.c
    SetWarpDestinationToMapWarp, SetPlayerCoordsFromWarp). None when the destination is
    set at run time."""
    from utils.mapping.porymap_state import _map_dirs_by_const

    dest = _map_dirs_by_const().get(str(warp.get("dest_map", "")))
    try:
        target = (_raw_map_json(dest).get("warp_events") or [])[int(warp.get("dest_warp_id"))]
    except (TypeError, ValueError, IndexError):
        return None
    return int(target.get("x", -1)), int(target.get("y", -1))


def came_through_warp(arrival: Tuple[int, int], warp_tile: Tuple[int, int], position: Tuple[int, int]) -> bool:
    """Whether the player at position went through a warp that leads to another spot of
    the same map (the map id does not change, so only where the player stands tells).

    On the arrival tile, or one tile off it: pret src/field_effect.c
    LavaridgeGymB1FWarpExitEffect_PopOut throws the player one tile east of where a
    geyser lands them. Still on the warp tile means it did not fire.
    """
    if tuple(position) == tuple(warp_tile):
        return False
    return abs(position[0] - arrival[0]) + abs(position[1] - arrival[1]) <= 1


@lru_cache(maxsize=256)
def _map_connections(porymap_map_name: str) -> tuple:
    """(direction, readable destination) pairs from a map's raw map.json."""
    import json as _json

    from pokemon_env.porymap_paths import get_porymap_root
    from utils.mapping.porymap_state import _readable_map_name

    root = get_porymap_root()
    if not root or not porymap_map_name:
        return ()
    try:
        with open(Path(root) / "data" / "maps" / porymap_map_name / "map.json", encoding="utf-8") as fh:
            connections = _json.load(fh).get("connections") or []
    except (OSError, ValueError):
        return ()
    return tuple((c.get("direction"), _readable_map_name(c.get("map", "?"))) for c in connections)


def exit_hint(location_name: str, direction: str) -> str:
    """'To go north to OLDALE TOWN use go_to_exit("north").' if this map has that edge exit.

    direction is a button (UP/DOWN/LEFT/RIGHT) or a connection direction (up/down/...).
    A small model that keeps walking into the trees along an edge needs the exact call,
    not a reminder that an EXITS list exists.
    """
    from server.field_control import EDGE_NAMES
    from utils.state_formatter import ROM_TO_PORYMAP_MAP

    wanted = str(direction).lower()
    for conn_dir, destination in _map_connections(ROM_TO_PORYMAP_MAP.get(location_name, "")):
        if conn_dir == wanted:
            edge = EDGE_NAMES[conn_dir]
            return f'To go {edge} to {destination} use go_to_exit("{edge}").'
    return ""


def exit_hint_for_text(location_name: str, text: str, position: Optional[Tuple[int, int]] = None) -> str:
    """The go_to_exit call for a neighbouring map that `text` names, or "".

    position is the player's map tile, so a far route starts from the part of the map
    the player can actually walk from.

    Seen live: the model wrote "Destination: Route 101 (south edge)" in its navigate_to
    reason and then walked one tile per call for minutes, wobbling against walls,
    instead of calling go_to_exit. Names are compared without spaces and case, so
    "Route101", "ROUTE 101" and "route 101" all match, and a town or city may be named
    without the suffix ("Littleroot"). The model often copies the whole route
    ("Oldale → Route 101 → Littleroot"), naming the maps on both sides, so every named
    neighbour is listed rather than guessing which way it means to go. When no neighbour
    is named but a map further away is, the hint gives the way there (_far_map_hint).
    """
    from server.field_control import EDGE_NAMES
    from utils.state_formatter import ROM_TO_PORYMAP_MAP

    squashed = "".join(str(text).upper().split())
    if not squashed:
        return ""

    def names(name: str) -> bool:
        # "ROUTE11" must not match inside "ROUTE110".
        start = squashed.find(name)
        while start != -1:
            end = start + len(name)
            if end == len(squashed) or not squashed[end].isdigit():
                return True
            start = squashed.find(name, start + 1)
        return False

    here = "".join(str(location_name).upper().split())

    def named(destination: str) -> bool:
        words = destination.split()
        short = words[:-1] if len(words) > 1 and words[-1] in ("TOWN", "CITY") else words
        # Seen live: inside RUSTBORO CITY GYM the model wrote "In Rustboro City Gym", and
        # "Rustboro City" matched, so every result said go_to_exit("RUSTBORO CITY") and
        # pulled it out of the gym it had to fight in. Inside a building, the town's name
        # in the text is about where the player is, not where to go.
        if any(here.startswith("".join(w)) for w in (words, short)):
            return False
        return names("".join(words)) or names("".join(short))

    found = []
    for conn_dir, destination in _map_connections(ROM_TO_PORYMAP_MAP.get(location_name, "")):
        if conn_dir in EDGE_NAMES and named(destination):
            found.append((destination, EDGE_NAMES[conn_dir]))
    if not found:
        return _far_map_hint(ROM_TO_PORYMAP_MAP.get(location_name, ""), named, position)
    if len(found) == 1:
        destination, edge = found[0]
        return (
            f'{destination} is off this map: go_to_exit("{edge}") walks to the opening '
            f"and through it in one call."
        )
    calls = "; ".join(f'to {d} use go_to_exit("{e}")' for d, e in found)
    return f"These maps are off this map: {calls}. Each call walks to the opening and through it."


@lru_cache(maxsize=None)
def _raw_map_json(map_dir: str) -> dict:
    """A map's data/maps/<dir>/map.json, or {} when it can't be read."""
    import json as _json

    from pokemon_env.porymap_paths import get_porymap_root

    root = get_porymap_root()
    if not root or not map_dir:
        return {}
    try:
        with open(Path(root) / "data" / "maps" / map_dir / "map.json", encoding="utf-8") as fh:
            return _json.load(fh)
    except (OSError, ValueError):
        return {}


# Map objects that block the way until an HM clears them. Seen live (2026-11-03): two
# BREAKABLE_ROCKs at (24,4) and (24,5) seal Rusturf Tunnel, but the walking graph read
# only tiles, so the heal-first route sent the AI through them to Verdanturf.
_OBSTACLE_GRAPHICS = ("OBJ_EVENT_GFX_BREAKABLE_ROCK", "OBJ_EVENT_GFX_PUSHABLE_BOULDER", "OBJ_EVENT_GFX_CUTTABLE_TREE")

# flag id -> set?, installed by the server (server/app.py) so routes know which obstacles
# are gone for good. Without it every obstacle counts as present: a missing route is
# better than one through solid rock.
flag_reader = None


@lru_cache(maxsize=1)
def _flag_ids() -> dict:
    """FLAG_HIDE_RUSTURF_TUNNEL_ROCK_1 -> id, from include/constants/flags.h (copied from pret)."""
    from pokemon_env.porymap_paths import get_porymap_root

    root = get_porymap_root()
    path = Path(root) / "include" / "constants" / "flags.h" if root else None
    ids = {}
    if path is None or not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "#define" and parts[1].startswith("FLAG_"):
            value = parts[2]
            try:
                ids[parts[1]] = int(value, 16) if value.lower().startswith("0x") else int(value)
            except ValueError:
                continue  # an expression like (FLAG_HIDDEN_ITEMS_START + 0x2D): not an obstacle flag
    return ids


SYSTEM_FLAGS = 0x860  # include/constants/flags.h: (TRAINER_FLAGS_END + 1)


@lru_cache(maxsize=1)
def _var_ids() -> dict:
    """VAR_TEMP_0 -> 0x4000, from include/constants/vars.h (copied from pret). Values are
    a number or "(NAME + 0xN)" with NAME defined earlier in the file."""
    from pokemon_env.porymap_paths import get_porymap_root

    root = get_porymap_root()
    path = Path(root) / "include" / "constants" / "vars.h" if root else None
    ids: dict = {}
    if path is None or not path.exists():
        return ids

    def number(text: str) -> Optional[int]:
        text = text.strip().strip("()")
        try:
            return int(text, 16) if text.lower().startswith("0x") else int(text)
        except ValueError:
            return ids.get(text)

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or parts[0] != "#define":
            continue
        value = parts[2].split("//")[0].strip()
        if value.startswith("(") and "+" in value:
            base, _, offset = value.strip("()").partition("+")
            left, right = number(base), number(offset)
            if left is not None and right is not None:
                ids[parts[1]] = left + right
        else:
            found = number(value)
            if found is not None:
                ids[parts[1]] = found
    return ids


def active_triggers(env, map_dir: str) -> List[Tuple[int, int]]:
    """Tiles of map.json coord_events "trigger"s that fire when stepped on now: the
    event's var equals its var_value (pret: a coord event runs its script when
    VarGet(var) == value). A var that can't be read or resolved counts as firing.

    Seen live in MAUVILLE CITY GYM: after the AI stood on the south-west switch (0,15),
    talk_to's walk to Wattson crossed the switches at (4,12) and (3,9), each of which
    flipped the barriers back. Already spent triggers (Birch's ROUTE 110 scene once
    VAR_REGISTER_BIRCH_STATE has moved on) stay walkable.
    """
    out = []
    for event in _raw_map_json(map_dir).get("coord_events") or []:
        if event.get("type") != "trigger":
            continue
        var_id = _var_ids().get(str(event.get("var", "")))
        try:
            wanted = int(str(event.get("var_value", "0")), 0)
        except ValueError:
            wanted = None
        fires = True
        if var_id is not None and wanted is not None and env is not None:
            try:
                value = env.memory_reader.read_game_var(var_id)
                fires = value is None or value == wanted
            except Exception as e:  # unreadable var: treat as live, the safer guess
                logger.debug(f"trigger var {event.get('var')} unreadable: {e}")
        if fires:
            out.append((int(event.get("x", -1)), int(event.get("y", -1))))
    return out


def story_scenes(env, map_dir: str) -> List[Tuple[str, List[Tuple[int, int]]]]:
    """[(scene name, its tiles), ...] for the map's story scenes that will play when
    stepped on now (caller holds the emulator lock).

    A story scene is a coord_events trigger on a saved var (not VAR_TEMP_*, which floor
    switches and gates use, and not TRIGGER_RUN_IMMEDIATELY weather) whose var equals its
    var_value now; an unreadable var is left out, so only scenes known to fire are named.

    Seen live: on METEOR FALLS 1F 1R the objective was "Witness Team Magma/Aqua
    confrontation", and the room looked empty: Magma and Aqua stay hidden (FLAG_HIDE_*)
    until the scene at (14,18) runs (pret MeteorFalls_1F_1R_EventScript_
    MagmaStealsMeteoriteScene, VAR_METEOR_FALLS_STATE == 0). The model walked back out.
    """
    scenes: Dict[str, List[Tuple[int, int]]] = {}
    for event in _raw_map_json(map_dir).get("coord_events") or []:
        name = str(event.get("var", ""))
        var_id = _var_ids().get(name)
        if event.get("type") != "trigger" or var_id is None or name.startswith("VAR_TEMP_") or env is None:
            continue
        try:
            wanted = int(str(event.get("var_value", "0")), 0)
            value = env.memory_reader.read_game_var(var_id)
        except Exception as e:  # unreadable: not named
            logger.debug(f"scene var {name} unreadable: {e}")
            continue
        if value != wanted:
            continue
        script = str(event.get("script", ""))
        short = script.rpartition("EventScript_")[2] or script
        scenes.setdefault(short, []).append((int(event.get("x", -1)), int(event.get("y", -1))))
    return list(scenes.items())


@lru_cache(maxsize=1)
def _first_visit_flags() -> dict:
    """'DEWFORD TOWN' -> FLAG_VISITED_DEWFORD_TOWN's id, and the same for FLAG_LANDMARK_*.

    The towns' map scripts set FLAG_VISITED_* on the first arrival (it unlocks FLY) and
    buildings and caves set FLAG_LANDMARK_* on the first entry (the region map's labels);
    neither is ever cleared. Both are written as (SYSTEM_FLAGS + 0x11) in flags.h.
    """
    from pokemon_env.porymap_paths import get_porymap_root

    root = get_porymap_root()
    path = Path(root) / "include" / "constants" / "flags.h" if root else None
    places = {}
    if path is None or not path.exists():
        return places
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0] != "#define" or parts[2] != "(SYSTEM_FLAGS" or parts[3] != "+":
            continue
        for prefix in ("FLAG_VISITED_", "FLAG_LANDMARK_"):
            if parts[1].startswith(prefix):
                try:
                    places[parts[1][len(prefix):].replace("_", " ")] = SYSTEM_FLAGS + int(parts[4].rstrip(")"), 16)
                except ValueError:
                    continue
    return places


def first_visit_flag(target: str) -> Optional[int]:
    """The flag the game sets the first time the player gets to the place target names.

    Seen live: "Travel to Mr. Briney's cottage" stayed open while the boat was already
    at sea; the map name only says where the player is now, and the model never called
    complete. FLAG_LANDMARK_MR_BRINEY_HOUSE was set. None for places without such a flag
    (routes, most caves).
    """
    for place, flag in _first_visit_flags().items():
        if _names_same_place(place, target):
            return flag
    return None


# Obstacles the party can clear on the walk: graphics -> (field move, badge number that
# allows it outside battle). pret: Cut needs the Stone Badge (FLAG_BADGE01_GET), Rock
# Smash the Dynamo Badge (FLAG_BADGE03_GET). Strength boulders are left out: pushing one
# needs room behind it, which a route can't promise.
FLAG_BADGE01_GET = 0x867  # include/constants/flags.h SYSTEM_FLAGS + 0x7
FIELD_MOVE_OBSTACLES = {
    "OBJ_EVENT_GFX_CUTTABLE_TREE": ("CUT", 1),
    "OBJ_EVENT_GFX_BREAKABLE_ROCK": ("ROCK_SMASH", 3),
}
# () -> set of field moves usable now (known by a party Pokémon, badge set), installed by
# the server. None: no obstacle is clearable.
field_moves_reader = None


_field_moves_cache: list = [0.0, set()]


def _usable_field_moves() -> set:
    """field_moves_reader's answer, kept for a second: a route search asks once per map."""
    import time

    if field_moves_reader is None:
        return set()
    if time.monotonic() - _field_moves_cache[0] < 1.0:
        return _field_moves_cache[1]
    try:
        moves = set(field_moves_reader())
    except Exception as e:  # unreadable party: treat obstacles as walls
        logger.debug(f"field moves unreadable: {e}")
        moves = set()
    _field_moves_cache[:] = [time.monotonic(), moves]
    return moves


def clearable_obstacles(map_dir: str) -> dict:
    """(x, y) -> field move for the map's rocks and trees the party can clear now.

    Seen live: on ROUTE 111 the breakable rocks at (19,100) and (18,101) block the only way
    north. With HM06 taught and the Dynamo Badge, go_to_exit("west") still said "no walkable
    path": every obstacle was a wall until smashed, so no walk ever led to it.
    """
    usable = _usable_field_moves()
    out = {}
    for obj in _raw_map_json(map_dir).get("object_events") or []:
        move = FIELD_MOVE_OBSTACLES.get(obj.get("graphics_id"), (None, 0))[0]
        if move in usable:
            out[(int(obj.get("x", -1)), int(obj.get("y", -1)))] = move
    return out


# () -> squashed names of the items in the bag (_squash_name), installed by the server.
# None: no item is held, so every gate below stays shut.
items_reader = None


_items_cache: list = [0.0, set()]


def _held_items() -> set:
    """items_reader's answer, kept for a second like _usable_field_moves."""
    import time

    if items_reader is None:
        return set()
    if time.monotonic() - _items_cache[0] < 1.0:
        return _items_cache[1]
    try:
        items = set(items_reader())
    except Exception as e:  # unreadable bag: gates stay shut
        logger.debug(f"bag unreadable: {e}")
        items = set()
    _items_cache[:] = [time.monotonic(), items]
    return items


@lru_cache(maxsize=None)
def _script_labels(map_dir: str) -> tuple:
    """(label -> its lines, labels in file order) from data/maps/<dir>/scripts.inc."""
    from pokemon_env.porymap_paths import get_porymap_root

    root = get_porymap_root()
    path = Path(root) / "data" / "maps" / map_dir / "scripts.inc" if root and map_dir else None
    labels: dict = {}
    order: list = []
    if path is None or not path.exists():
        return labels, order
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.split("@")[0].strip()
        if not text:
            continue
        if not line[0].isspace() and text.endswith(":"):
            current = text.rstrip(":")
            labels[current] = []
            order.append(current)
        elif current is not None:
            labels[current].append(text)
    return labels, order


def _script_lines(map_dir: str, label: str) -> List[str]:
    """Every line a map script can run from label: its gotos, calls and the labels it
    falls through into (a label without end/return/goto runs on into the next one)."""
    labels, order = _script_labels(map_dir)
    lines, seen, stack = [], set(), [label]
    while stack:
        name = stack.pop()
        if name in seen or name not in labels:
            continue  # shared scripts (data/scripts/*.inc) aren't read
        seen.add(name)
        body = labels[name]
        lines.extend(body)
        for line in body:
            command, _, args = line.partition(" ")
            if command in ("goto", "call") or command.startswith(("goto_if", "call_if")):
                stack.append(args.split(",")[-1].strip())
        last = body[-1].partition(" ")[0] if body else ""
        if last not in ("end", "return", "goto") and order.index(name) + 1 < len(order):
            stack.append(order[order.index(name) + 1])
    return lines


_PLAYER_IDS = ("LOCALID_PLAYER", "OBJ_EVENT_ID_PLAYER")


@lru_cache(maxsize=None)
def turn_back_gates(map_dir: str) -> tuple:
    """((x, y), items that open it) for each trigger that turns the player back.

    A coord trigger on a VAR_TEMP_* var fires on every visit (temp vars are cleared on
    each map load, pret ClearTempFieldEventData), and one whose script moves the player
    is a gate: pret Route111_EventScript_ViciousSandstormTrigger pushes the player back
    out of the desert unless checkitem ITEM_GO_GOGGLES finds them; the Cycling Road
    entrances push a player on foot back. The items are the ones its script checks for
    (none: nothing in the bag opens it).

    Seen live: go_to("Route 113") from ROUTE 111 (0,66), south of the desert, planned
    straight north across it with no Go-Goggles; the real way is round by Fiery Path.
    """
    gates = []
    for event in _raw_map_json(map_dir).get("coord_events") or []:
        if event.get("type") != "trigger" or not str(event.get("var", "")).startswith("VAR_TEMP_"):
            continue
        lines = _script_lines(map_dir, str(event.get("script", "")))
        moves_player = any(
            line.partition(" ")[0] == "applymovement" and line.partition(" ")[2].split(",")[0].strip() in _PLAYER_IDS
            for line in lines
        )
        if not moves_player:
            continue
        items = frozenset(
            _squash_name(line.partition(" ")[2].split(",")[0].strip().removeprefix("ITEM_"))
            for line in lines
            if line.partition(" ")[0] == "checkitem"
        )
        gates.append(((int(event.get("x", -1)), int(event.get("y", -1))), items))
    return tuple(gates)


def _shut_gates(map_dir: str) -> set:
    """turn_back_gates tiles the bag does not open now."""
    held = _held_items()
    return {tile for tile, items in turn_back_gates(map_dir) if not items or not items <= held}


def _obstacle_tiles(map_dir: str) -> frozenset:
    """Tiles holding an obstacle object that is still there (its hide flag not set), less
    the ones the party can clear now (clearable_obstacles: routes go through those), and
    the turn_back_gates the bag does not open."""
    blocked = _shut_gates(map_dir)
    usable = _usable_field_moves()
    for obj in _raw_map_json(map_dir).get("object_events") or []:
        if obj.get("graphics_id") not in _OBSTACLE_GRAPHICS:
            continue
        if FIELD_MOVE_OBSTACLES.get(obj.get("graphics_id"), (None, 0))[0] in usable:
            continue
        flag = _flag_ids().get(str(obj.get("flag", "")))
        if flag is not None and flag_reader is not None:
            try:
                if flag_reader(flag):
                    continue  # removed for good (e.g. the tunnel opened)
            except Exception:
                pass  # unreadable: keep it as an obstacle
        blocked.add((int(obj.get("x", -1)), int(obj.get("y", -1))))
    return frozenset(blocked)


def _walk_areas(map_dir: str) -> dict:
    """(x, y) -> area number for every tile a player can stand on, on foot.

    Two tiles share an area when one can be walked to from the other without
    surfing, cutting, smashing or crossing a wall. Ledges count as two-way here, which
    keeps a route's halves whole for arrived_at; routes use _route_areas.
    """
    return _walk_areas_without(map_dir, _obstacle_tiles(map_dir) if map_dir else frozenset())


# Ledge tiles in the static grid, with the way they can be jumped (pokeemerald
# MB_JUMP_*: the player hops over the ledge tile, from behind it to the tile beyond).
_LEDGE_STEPS = {"→": (1, 0), "←": (-1, 0), "↓": (0, 1), "↑": (0, -1)}


def _route_areas(map_dir: str) -> dict:
    """_walk_areas with every ledge tile a wall; _ledge_jumps joins the two sides one way.

    Seen live: on GRANITE CAVE 1F a row of east-facing ledges cuts the ladder from B1F
    (4,21) off from Steven's Room; counted two-way, the route sent the AI up that
    ladder, where the door can only be seen.
    """
    return _walk_areas_without(map_dir, _obstacle_tiles(map_dir) if map_dir else frozenset(), True)


@lru_cache(maxsize=None)
def _walk_areas_without(map_dir: str, obstacles: frozenset, ledges_are_walls: bool = False) -> dict:
    from utils.mapping.porymap_state import _EDGE_BLOCKED, _static_grid

    grid = _static_grid(map_dir) if map_dir else None
    if not grid:
        return {}
    grid = [
        "".join("#" if (x, y) in obstacles else cell for x, cell in enumerate(row))
        for y, row in enumerate(grid)
    ] if obstacles else grid
    blocked = set(_EDGE_BLOCKED) | (set(_LEDGE_STEPS) if ledges_are_walls else set())
    area: dict = {}
    number = -1
    for y0, row in enumerate(grid):
        for x0, cell in enumerate(row):
            if cell in blocked or (x0, y0) in area:
                continue
            number += 1
            stack = [(x0, y0)]
            area[(x0, y0)] = number
            while stack:
                x, y = stack.pop()
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if (nx, ny) in area or ny < 0 or ny >= len(grid) or nx < 0 or nx >= len(grid[ny]):
                        continue
                    if grid[ny][nx] in blocked:
                        continue
                    area[(nx, ny)] = number
                    stack.append((nx, ny))
    return area


def _ledge_jumps(map_dir: str) -> frozenset:
    """(from area, to area) for each ledge, in _route_areas numbers: the only way across."""
    from utils.mapping.porymap_state import _static_grid

    grid = _static_grid(map_dir) if map_dir else None
    areas = _route_areas(map_dir)
    jumps = set()
    for y, row in enumerate(grid or ()):
        for x, cell in enumerate(row):
            step = _LEDGE_STEPS.get(cell)
            if step is None:
                continue
            behind, beyond = (x - step[0], y - step[1]), (x + step[0], y + step[1])
            if behind in areas and beyond in areas and areas[behind] != areas[beyond]:
                jumps.add((areas[behind], areas[beyond]))
    return frozenset(jumps)


def _areas_at(map_dir: str, x: int, y: int, areas: Optional[dict] = None) -> set:
    """Areas of a tile, or of the tiles beside it when it is a door or warp in a wall."""
    areas = _walk_areas(map_dir) if areas is None else areas
    if (x, y) in areas:
        return {areas[(x, y)]}
    return {areas[c] for c in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)) if c in areas}


@lru_cache(maxsize=256)
def _edge_dead_ends(map_dir: str, dest_const: str, direction: str, offset: int) -> frozenset:
    """Positions along this map's edge (x for up/down, y for left/right, this map's frame)
    whose arrival tile next door is in an area with no way on but back here.

    Seen live: go_to_exit("east") from VERDANTURF TOWN (19,6) took the nearest edge tile,
    y=6, and landed on ROUTE 117 (0,6): a one-tile notch walled in on three sides, so
    every walk and train call from there failed. Route 117 goes on from y=8.
    """
    from utils.mapping.porymap_state import _map_dirs_by_const, _static_grid

    dest = _map_dirs_by_const().get(dest_const)
    grid = _static_grid(dest) if dest else None
    if not grid or direction not in ("up", "down", "left", "right"):
        return frozenset()
    height, width = len(grid), max(len(row) for row in grid)
    areas = _route_areas(dest)
    leads_on = {area for area, _, to_map, _, _ in _walk_exits(dest) if to_map != map_dir}
    if direction in ("up", "down"):
        arrivals = {i: (i, height - 1 if direction == "up" else 0) for i in range(width)}
    else:
        arrivals = {i: (width - 1 if direction == "left" else 0, i) for i in range(height)}
    return frozenset(
        i + offset for i, tile in arrivals.items() if tile in areas and areas[tile] not in leads_on
    )


def _walk_exits(map_dir: str) -> tuple:
    """Ways off a map on foot: (from area, go_to_exit argument, to map dir, to area, to name).

    Map edges count where the tiles on both sides are walkable (Oldale's south row is
    open but Route 101 has trees under most of it). Doors and other warps count from
    the area beside them to the area they arrive in.
    """
    from server.field_control import EDGE_NAMES
    from utils.mapping.porymap_state import _map_dirs_by_const, _readable_map_name, _static_grid, arrival_only_warps

    raw = _raw_map_json(map_dir)
    grid = _static_grid(map_dir) if raw else None
    if not grid:
        return ()
    here = _route_areas(map_dir)
    height, width = len(grid), max(len(row) for row in grid)
    out = set()
    connections = raw.get("connections") or []
    # Seen live: ROUTE 111's west side joins ROUTE 113 (top) and ROUTE 112 (lower down).
    # go_to("Route 113") said go_to_exit("west"), which took the nearest west tile into
    # ROUTE 112; from there the way back was "east", and the walk bounced between the two
    # for six minutes. Where one side has several neighbours the argument names the map,
    # which go_to_exit narrows to that connection's span.
    sides = [conn.get("direction") for conn in connections]
    for conn in connections:
        direction = conn.get("direction")
        dest = _map_dirs_by_const().get(str(conn.get("map", "")))
        dest_grid = _static_grid(dest) if dest else None
        if direction not in EDGE_NAMES or not dest_grid:
            continue
        offset = int(conn.get("offset", 0) or 0)
        dest_height, dest_width = len(dest_grid), max(len(row) for row in dest_grid)
        if direction == "up":
            pairs = [((x, 0), (x - offset, dest_height - 1)) for x in range(width)]
        elif direction == "down":
            pairs = [((x, height - 1), (x - offset, 0)) for x in range(width)]
        elif direction == "left":
            pairs = [((0, y), (dest_width - 1, y - offset)) for y in range(height)]
        else:
            pairs = [((width - 1, y), (0, y - offset)) for y in range(height)]
        there = _route_areas(dest)
        name = _readable_map_name(conn["map"])
        exit_arg = name if sides.count(direction) > 1 else EDGE_NAMES[direction]
        for mine, theirs in pairs:
            if mine in here and theirs in there:
                out.add((here[mine], exit_arg, dest, there[theirs], name))
    warps = raw.get("warp_events") or []
    # Seen live: GRANITE CAVE B1F has two ladders up to GRANITE CAVE 1F, one to the
    # entrance and one to the part with Steven's Room; go_to_exit("GRANITE CAVE 1F") took
    # the nearer one, out of the cave. Where warps to one map arrive in different parts
    # of it, the argument names the warp's tile too, as the EXITS list shows it (a
    # two-tile doormat leads to one spot, so its name alone is enough).
    found = []
    arrival_only = arrival_only_warps(map_dir)
    for i, warp in enumerate(warps):
        if i in arrival_only:
            continue  # where warps land (porymap_state.arrival_only_warps), not a way off
        dest = _map_dirs_by_const().get(str(warp.get("dest_map", "")))
        try:
            dest_warp = (_raw_map_json(dest).get("warp_events") or [])[int(warp.get("dest_warp_id"))]
        except (TypeError, ValueError, IndexError):
            continue  # dynamic warps (dest set at run time) can't be followed statically
        x, y = int(warp.get("x", -1)), int(warp.get("y", -1))
        theirs = frozenset(
            _areas_at(dest, int(dest_warp.get("x", -1)), int(dest_warp.get("y", -1)), _route_areas(dest))
        )
        found.append((x, y, dest, theirs, _readable_map_name(warp["dest_map"])))
    arrivals = {}
    for _, _, dest, theirs, _ in found:
        arrivals.setdefault(dest, set()).add(theirs)
    for x, y, dest, theirs, name in found:
        exit_arg = f"{name} ({x}, {y})" if len(arrivals[dest]) > 1 else name
        for mine in _areas_at(map_dir, x, y, here):
            for area in theirs:
                out.add((mine, exit_arg, dest, area, name))
    return tuple(sorted(out))


# Rides a go_to walk can take besides exits: the station's attendant takes the player to
# the other station (pret data/maps/Route112_CableCarStation/scripts.inc: the attendant's
# MSGBOX_YESNO, YES → special CableCarWarp). Each is {map: (the attendant's tile in the
# map's object_events, the other station)}. The attendant faces down from that tile.
RIDE_EXIT = "CABLE CAR"
CABLE_CAR_STATIONS = {
    "Route112_CableCarStation": ((6, 6), "MtChimney_CableCarStation"),
    "MtChimney_CableCarStation": ((6, 6), "Route112_CableCarStation"),
}


@lru_cache(maxsize=None)
def _ride_exits(map_dir: str) -> tuple:
    """_walk_exits' shape for a ride (RIDE_EXIT) from this map, or () if it has none.

    emerald-new (2026-11-04): from the Lavaridge Pokémon Center, go_to("Jagged Pass")
    walked into Jagged Pass's bottom (14,40), below one-way ledges, where the tall grass
    can't be reached; the way to it is the Cable Car up to Mt. Chimney.
    """
    from utils.mapping.porymap_state import _readable_map_name, _map_dirs_by_const

    ride = CABLE_CAR_STATIONS.get(map_dir)
    if ride is None:
        return ()
    (x, y), dest = ride
    const = next((c for c, d in _map_dirs_by_const().items() if d == dest), dest)
    mine = _areas_at(map_dir, x, y + 1, _route_areas(map_dir))
    theirs = _areas_at(dest, x, y + 1, _route_areas(dest))
    return tuple((m, RIDE_EXIT, dest, t, _readable_map_name(const)) for m in mine for t in theirs)


def _downhill(map_dir: str, area: int) -> set:
    """The areas of a map reachable on foot from `area`: itself and every area below a
    ledge from it (one-way jumps; _ledge_jumps)."""
    jumps = _ledge_jumps(map_dir)
    out, stack = {area}, [area]
    while stack:
        here = stack.pop()
        for low, high in jumps:
            if low == here and high not in out:
                out.add(high)
                stack.append(high)
    return out


def _squash_name(text: str) -> str:
    """'Birch's Lab' -> 'BIRCHSLAB': letters and digits only, upper case, accents dropped.

    Map names are plain ASCII (POKEMON CENTER); objectives write "PokéMart"/"Pokémon
    Center", and 'É'.isalnum() is True, so the accent has to go, not just the apostrophe.
    """
    decomposed = unicodedata.normalize("NFKD", str(text).upper())
    return "".join(ch for ch in decomposed if ch.isalnum() and not unicodedata.combining(ch))


_HALVES = {"NORTH": (1, -1), "SOUTH": (1, 1), "WEST": (0, -1), "EAST": (0, 1)}


def _is_floor_or_variant(word: str) -> bool:
    """'1F', 'B2F', '1R', 'ALT': parts of a map name that only say which floor, room or
    copy it is.

    Seen live: on METEOR FALLS 1F 1R, inside the Route 114 door, "Head west through Route
    114 to Meteor Falls" (meteor_falls_reached) never showed as done, and the model walked
    back out twice. The decomp numbers the rooms of a floor 1R, 2R (Meteor Falls, Magma
    Hideout).
    """
    if word == "ALT":
        return True
    core = word[1:] if word.startswith("B") else word
    return len(core) >= 2 and core[-1] in "FR" and core[:-1].isdigit()


# Words the objective list uses for what the game's map names call something else.
# Seen live: "Travel to Mr. Briney's cottage" had no route and never counted as reached,
# because the map is ROUTE 104 MR BRINEYS HOUSE.
_SAME_PLACE_WORDS = {"COTTAGE": "HOUSE"}


def _same_word(a: str, b: str) -> bool:
    """Equal, or one is a short form of the other ('CORP' / 'CORPORATION').

    Seen live: "Visit Devon Corporation" never counted as done inside RUSTBORO CITY DEVON
    CORP 3F. Four letters at least, so 'MT' and 'ST' don't match everything.
    """
    if a == b or _SAME_PLACE_WORDS.get(a) == b or _SAME_PLACE_WORDS.get(b) == a:
        return True
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return len(short) >= 4 and long_.startswith(short)


def _names_same_place(location_name: str, target: str) -> bool:
    """True when a loosely written target names this map.

    Seen live: "Enter Rusturf Tunnel" never counted as done on RUSTURF TUNNEL ALT, and
    the objective list writes "Dewford Gym" for DEWFORD TOWN GYM and "Granite Cave" for
    GRANITE CAVE 1F. The target's words must appear in the map name in order, and the
    map name, without a floor or ALT suffix, must end with the target's last word. So
    "Rustboro City" does not match RUSTBORO CITY GYM.
    """
    have = _place_words(location_name)
    while have and _is_floor_or_variant(have[-1]):
        have = have[:-1]
    want = _place_words(target)
    if not have or not want or not _same_word(have[-1], want[-1]):
        return False
    at = 0
    for word in want:
        while at < len(have) and not _same_word(have[at], word):
            at += 1
        if at == len(have):
            return False
        at += 1
    return True


def arrived_at(location_name: str, position: Optional[Tuple[int, int]], target: str) -> bool:
    """True when the player is on the map an objective's target_location names.

    The target is matched against the end of the map name, so "Birch's Lab" matches
    LITTLEROOT TOWN PROFESSOR BIRCHS LAB. A trailing North/South/East/West ("Route 104
    South") names part of a route: it holds when the walkable area the player is in
    lies clearly on that side of the map. Route 104's halves are separate areas joined
    only through Petalburg Woods; a map that is one open area claims neither half.
    """
    from utils.state_formatter import ROM_TO_PORYMAP_MAP

    words = str(target).split()
    half = _HALVES.get(words[-1].upper()) if len(words) > 1 else None
    wanted = _squash_name(" ".join(words[:-1] if half else words))
    here = _squash_name(location_name)
    if not wanted:
        return False
    if not here.endswith(wanted) and not _names_same_place(location_name, " ".join(words[:-1] if half else words)):
        return False
    if half is None:
        return True
    map_dir = ROM_TO_PORYMAP_MAP.get(location_name, "")
    areas = _walk_areas(map_dir)
    mine = _areas_at(map_dir, *position) if position is not None else set()
    if len(mine) != 1 or not areas:
        return False
    (number,) = mine
    axis, sign = half
    cells = [cell[axis] for cell, a in areas.items() if a == number]
    span = max(cell[axis] for cell in areas) + 1
    # Signed distance of the area's centre from the map's middle, as a share of the map.
    lean = (sum(cells) / len(cells) - span / 2) / span
    return lean * sign > 0.1


_FAR_MAP_MAX_STEPS = 16


def _far_map_hint(map_dir: str, matches, position: Optional[Tuple[int, int]] = None, nearest: bool = False) -> str:
    """The first go_to_exit toward a map `matches(readable map name)` accepts.

    nearest=True keeps only the single nearest match: right for one destination (an
    objective's target), where several maps can match the name. Seen live: "Rustboro City
    Pokemon Center" matched 1F and 2F and the farthest-match rule sent the AI upstairs,
    away from the nurse.

    Seen live: in Littleroot the model wrote "travel west through Route 102 to Petalburg
    City" and walked one tile at a time around Littleroot for twenty minutes. Route 102
    is three maps away (north via Route 101 and Oldale). A breadth-first walk over the
    maps' walkable areas, joined by edges and doors, finds the shortest way on foot, so
    a route never runs across the sea or through the trees between two halves of a
    route. Destinations whose ways start through the same exit share one line, naming
    the farthest so the whole way is shown.
    """
    by_first_exit = {}
    for path in _walk_paths(map_dir, matches, position):
        if nearest:
            by_first_exit = {path[0][0]: path}  # paths arrive nearest first
            break
        # Paths arrive nearest first, so a later one with the same first exit is farther.
        by_first_exit[path[0][0]] = path
    lines = []
    for exit_arg, path in by_first_exit.items():
        if len(path) == 1:
            lines.append(f'{path[0][1]} is through an exit of this map: go_to_exit("{exit_arg}").')
            continue
        way = " → ".join(name for _, name in path)
        lines.append(
            f"{path[-1][1]} is not next to this map; the way on foot is {way}. "
            f'go_to_exit("{exit_arg}") takes the first step.'
        )
    return " ".join(lines)


def _walk_paths(
    map_dir: str, matches, position: Optional[Tuple[int, int]] = None, other_parts_of_here: bool = False,
    rides: bool = False, goal: Optional[frozenset] = None,
) -> List[List[Tuple[str, str]]]:
    """Ways on foot to every map `matches(readable map name)` accepts, nearest first.

    Each way is [(go_to_exit argument, name of the map entered), ...]; "nearest" counts
    maps walked through (a breadth-first walk over walkable areas joined by edges and
    doors, starting from the player's area).

    other_parts_of_here: the player's own map counts too, entered again into another
    area (from ROUTE 117's walled-in notch at (0,6), Route 117's grass is west to
    VERDANTURF TOWN and back east).

    rides: the Cable Car counts as a step too (RIDE_EXIT, which only go_to can take, so
    hints for the model leave it out), and every way into a matching map is weighed: one
    whose part of the map another way in reaches as well, and more, is dropped. Seen
    live: go_to("Jagged Pass") from Lavaridge entered at the bottom, below one-way ledges;
    from the top (Mt. Chimney, by Cable Car) the ledges lead everywhere, the bottom too.

    goal: (map dir, area) nodes to reach instead of maps `matches` accepts (tile_route);
    the ways to the nearest of them, [] among them when the player's area is one.
    """
    if not map_dir:
        return []
    areas = _route_areas(map_dir)
    start = set()
    if position is not None:
        start = {(map_dir, a) for a in _areas_at(map_dir, *position, areas)}
    if not start:
        start = {(map_dir, a) for a in set(areas.values())}
    # parent[node] = (previous node, go_to_exit argument, name of the map entered); a
    # ledge jump has no argument (the walk to the next exit jumps it on the way).
    parent = {node: None for node in start}

    def with_jumps(nodes):
        """The nodes plus every area below a ledge from them, on the same map."""
        out, stack = list(nodes), list(nodes)
        while stack:
            node = stack.pop()
            for low, high in _ledge_jumps(node[0]):
                below = (node[0], high)
                if low == node[1] and below not in parent:
                    parent[below] = (node, None, None)
                    out.append(below)
                    stack.append(below)
        return out

    frontier, hits, seen_maps = with_jumps(start), [], set() if other_parts_of_here else {map_dir}
    if goal is not None:
        hits = [node for node in frontier if node in goal]
    for _ in range(_FAR_MAP_MAX_STEPS):
        if hits and goal is not None:
            break  # the goal's nearest areas: no way through more exits is wanted
        nxt = []
        for node in frontier:
            for from_area, exit_arg, dest, dest_area, name in _walk_exits(node[0]) + (_ride_exits(node[0]) if rides else ()):
                target = (dest, dest_area)
                if from_area != node[1] or target in parent:
                    continue
                parent[target] = (node, exit_arg, name)
                nxt.append(target)
                if goal is not None or (dest in seen_maps and not (rides and dest != map_dir)):
                    continue
                seen_maps.add(dest)
                if matches(name):
                    hits.append(target)
        frontier = with_jumps(nxt)
        if goal is not None:
            hits = [node for node in frontier if node in goal]
        if not frontier:
            break
    if rides:
        reach = {hit: _downhill(*hit) for hit in hits}
        hits = [
            hit for hit in hits
            if not any(o[0] == hit[0] and reach[o] > reach[hit] for o in hits)
        ]
    paths = []
    for hit in hits:
        path, node = [], hit
        while parent[node] is not None:
            previous, exit_arg, name = parent[node]
            if exit_arg is not None:
                path.append((exit_arg, name))
            node = previous
        path.reverse()
        paths.append(path)
    if position is not None:
        # Equally many maps away, the nearer first exit wins. Seen live: at ROUTE 110
        # (16,92) the way to heal went north to MAUVILLE (about 90 tiles) instead of south
        # to SLATEPORT (7), only because the search met Mauville first.
        paths.sort(key=lambda p: (len(p), _exit_distance(map_dir, position, p[0][0]) if p else 0))
    return paths


def tile_route(
    map_dir: str, position: Optional[Tuple[int, int]], target_tiles, here_dir: Optional[str] = None
) -> Optional[List[Tuple[str, str]]]:
    """[(go_to_exit argument, name of the map entered), ...]: the way through warps, edges
    and ledge jumps from the player at position (on here_dir, default map_dir) to a part
    of map_dir holding any of target_tiles (map_dir's coordinates). [] when the player's
    part already holds one (a ledge jump down counts), None when none can be reached.

    Seen live: in LAVARIDGE TOWN GYM 1F, talk_to(13,9) (Flannery) answered "No path found
    from (9, 9) to (12, 9)": her side of the floor is reached only by dropping through
    holes to B1F and riding geysers back up. PETALBURG CITY GYM's rooms are joined by
    doors that warp within the same map.
    """
    if not map_dir:
        return None
    areas = _route_areas(map_dir)
    goal = frozenset((map_dir, areas[tuple(t)]) for t in target_tiles if tuple(t) in areas)
    if not goal:
        return None
    paths = _walk_paths(here_dir or map_dir, lambda name: False, position, goal=goal)
    return paths[0] if paths else None


def map_coord_offset(map_dir: str, badge_count: int) -> Tuple[int, int]:
    """(offset_x, offset_y) between map_dir's coordinates and the ones the walking tools
    use there (local = map - offset): non-zero only on an override map that is a cut-out
    of the map (ascii_map_loader PetalburgCity_Gym_Lobby, before the 4th badge)."""
    from utils.mapping.ascii_map_loader import get_effective_map_name, get_override

    override = get_override(get_effective_map_name(map_dir, badge_count=badge_count)) or {}
    return int(override.get("offset_x", 0)), int(override.get("offset_y", 0))


def _exit_distance(map_dir: str, position: Tuple[int, int], exit_arg: str) -> int:
    """Tiles (straight-line steps) from position to an exit of this map; 0 if unknown."""
    from server.field_control import EDGE_NAMES
    from utils.mapping.porymap_state import _readable_map_name, _static_grid

    x, y = position
    grid = _static_grid(map_dir) or ()
    edges = {name: direction for direction, name in EDGE_NAMES.items()}
    # An edge named by its map (a side with several neighbours, _walk_exits).
    for conn in _raw_map_json(map_dir).get("connections") or []:
        if conn.get("direction") in EDGE_NAMES:
            edges.setdefault(_readable_map_name(str(conn.get("map", ""))), conn["direction"])
    if exit_arg in edges and grid:
        height, width = len(grid), max(len(row) for row in grid)
        return {"up": y, "down": height - 1 - y, "left": x, "right": width - 1 - x}[edges[exit_arg]]
    spots = []
    for warp in _raw_map_json(map_dir).get("warp_events") or []:
        wx, wy = int(warp.get("x", -1)), int(warp.get("y", -1))
        name = _readable_map_name(warp.get("dest_map", ""))
        if exit_arg in (name, f"{name} ({wx}, {wy})"):
            spots.append(abs(wx - x) + abs(wy - y))
    return min(spots, default=0)


POKEMON_CENTER_1F = "POKEMON CENTER 1F"


def pokemon_center_route(location_name: str, position: Optional[Tuple[int, int]]) -> List[Tuple[str, str]]:
    """[(go_to_exit argument, map entered), ...] to the nearest Pokémon Center 1F on foot;
    [] when inside one already or none can be reached."""
    from utils.state_formatter import ROM_TO_PORYMAP_MAP

    target = _squash_name(POKEMON_CENTER_1F)
    if _squash_name(location_name).endswith(target):
        return []
    paths = _walk_paths(
        ROM_TO_PORYMAP_MAP.get(location_name, ""), lambda name: _squash_name(name).endswith(target), position
    )
    return paths[0] if paths else []


def way_to_pokemon_center(location_name: str, position: Optional[Tuple[int, int]]) -> str:
    """Where to heal: "" when already inside a Pokémon Center 1F (the nurse's floor),
    else the way on foot to the nearest one, or a note that none can be reached.

    Map names end in "POKEMON CENTER 1F" (RUSTBORO CITY POKEMON CENTER 1F); a 2F's way
    down the stairs is found the same way as any other.
    """
    if _squash_name(location_name).endswith(_squash_name(POKEMON_CENTER_1F)):
        return ""
    path = pokemon_center_route(location_name, position)
    if not path:
        return "no Pokémon Center can be reached on foot from here: go back the way you came."
    # heal_at_pokecenter walks the way itself (server _walk_to_pokemon_center); the way
    # is named so the model knows where it is going.
    way = " → ".join(name for _, name in path)
    return f"call heal_at_pokecenter(): it walks to {path[-1][1]} ({way}) and heals."


# ---------------------------------------------------------------------------
# Wild encounter tiles (train)
# ---------------------------------------------------------------------------

# Tiles a step can roll a wild Pokémon on, by metatile behavior (pokeemerald
# sTileBitAttributes, TILE_FLAG_HAS_ENCOUNTERS): the grass family anywhere, and the cave
# floor on underground maps. Sand and the Abandoned Ship's indoor tiles carry the flag
# too, but beaches and the ship's corridors have no land Pokémon, and a step loop there
# would wait out its whole budget for nothing.
WILD_WALKABLE = (".", "~")
_UNDERGROUND = "MAP_TYPE_UNDERGROUND"


def is_underground(map_dir: str) -> bool:
    return _raw_map_json(map_dir).get("map_type") == _UNDERGROUND if map_dir else False


def encounter_tiles(grid, raw_tiles, underground: bool) -> set:
    """(x, y) of every walkable tile where stepping can start a wild battle.

    raw_tiles are porymap's (tile id, behavior, collision, elevation) per cell; without
    them (an ASCII override map) only '~' counts.
    """
    from pokemon_env.enums import MetatileBehavior

    behaviors = {
        int(MetatileBehavior.TALL_GRASS),
        int(MetatileBehavior.LONG_GRASS),
        int(MetatileBehavior.LONG_GRASS_SOUTH_EDGE),
        int(MetatileBehavior.ASHGRASS),
    }
    if underground:
        behaviors.add(int(MetatileBehavior.CAVE))
    tiles = set()
    for y, row in enumerate(grid or []):
        raw_row = raw_tiles[y] if raw_tiles and y < len(raw_tiles) else None
        for x, cell in enumerate(row):
            if cell not in WILD_WALKABLE:
                continue
            raw = raw_row[x] if raw_row is not None and x < len(raw_row) else None
            if raw is not None and len(raw) > 1 and raw[1] is not None:
                hit = int(raw[1]) in behaviors
            else:
                hit = cell == "~"
            if hit:
                tiles.add((x, y))
    return tiles


def _neighbours(tile: Tuple[int, int]) -> List[Tuple[int, int]]:
    x, y = tile
    return [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]


def encounter_pairs(
    tiles: set, start: Tuple[int, int], grid, limit: int = 6, occupied: frozenset = frozenset()
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Up to `limit` pairs of side-by-side encounter tiles, nearest to `start` first.

    Stepping back and forth between two encounter tiles rolls for a battle on every
    step: a plain tile rolls nothing, and a step onto the patch from one skips 40% of
    its rolls (pokeemerald StandardWildEncounter, AllowWildCheckOnNewMetatile, when the
    behavior changes). Distance is a breadth-first walk over '.' and '~'
    tiles; pairs it can't reach (behind a ledge, say) come after, by straight-line
    distance, for the pathfinder to try.

    `occupied`: tiles people stand on now. Seen live: on ROUTE 117 someone stood in the
    one-tile gap at (47,4) to the grass north of the wall; all 6 pairs were up there, and
    every train call stopped in 2 s while a big patch south of the wall went unused.
    """
    paired = {t for t in tiles if any(n in tiles for n in _neighbours(t))}
    if not paired:
        return []
    from collections import deque

    seen = {start}
    reached: List[Tuple[int, int]] = []
    queue = deque([start])
    while queue:
        tile = queue.popleft()
        if tile in paired:
            reached.append(tile)
            if len(reached) >= limit:
                break
        for nxt in _neighbours(tile):
            x, y = nxt
            if nxt in seen or not (0 <= y < len(grid) and 0 <= x < len(grid[y])) or grid[y][x] not in WILD_WALKABLE:
                continue
            if nxt in occupied:
                continue
            seen.add(nxt)
            queue.append(nxt)
    rest = sorted(paired - set(reached), key=lambda t: (abs(t[0] - start[0]) + abs(t[1] - start[1]), t))
    pairs = []
    for tile in (reached + rest)[:limit]:
        other = min((n for n in _neighbours(tile) if n in tiles), key=lambda n: (n not in seen, n))
        pairs.append((tile, other))
    return pairs


# wild_encounter_spots: pairs handed to train, and how many candidates it may path-check.
ENCOUNTER_PAIRS = 6
ENCOUNTER_CANDIDATES = 40


def wild_encounter_spots(env) -> dict:
    """The nearest encounter-tile pairs on the player's map, in navigate_to's frame.

    {"pairs": [((x, y), (x, y)), ...], "coord_offset": (ox, oy) or None, "map_dir": str};
    "pairs" is empty when the map has no tall grass and isn't a cave.
    """
    state, coord_offset, source = load_pathfinding_state(env)
    porymap = (state.get("map") or {}).get("porymap") or {}
    grid = porymap.get("grid") or []
    position = (state.get("player") or {}).get("position") or {}
    ox, oy = coord_offset or (0, 0)
    start = (int(position.get("x", 0)) - ox, int(position.get("y", 0)) - oy)
    map_dir = source.get("porymap_map_name", "")
    tiles = encounter_tiles(grid, porymap.get("raw_tiles"), is_underground(map_dir))
    occupied = frozenset(
        (int(o["current_x"]) - ox, int(o["current_y"]) - oy) for o in porymap.get("runtime_objects") or []
    ) - {start}
    # The grid walk in encounter_pairs knows neither ledges nor elevation, and lists
    # pairs it can't reach last "for the pathfinder to try". Seen live from ROUTE 117
    # (0,6), a walled-in notch: all 6 were unreachable and every train call stopped in
    # 0.2 s ("No path found from (0, 6) to (11, 1)"). Keep only pairs the pathfinder
    # reaches; "unreachable" says grass exists but none of it can be walked to.
    from server.field_control import end_of_path

    candidates = encounter_pairs(tiles, start, grid, limit=ENCOUNTER_CANDIDATES, occupied=occupied)
    pairs = []
    for tile, other in candidates:
        if tile != start:
            path = pathfinder.find_path(start, tile, state, consider_npcs=True, allow_partial=False)
            if not path or end_of_path(start, path) != tile:
                continue
        pairs.append((tile, other))
        if len(pairs) == ENCOUNTER_PAIRS:
            break
    return {
        "pairs": pairs,
        "unreachable": bool(candidates) and not pairs,
        "coord_offset": coord_offset,
        "map_dir": map_dir,
    }


@lru_cache(maxsize=None)
def map_has_wild_encounters(map_dir: str) -> bool:
    """Whether a map (data/maps directory) has tall grass or is a cave, from the decomp."""
    from pokemon_env.porymap_paths import get_porymap_root
    from utils.mapping.porymap_json_builder import build_json_map

    root = get_porymap_root()
    if not root or not map_dir:
        return False
    try:
        built = build_json_map(map_dir, root, include_grid=True, include_ascii=False) or {}
    except Exception as e:  # a broken map must not take the hint down with it
        logger.warning(f"Could not read {map_dir} for wild encounters: {e}")
        return False
    return bool(encounter_tiles(built.get("grid"), built.get("raw_tiles"), is_underground(map_dir)))


@lru_cache(maxsize=1)
def _map_dirs_by_name() -> dict:
    """'ROUTE 116' -> 'Route116', for every map the decomp has."""
    from utils.mapping.porymap_state import _map_dirs_by_const, _readable_map_name

    return {_readable_map_name(const): map_dir for const, map_dir in _map_dirs_by_const().items()}


def way_to_wild_encounters(location_name: str, position: Optional[Tuple[int, int]]) -> str:
    """The way on foot to the nearest map with tall grass or a cave floor."""
    from utils.state_formatter import ROM_TO_PORYMAP_MAP

    paths = _walk_paths(
        ROM_TO_PORYMAP_MAP.get(location_name, ""),
        lambda name: map_has_wild_encounters(_map_dirs_by_name().get(name, "")),
        position,
        other_parts_of_here=True,
    )
    if not paths:
        return "no tall grass or cave can be reached on foot from here."
    # Equally near, the player's own map wins: its Pokémon are the ones an objective
    # there wants (Marill on Route 117, not Rusturf Tunnel's Whismur).
    path = min(paths, key=lambda p: (len(p), p[-1][1] != location_name))
    place = path[-1][1]
    if len(path) == 1:
        return f'{place} has wild Pokémon: go_to_exit("{path[0][0]}"), then call train again.'
    way = " → ".join(name for _, name in path)
    return (
        f"the nearest wild Pokémon are on {place}; the way on foot: {way}. "
        f'go_to_exit("{path[0][0]}") takes the first step; call train again there.'
    )


# Places the walkthrough calls by the name on their sign, not their map name.
# Seen live: "Get free Bicycle from Rydel's Cycles" inside MAUVILLE CITY BIKE SHOP; the
# model walked out to look for "Rydel's Cycles" five times.
# Seen live: go_to("Mauville City Poke Mart") said it "can't be reached on foot"; the
# map is MAUVILLE CITY MART, and the objectives write "Oldale Poke Mart", "Rustboro Poke
# Mart"... The game's own maps say MART and POKEMON CENTER, however the sign is spelled.
_PLACE_ALIASES = {
    ("RYDELS", "CYCLES"): ["BIKE", "SHOP"],
    ("POKE", "MART"): ["MART"],
    ("POKEMON", "MART"): ["MART"],
    ("POKEMART",): ["MART"],
    ("POKE", "CENTER"): ["POKEMON", "CENTER"],
    ("POKECENTER",): ["POKEMON", "CENTER"],
    ("POKEMONCENTER",): ["POKEMON", "CENTER"],
    # Seen live: "Return to Route 112 Cable Car Station" (target "Route 112 Cable Car")
    # never showed as done inside ROUTE 112 CABLE CAR STATION. The longer form first
    # (aliases go longest first), so the map's own name isn't given STATION twice.
    ("CABLE", "CAR", "STATION"): ["CABLE", "CAR", "STATION"],
    ("CABLE", "CAR"): ["CABLE", "CAR", "STATION"],
}


def _place_words(text: str) -> List[str]:
    """'Birch's Lab' -> ['BIRCHS', 'LAB']: upper-case words, letters and digits only,
    with every _PLACE_ALIASES phrase replaced by the words the map names use."""
    words = [w for w in (_squash_name(word) for word in str(text).split()) if w]
    out, at = [], 0
    while at < len(words):
        for alias, words_instead in sorted(_PLACE_ALIASES.items(), key=lambda item: -len(item[0])):
            if tuple(words[at:at + len(alias)]) == alias:
                out.extend(words_instead)
                at += len(alias)
                break
        else:
            out.append(words[at])
            at += 1
    return out


def way_to_place(location_name: str, position: Optional[Tuple[int, int]], target: str) -> str:
    """How to walk to the map an objective's target_location names, or "" if unknown or here.

    Seen live: objective "Defeat Gym trainers" with target "Rustboro Gym". The model went
    in, came out, and then walked Rustboro one tile at a time "to find trainers"; the
    door at (27,19) -> RUSTBORO CITY GYM was in its exit list. Targets name maps loosely,
    so a map matches when every word of the target appears in its name in order ("Rustboro
    Gym" ~ RUSTBORO CITY GYM, "Birch's Lab" ~ LITTLEROOT TOWN PROFESSOR BIRCHS LAB). A
    trailing North/South/East/West is dropped here; arrived_at handles route halves.
    """
    from utils.state_formatter import ROM_TO_PORYMAP_MAP

    if arrived_at(location_name, position, target):
        return ""
    words = _place_words(target)
    if len(words) > 1 and words[-1] in _HALVES:
        words = words[:-1]
    if not words:
        return ""

    def matches(name: str) -> bool:
        have, at = _place_words(name), 0
        for word in words:
            while at < len(have) and not _same_word(have[at], word):
                at += 1
            if at == len(have):
                return False
            at += 1
        return True

    if matches(location_name):
        return ""  # on the named map already (e.g. the other half of a route)
    map_dir = ROM_TO_PORYMAP_MAP.get(location_name, "")
    # A map named exactly as the target wins. Seen live: "Route 116" matched ROUTE 116
    # TUNNELERS REST HOUSE too, and the hint (which shows the farthest match per first
    # exit) routed into that house instead of onto the route.
    exact = _far_map_hint(map_dir, lambda name: _place_words(name) == words, position, nearest=True)
    return exact or _far_map_hint(map_dir, matches, position, nearest=True)


def place_route(location_name: str, position: Optional[Tuple[int, int]], target: str) -> Optional[List[Tuple[str, str]]]:
    """[(go_to_exit argument, map entered), ...] to the map `target` names, on foot.

    [] when already there (arrived_at), None when no map matches or none can be reached.
    The same matching as way_to_place: a map named exactly as the target first (so from
    ROUTE 117 POKEMON DAY CARE, "Route 117" means the route, not the house), else the
    nearest map whose name holds the target's words in order.
    """
    from utils.state_formatter import ROM_TO_PORYMAP_MAP

    if arrived_at(location_name, position, target):
        return []
    words = _place_words(target)
    if len(words) > 1 and words[-1] in _HALVES:
        words = words[:-1]
    if not words:
        return None

    def matches(name: str) -> bool:
        have, at = _place_words(name), 0
        for word in words:
            while at < len(have) and not _same_word(have[at], word):
                at += 1
            if at == len(have):
                return False
            at += 1
        return True

    map_dir = ROM_TO_PORYMAP_MAP.get(location_name, "")
    for matcher in (lambda name: _place_words(name) == words, matches):
        # go_to walks this, so the Cable Car is a step it can take (RIDE_EXIT).
        paths = _walk_paths(map_dir, matcher, position, rides=True)
        if paths:
            return paths[0]
    return None


def _live_tile_reader(env):
    """(x, y) map coordinates -> the metatile in the game's map buffer now, or None
    (utils/mapping/dynamic_map_overlay: the buffer has a 7-tile border)."""
    from utils.mapping.dynamic_map_overlay import GBA_MAP_BORDER_OFFSET

    reader = getattr(env, "memory_reader", None)
    if reader is None:
        return None

    def tile(x: int, y: int):
        if not getattr(reader, "_map_buffer_addr", None) and not reader._find_map_buffer_addresses():
            return None
        rows = reader.read_map_metatiles(
            x_start=x + GBA_MAP_BORDER_OFFSET, y_start=y + GBA_MAP_BORDER_OFFSET, width=1, height=1
        )
        return rows[0][0] if rows and rows[0] else None

    return tile


def plan_exit_route(env, exit_query: str, max_edge_candidates: int = 16, avoid: Tuple = ()) -> dict:
    """Resolve an EXITS entry to a walkable route (caller holds the emulator lock).

    Edge exits: the reachable standable tile on that edge nearest the player, then one
    more press off the edge. Warps: the warp tile itself (doors and stairs fire when
    stepped on; the caller handles arrow warps that fire when walked off). Sign doors
    (kind "sign_door"): the nearest tile in front of the door, and "face", the press
    that turns to it. Every warp plan has "warp_tile", and "arrival_here": the tile it
    lands on when it leads to another spot of this same map, else None.

    avoid: map tiles (raw map coordinates) to treat as blocked, e.g. where a person
    stopped the last walk.
    """
    from server.field_control import (
        DIRECTION_DELTAS,
        EDGE_BUTTON,
        edge_candidates,
        end_of_path,
        match_exits,
    )

    state, coord_offset, source = load_pathfinding_state(env)
    location = state.get("player", {}).get("location", "Unknown")
    json_map = source.get("json_map")
    grid = state.get("map", {}).get("porymap", {}).get("grid")
    if not json_map or not grid:
        return {"success": False, "error": f"No map data for {location}; walk with press_buttons instead."}

    exits = list_map_exits(json_map, source.get("porymap_map_name"), coord_offset, _live_tile_reader(env))
    if not exits:
        return {"success": False, "error": f"{location} has no exits in the map data."}
    matches = match_exits(exits, exit_query)
    if not matches:
        options = "; ".join(e["label"] for e in exits)
        return {
            "success": False,
            "error": f"No exit of {location} matches '{exit_query}'. Exits here: {options}",
        }

    pos = state.get("player", {}).get("position", {})
    ox, oy = coord_offset or (0, 0)
    start = (int(pos.get("x", 0)) - ox, int(pos.get("y", 0)) - oy)
    this_map = _raw_map_json(source.get("porymap_map_name") or "").get("id")

    blocked = [(int(x) - ox, int(y) - oy) for x, y in avoid] or None

    def route_to(tile):
        if tile == start:
            return []
        buttons = pathfinder.find_path(
            start, tile, state, consider_npcs=True, allow_partial=False, blocked_coords=blocked
        )
        if buttons and end_of_path(start, buttons) == tile:
            return buttons
        return None

    best = None
    tried = []
    for exit_entry in matches:
        if exit_entry["kind"] == "edge":
            candidates = edge_candidates(grid, exit_entry["direction"])
            span = exit_entry.get("open_span")
            if span is not None:
                # Only where the neighbouring map is walkable too (span is in full-map coords).
                along, shift = (0, ox) if exit_entry["direction"] in ("up", "down") else (1, oy)
                candidates = [c for c in candidates if c[along] + shift in span]
            # Tiles that arrive in a dead-end pocket next door go last (_edge_dead_ends).
            along, shift = (0, ox) if exit_entry["direction"] in ("up", "down") else (1, oy)
            dead = exit_entry.get("dead_ends") or frozenset()
            candidates.sort(
                key=lambda c: (c[along] + shift in dead, abs(c[0] - start[0]) + abs(c[1] - start[1]))
            )
            if not candidates:
                tried.append(f"{exit_entry['label']}: no standable tile on that edge (water or wall)")
                continue
            for tile in candidates[:max_edge_candidates]:
                buttons = route_to(tile)
                if buttons is not None:
                    if best is None or len(buttons) < len(best["buttons"]):
                        best = {
                            "kind": "edge",
                            "label": exit_entry["label"],
                            "target": tile,
                            "buttons": buttons,
                            "final_button": EDGE_BUTTON[exit_entry["direction"]],
                        }
                    break
            else:
                tried.append(f"{exit_entry['label']}: no walkable path to that edge from {start}")
        else:
            tile = (exit_entry["x"], exit_entry["y"])
            # A warp to another spot of this map: the map id won't change, so the caller
            # judges it by where the player ends up (came_through_warp), in plan coordinates.
            arrival = exit_entry.get("arrival")
            here = arrival is not None and exit_entry.get("dest_map") == this_map
            way = {
                "warp_tile": tile,
                "arrival_here": (arrival[0] - ox, arrival[1] - oy) if here else None,
            }
            if exit_entry.get("sign_door"):
                # Read from a tile beside it, facing it (porymap_state.sign_door_warps).
                fronts = []
                for button, (dx, dy) in DIRECTION_DELTAS.items():
                    front = (tile[0] - dx, tile[1] - dy)
                    buttons = route_to(front)
                    if buttons is not None:
                        fronts.append((len(buttons), front, button, buttons))
                if not fronts:
                    tried.append(f"{exit_entry['label']}: no walkable path to a tile in front of the door at {tile}")
                    continue
                _, front, button, buttons = min(fronts)
                way.update({"kind": "sign_door", "target": front, "face": button})
            else:
                buttons = route_to(tile)
                if buttons is None:
                    tried.append(f"{exit_entry['label']}: no walkable path to the warp at {tile}")
                    continue
                way.update({"kind": "warp", "target": tile})
            if best is None or len(buttons) < len(best["buttons"]):
                best = {**way, "label": exit_entry["label"], "buttons": buttons, "final_button": None}

    if best is None:
        return {
            "success": False,
            "error": (
                f"Cannot reach '{exit_query}' from ({pos.get('x')}, {pos.get('y')}) on {location}: "
                + "; ".join(tried)
                + ". A ledge (one-way), water (needs Surf) or a person may be in the way."
            ),
        }
    best.update({
        "success": True,
        "grid": grid,
        "raw_tiles": state.get("map", {}).get("porymap", {}).get("raw_tiles"),
        "coord_offset": coord_offset,
        "location": location,
    })
    return best


# ---------------------------------------------------------------------------
# Memory (long-term) helpers — formerly "knowledge base"
# ---------------------------------------------------------------------------

def add_memory_direct(category=None, title="", content="", location=None, coordinates=None, importance=3, path=None) -> dict:
    """Add an entry to long-term memory."""
    try:
        title_s = (title or "").strip()
        content_s = (content or "").strip()
        if not title_s and not content_s:
            return {
                "success": False,
                "error": "title and content cannot both be empty (refusing useless memory rows)",
            }
        effective_path = path or category or "uncategorized"
        entry_id = memory_store.add(
            path=effective_path,
            title=title_s,
            content=content_s,
            location=location,
            coordinates=coordinates,
            importance=importance,
        )
        logger.info(f"📝 Added memory: {title_s} ({effective_path})")
        return {"success": True, "entry_id": entry_id, "message": f"Stored memory: {title_s}"}
    except Exception as e:
        logger.error(f"Failed to add memory: {e}")
        return {"success": False, "error": str(e)}


def search_memory_direct(category=None, query="", location="", min_importance=1, path=None) -> dict:
    """Search long-term memory."""
    try:
        results = memory_store.search(
            path=path, category=category, location=location or None, query=query or None, min_importance=min_importance
        )
        return {"success": True, "count": len(results), "results": results}
    except Exception as e:
        logger.error(f"Failed to search memory: {e}")
        return {"success": False, "error": str(e)}


def get_memory_summary_direct(min_importance=3) -> dict:
    """Get long-term memory summary (legacy content dump)."""
    try:
        summary = memory_store.get_summary(min_importance=min_importance)
        return {"success": True, "summary": summary}
    except Exception as e:
        logger.error(f"Failed to get memory summary: {e}")
        return {"success": False, "error": str(e)}


def get_memory_overview_direct() -> dict:
    """Get compact tree overview of long-term memory ([id] title grouped by path)."""
    try:
        overview = memory_store.get_tree_overview()
        return {"success": True, "overview": overview}
    except Exception as e:
        logger.error(f"Failed to get memory overview: {e}")
        return {"success": False, "error": str(e)}


def _normalize_process_reasoning(reasoning: object) -> Optional[str]:
    """Return stripped non-empty reasoning string, or None if missing/invalid."""
    if reasoning is None:
        return None
    text = str(reasoning).strip()
    return text if text else None


def _validate_process_entries_list(entries: object) -> Optional[str]:
    """Return an error message if *entries* is not a non-empty list."""
    if not isinstance(entries, list):
        return "entries must be a list"
    if len(entries) == 0:
        return "entries must be non-empty (provide at least one entry object for this action)"
    return None


def _batch_all_succeeded(results: list) -> bool:
    """True only if every per-entry result reports success (batch semantics)."""
    return bool(results) and all(r.get("success") for r in results)


def process_memory_direct(action: str, entries: list, reasoning: object) -> dict:
    """Unified CRUD dispatch for long-term memory.

    ``action`` is one of: read, add, update, delete.
    ``entries`` is a list of per-entry dicts with action-specific fields.
    ``reasoning`` is required (non-empty): why this memory operation is appropriate.
    """
    try:
        if _normalize_process_reasoning(reasoning) is None:
            return {
                "success": False,
                "error": "reasoning is required (non-empty string explaining why this memory operation is needed)",
            }
        ent_err = _validate_process_entries_list(entries)
        if ent_err:
            return {"success": False, "error": ent_err, "results": []}
        results = []
        for entry_data in entries:
            if action == "read":
                entry_id = entry_data.get("id")
                if not entry_id:
                    results.append({"success": False, "error": "Missing 'id' for read"})
                    continue
                entry = memory_store.get(entry_id)
                if entry is None:
                    results.append({"success": False, "error": f"Entry {entry_id} not found"})
                else:
                    results.append({"success": True, "entry": memory_store.to_display_dict(entry)})

            elif action == "add":
                path = entry_data.get("path", entry_data.get("category", "uncategorized"))
                title = (entry_data.get("title") or "").strip()
                content = (entry_data.get("content") or "").strip()
                if not title and not content:
                    results.append(
                        {
                            "success": False,
                            "error": "add requires non-empty title and/or content (empty entries create useless memory rows)",
                        }
                    )
                    continue
                importance = int(entry_data.get("importance", 3))
                location = entry_data.get("location")
                coordinates = entry_data.get("coordinates")
                if isinstance(coordinates, str) and "," in coordinates:
                    parts = coordinates.split(",")
                    try:
                        coordinates = (int(parts[0].strip()), int(parts[1].strip()))
                    except (ValueError, IndexError):
                        coordinates = None
                add_kwargs = dict(
                    path=path, title=title or "", content=content or "",
                    importance=importance, location=location,
                    coordinates=coordinates,
                )
                custom_id = entry_data.get("id", "")
                if custom_id and isinstance(custom_id, str) and custom_id.strip():
                    add_kwargs["id"] = custom_id.strip()
                entry_id = memory_store.add(**add_kwargs)
                results.append({"success": True, "entry_id": entry_id})

            elif action == "update":
                entry_id = entry_data.get("id")
                if not entry_id:
                    results.append({"success": False, "error": "Missing 'id' for update"})
                    continue
                update_fields = {k: v for k, v in entry_data.items() if k != "id" and v is not None}
                ok = memory_store.update(entry_id, **update_fields)
                results.append({"success": ok, "entry_id": entry_id})

            elif action == "delete":
                entry_id = entry_data.get("id")
                if not entry_id:
                    results.append({"success": False, "error": "Missing 'id' for delete"})
                    continue
                ok = memory_store.remove(entry_id)
                results.append({"success": ok, "entry_id": entry_id})

            else:
                results.append({"success": False, "error": f"Unknown action: {action}"})

        return {"success": _batch_all_succeeded(results), "results": results}
    except Exception as e:
        logger.error(f"process_memory error: {e}")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Skill library helpers
# ---------------------------------------------------------------------------

def get_skill_overview_direct() -> dict:
    """Get compact tree overview of the skill library ([id] name grouped by path)."""
    try:
        overview = skill_store.get_tree_overview()
        return {"success": True, "overview": overview}
    except Exception as e:
        logger.error(f"Failed to get skill overview: {e}")
        return {"success": False, "error": str(e)}


def process_skill_direct(action: str, entries: list, reasoning: object) -> dict:
    """Unified CRUD dispatch for the skill library.

    ``reasoning`` is required (non-empty): why this skill operation is appropriate.
    """
    try:
        if _normalize_process_reasoning(reasoning) is None:
            return {
                "success": False,
                "error": "reasoning is required (non-empty string explaining why this skill operation is needed)",
            }
        ent_err = _validate_process_entries_list(entries)
        if ent_err:
            return {"success": False, "error": ent_err, "results": []}
        results = []
        for entry_data in entries:
            if action == "read":
                entry_id = entry_data.get("id")
                if not entry_id:
                    results.append({"success": False, "error": "Missing 'id' for read"})
                    continue
                entry = skill_store.get(entry_id)
                if entry is None:
                    results.append({"success": False, "error": f"Skill {entry_id} not found"})
                else:
                    results.append({"success": True, "entry": skill_store.to_display_dict(entry)})

            elif action == "add":
                raw_name = entry_data.get("name", entry_data.get("title", ""))
                name = (raw_name or "").strip() if isinstance(raw_name, str) else str(raw_name or "").strip()
                raw_desc = entry_data.get("description", "")
                description = (
                    (raw_desc or "").strip() if isinstance(raw_desc, str) else str(raw_desc or "").strip()
                )
                if not name or not description:
                    results.append(
                        {
                            "success": False,
                            "error": (
                                "add requires non-empty name (or title) and description "
                                "(empty objects {} are rejected; do not batch placeholder rows)"
                            ),
                        }
                    )
                    continue
                path = entry_data.get("path", "general")
                effectiveness = entry_data.get("effectiveness", "medium")
                importance = int(entry_data.get("importance", 3))
                code = entry_data.get("code", "")
                # Allow agent to specify a custom ID (human-readable name)
                custom_id = entry_data.get("id", "")
                add_kwargs = dict(
                    path=path, name=name, title=name, description=description,
                    effectiveness=effectiveness, importance=importance,
                    code=code if isinstance(code, str) else "",
                )
                if custom_id and isinstance(custom_id, str) and custom_id.strip():
                    add_kwargs["id"] = custom_id.strip()
                entry_id = skill_store.add(**add_kwargs)
                results.append({"success": True, "entry_id": entry_id})

            elif action == "update":
                entry_id = entry_data.get("id")
                if not entry_id:
                    results.append({"success": False, "error": "Missing 'id' for update"})
                    continue
                update_fields = {k: v for k, v in entry_data.items() if k != "id" and v is not None}
                if "name" in update_fields:
                    update_fields.setdefault("title", update_fields["name"])
                ok = skill_store.update(entry_id, **update_fields)
                results.append({"success": ok, "entry_id": entry_id})

            elif action == "delete":
                entry_id = entry_data.get("id")
                if not entry_id:
                    results.append({"success": False, "error": "Missing 'id' for delete"})
                    continue
                ok = skill_store.remove(entry_id)
                results.append({"success": ok, "entry_id": entry_id})

            else:
                results.append({"success": False, "error": f"Unknown action: {action}"})

        return {"success": _batch_all_succeeded(results), "results": results}
    except Exception as e:
        logger.error(f"process_skill error: {e}")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Subagent registry helpers
# ---------------------------------------------------------------------------

def get_subagent_overview_direct() -> dict:
    """Get compact tree overview of the subagent registry ([id] name grouped by path)."""
    try:
        overview = subagent_store.get_tree_overview()
        return {"success": True, "overview": overview}
    except Exception as e:
        logger.error(f"Failed to get subagent overview: {e}")
        return {"success": False, "error": str(e)}


def process_subagent_direct(action: str, entries: list, reasoning: object) -> dict:
    """Unified CRUD dispatch for the subagent registry.

    ``reasoning`` is required (non-empty): why this subagent operation is appropriate.
    """
    try:
        if _normalize_process_reasoning(reasoning) is None:
            return {
                "success": False,
                "error": "reasoning is required (non-empty string explaining why this subagent operation is needed)",
            }
        ent_err = _validate_process_entries_list(entries)
        if ent_err:
            return {"success": False, "error": ent_err, "results": []}
        results = []
        for entry_data in entries:
            if action == "read":
                entry_id = entry_data.get("id")
                if not entry_id:
                    results.append({"success": False, "error": "Missing 'id' for read"})
                    continue
                entry = subagent_store.get(entry_id)
                if entry is None:
                    results.append({"success": False, "error": f"Subagent {entry_id} not found"})
                else:
                    results.append({"success": True, "entry": subagent_store.to_display_dict(entry)})

            elif action == "add":
                raw_name = entry_data.get("name", entry_data.get("title", ""))
                name = (raw_name or "").strip() if isinstance(raw_name, str) else str(raw_name or "").strip()
                raw_desc = entry_data.get("description", "")
                description = (
                    (raw_desc or "").strip() if isinstance(raw_desc, str) else str(raw_desc or "").strip()
                )
                if not name or not description:
                    results.append(
                        {
                            "success": False,
                            "error": (
                                "add requires non-empty name (or title) and description "
                                "(empty objects {} are rejected; fill fields before calling)"
                            ),
                        }
                    )
                    continue
                path = entry_data.get("path", "custom")
                handler_type = entry_data.get("handler_type", "looping")
                max_turns = int(entry_data.get("max_turns", 25))
                available_tools = entry_data.get("available_tools", [])
                # Auto-fill default tools for looping subagents if the model omits them
                if handler_type == "looping" and not available_tools:
                    available_tools = ["press_buttons", "get_game_state"]
                    logger.info(f"Auto-filled available_tools for looping subagent '{name}': {available_tools}")
                # Ensure looping subagents always have access to run_skill and run_code
                if handler_type == "looping":
                    for essential in ["run_skill", "run_code", "get_map_data"]:
                        if essential not in available_tools:
                            available_tools.append(essential)
                system_instructions = entry_data.get("system_instructions", "")
                directive = entry_data.get("directive", "")
                return_condition = entry_data.get("return_condition", "")
                importance = int(entry_data.get("importance", 3))
                try:
                    add_kwargs = dict(
                        path=path, name=name, title=name, description=description,
                        handler_type=handler_type, max_turns=max_turns,
                        available_tools=available_tools,
                        system_instructions=system_instructions,
                        directive=directive, return_condition=return_condition,
                        importance=importance,
                    )
                    custom_id = entry_data.get("id", "")
                    if custom_id and isinstance(custom_id, str) and custom_id.strip():
                        add_kwargs["id"] = custom_id.strip()
                    entry_id = subagent_store.add(**add_kwargs)
                    results.append({"success": True, "entry_id": entry_id})
                except ValueError as ve:
                    results.append({"success": False, "error": str(ve)})

            elif action == "update":
                entry_id = entry_data.get("id")
                if not entry_id:
                    results.append({"success": False, "error": "Missing 'id' for update"})
                    continue
                update_fields = {k: v for k, v in entry_data.items() if k != "id" and v is not None}
                if "name" in update_fields:
                    update_fields.setdefault("title", update_fields["name"])
                try:
                    ok = subagent_store.update(entry_id, **update_fields)
                    results.append({"success": ok, "entry_id": entry_id})
                except ValueError as ve:
                    results.append({"success": False, "error": str(ve)})

            elif action == "delete":
                entry_id = entry_data.get("id")
                if not entry_id:
                    results.append({"success": False, "error": "Missing 'id' for delete"})
                    continue
                entry = subagent_store.get(entry_id)
                if entry is not None and getattr(entry, "is_builtin", False):
                    results.append({"success": False, "error": f"Cannot delete built-in subagent {entry_id}"})
                    continue
                ok = subagent_store.remove(entry_id)
                results.append({"success": ok, "entry_id": entry_id})

            else:
                results.append({"success": False, "error": f"Unknown action: {action}"})

        return {"success": _batch_all_succeeded(results), "results": results}
    except Exception as e:
        logger.error(f"process_subagent error: {e}")
        return {"success": False, "error": str(e)}


# Backward-compat aliases
add_knowledge_direct = add_memory_direct
search_knowledge_direct = search_memory_direct
get_knowledge_summary_direct = get_memory_summary_direct
get_knowledge_base = lambda: memory_store
