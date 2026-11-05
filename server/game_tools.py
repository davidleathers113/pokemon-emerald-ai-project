"""
Server-side game tool implementations (no HTTP calls).

These helpers operate directly on the EmeraldEmulator instance and are called
by the /mcp/* endpoint handlers in server/app.py.  The MCP server
(server/cli/pokemon_mcp_server.py) does NOT import this module; it routes
tool calls to app.py over HTTP instead.
"""

import logging
import os
from dataclasses import dataclass
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
                note_live_layout(env.memory_reader)

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
                                    f"Skipping elevation filtering - pathfinding carries the walker's "
                                    f"elevation (player's tile at elevation {player_elevation})"
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
                            from server.field_control import npc_elevations

                            # With each one's elevation: on a bridge it decides whom they
                            # stop (pathfinding _get_blocked_positions).
                            state["map"]["porymap"]["runtime_objects"] = [
                                {"current_x": nx, "current_y": ny, "elevation": elevation}
                                for (nx, ny), elevation in npc_elevations(env.memory_reader).items()
                            ]
                        except Exception as e:  # positions are a refinement; the static list remains
                            logger.debug(f"live object positions unavailable: {e}")
                        # Each object on the tile it stands on now, matched by local id
                        # (field_control.object_places): a trainer who walked over to battle
                        # stays where they stopped, a script moves statues; their map-file
                        # tile is free.
                        try:
                            from server.field_control import object_places
                            from utils.mapping.porymap_state import objects_where_they_stand

                            state["map"]["porymap"]["objects"] = objects_where_they_stand(
                                state["map"]["porymap"]["objects"], object_places(env.memory_reader), coord_offset
                            )
                        except Exception as e:  # the map-file tiles stay blocked, as before
                            logger.debug(f"object places unavailable: {e}")
                        # Strength boulders where they stand now, apart from the people:
                        # walls, or pushed by the walk when the party can use STRENGTH
                        # (pathfinding _push_target; the walk pushes, server _push_boulder).
                        try:
                            from server.field_control import loaded_boulders

                            here = boulder_tiles(porymap_map_name, loaded_boulders(env.memory_reader))
                        except Exception as e:  # unreadable: the map-file tiles
                            logger.debug(f"live boulders unavailable: {e}")
                            here = boulder_tiles(porymap_map_name, {})
                        ox, oy = coord_offset or (0, 0)
                        boulders = [(bx - ox, by - oy) for bx, by in sorted(here)]
                        porymap = state["map"]["porymap"]
                        porymap["boulders"] = boulders
                        porymap["strength"] = can_push_boulders()
                        porymap["objects"] = [o for o in porymap["objects"] if o.get("graphics_id") != PUSHABLE_BOULDER]
                        if porymap.get("runtime_objects"):
                            porymap["runtime_objects"] = [
                                o for o in porymap["runtime_objects"]
                                if (o.get("current_x"), o.get("current_y")) not in here  # live tiles, as listed
                            ]
                        state["map"]["porymap"]["dimensions"] = json_map.get("dimensions", {})
                        state["map"]["porymap"]["warps"] = json_map.get("warps", [])
                        # The tiles that really warp (map.json warp_events), kept apart
                        # from "warps": an override map's are its own copies, and spots
                        # warps only land on are plain floor (arrival_only_warps).
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
                        # Story scenes that take people out of the way when they play:
                        # walks blocked only by those people go through the scene.
                        state["map"]["porymap"]["scenes"] = scene_clears(
                            env, porymap_map_name, state["map"]["porymap"]["objects"], coord_offset
                        )
                        # A rotating tile puzzle's switches are never walked over: a press
                        # moves statues, so only the puzzle's plan steps on one (puzzle_route).
                        ox, oy = coord_offset or (0, 0)
                        state["map"]["porymap"]["puzzle_switches"] = [
                            (sx - ox, sy - oy) for sx, sy in puzzle_switches(porymap_map_name)
                        ]
                        # Rocks and trees the party can clear: walkable for the planner
                        # (at a cost), cleared by the walk when it bumps into one.
                        state["map"]["porymap"]["clearable"] = list(clearable_obstacles(porymap_map_name))
                        # Water the party can Surf joins the walk (pathfinding._water_step);
                        # the walk starts Surf when it bumps into the water (server
                        # _start_surf).
                        state["map"]["porymap"]["surf"] = can_surf()
                        state["map"]["porymap"]["raw_tiles"] = raw_tiles
                        # The player's tile and elevation now: a walk from there starts at
                        # it (pathfinding _astar). On a bridge (elevation 15) the tile can't
                        # tell which floor the player walks on.
                        try:
                            from server.field_control import player_elevation

                            ox, oy = coord_offset or (0, 0)
                            state["map"]["porymap"]["player_position"] = (px - ox, py - oy)
                            state["map"]["porymap"]["player_elevation"] = player_elevation(env.memory_reader)
                        except Exception as e:  # unread: the start tile's elevation is used
                            logger.debug(f"player elevation unavailable: {e}")

                        # For selected dynamic maps (e.g. Mauville Gym), replace static
                        # porymap tiles with the current live map buffer state.
                        apply_live_metatile_overlay(state, env, location_name)
                        # Tiles that drop the walker to the floor below: walks go round
                        # them; only a drop the route takes (FALL_STEP) walks onto one.
                        from utils.mapping.porymap_state import fall_tiles

                        state["map"]["porymap"]["falls"] = sorted(
                            fall_tiles(porymap_map_name, state["map"]["porymap"].get("raw_tiles"))
                        )
                        # Rotating gates (FORTREE CITY GYM): walks plan their pushes from
                        # the gates' live orientations (rotating_gates.py).
                        puzzle, orientations = rotating_gate_state(
                            env, porymap_map_name, state["map"]["porymap"].get("raw_tiles"), coord_offset
                        )
                        state["map"]["porymap"]["rotating_gates"] = puzzle
                        state["map"]["porymap"]["gate_orientations"] = orientations
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


def rotating_gate_state(env, porymap_map_name: str, raw_tiles, coord_offset=None) -> tuple:
    """(puzzle, live orientations) for a map with rotating gates, else (None, None).

    The puzzle is found by the map's id (pret rotating_gate.c
    GetCurrentMapRotatingGatePuzzleType); the orientations are read from VAR_TEMP_0 on.
    When they can't be read, the gates are left out (the walk then bumps into them as
    before) rather than planned from a guess."""
    from utils.mapping import rotating_gates

    puzzle = rotating_gates.puzzle_for_map(
        _raw_map_json(porymap_map_name).get("id"), raw_tiles, tuple(coord_offset or (0, 0))
    )
    if puzzle is None:
        return None, None
    reader = getattr(env, "memory_reader", None)
    orientations = None
    try:
        orientations = rotating_gates.read_orientations(reader, len(puzzle.gates)) if reader else None
    except Exception as e:  # unreadable: walk as before rather than fail the walk
        logger.warning(f"rotating gates: orientations unreadable: {e}")
    if orientations is None:
        logger.warning(f"rotating gates on {porymap_map_name}: no valid orientations in VAR_TEMP_0..; not modelled")
        return None, None
    return puzzle, orientations


def live_gate_orientations(env, porymap_map_name: str) -> tuple | None:
    """The gates' orientations now on this map (caller holds the lock), or None."""
    from utils.mapping import rotating_gates

    config = rotating_gates.PUZZLES.get(_raw_map_json(porymap_map_name).get("id") or "")
    reader = getattr(env, "memory_reader", None)
    if not config or reader is None:
        return None
    return rotating_gates.read_orientations(reader, len(config))


def rotating_gate_blocks(reader, porymap_map_name: str, button: str, tile) -> bool:
    """Did a rotating gate stop the step in direction button onto tile (caller holds the
    lock)? The tile is floor; one of the gate's arms is in the way and a wall is in the
    sweep of its turn (pret CheckForRotatingGatePuzzleCollision)."""
    from utils.mapping import rotating_gates
    from utils.mapping.porymap_state import layout_tiles

    tiles = layout_tiles(porymap_map_name)
    puzzle = rotating_gates.puzzle_for_map(_raw_map_json(porymap_map_name).get("id"), tiles)
    if puzzle is None or button not in rotating_gates.DELTAS or puzzle.collision_at(*tile) != 0:
        return False
    orientations = rotating_gates.read_orientations(reader, len(puzzle.gates))
    return orientations is not None and puzzle.step(orientations, button, tuple(tile)) is None


def planned_end(start: Tuple[int, int], buttons: List[str], pushes: List[dict]) -> Tuple[int, int]:
    """Where a planned walk ends: a boulder push's press leaves the player where they are."""
    from server.field_control import end_of_path

    skip = {p["index"] for p in pushes}
    return end_of_path(start, [b for i, b in enumerate(buttons) if i not in skip])


def push_plan(start: Tuple[int, int], buttons: List[str], pushes: List[dict]) -> dict:
    """For a plan with boulder pushes (pathfinder.last_pushes): "buttons" cut short at the
    first push, "push" (the boulder's tile, where it goes, the press and the tile the
    player pushes from) and "pushes" (how many the whole plan makes); {} without pushes.

    The walk stops before the push and the server pushes (server _push_boulder): the push
    locks the field controls for its animation, which a walk's movement check reports as an
    interruption, and the press moves no one. The server plans again after each push from
    where the boulders really are."""
    from server.field_control import end_of_path

    if not pushes:
        return {}
    first = pushes[0]
    walk = list(buttons[: first["index"]])
    return {
        "buttons": walk,
        "push": {
            "boulder": tuple(first["boulder"]),
            "to": tuple(first["to"]),
            "button": first["button"],
            "stand": end_of_path(start, walk),
        },
        "pushes": len(pushes),
    }


# pret src/event_object_movement.c GetLedgeJumpDirection (from field_player_avatar.c
# CheckForObjectEventCollision): a press toward an MB_JUMP_SOUTH/NORTH/WEST/EAST tile, the
# way it points, jumps over it (PlayerJumpLedge: a two-tile jump) onto the tile beyond.
# The diagonal MB_JUMP_* are not in its table, and slides drawn with the same arrows are
# no jump.
LEDGE_JUMP_STEPS = {"JUMP_SOUTH": (0, 1), "JUMP_NORTH": (0, -1), "JUMP_WEST": (-1, 0), "JUMP_EAST": (1, 0)}


def jump_presses(start: Tuple[int, int], buttons: List[str], tiles) -> List[str]:
    """The presses that walk a planned path: the planner gives one press per tile (onto a
    ledge, then onto its landing), the game jumps both tiles on the first press, so the
    landing's press is dropped. tiles: the plan's layout tuples by row, in its coordinates.

    Seen live in LILYCOVE CITY: go_to_exit("east") from (24,15) jumped the east ledge
    (47,10) with one RIGHT; the press planned for its landing carried the player a tile
    past the path, and every press after that was one tile off, until "BLOCKED moving DOWN
    at (49,13): (49,14) is not walkable"."""
    from server.field_control import DIRECTION_DELTAS
    from utils.mapping.pathfinding import tile_behavior_name

    x, y = start
    out: List[str] = []
    landing = None  # the step that lands a jump just made, already walked by the game
    for button in buttons:
        step = DIRECTION_DELTAS.get(button)
        if step is None:
            out.append(button)
            continue
        x, y = x + step[0], y + step[1]
        if landing == step:
            landing = None
            continue
        out.append(button)
        tile = tiles[y][x] if tiles and 0 <= y < len(tiles) and 0 <= x < len(tiles[y]) else None
        landing = step if LEDGE_JUMP_STEPS.get(tile_behavior_name(tile)) == step else None
    return out


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
        pushes = list(pathfinder.last_pushes)

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
                            # With Surf the planner crosses water already: no way is no way.
                            error_msg += " - Target is water that no shore here reaches" if can_surf() else " - Target requires Surf (water)"
                        elif cell in ["X", "?"]:
                            error_msg += " - Target is out of bounds or unexplored"
                        else:
                            error_msg += f" - Target tile '{cell}' may be unreachable from current position"

            manhattan_dist = abs(x - start_x) + abs(y - start_y)
            if manhattan_dist > 100:
                error_msg += f" (distance: {manhattan_dist} tiles - may be too far)"
            if map_data.get("rotating_gates") is not None:
                error_msg += " (planned through this map's rotating gates as they stand now)"

            return {"success": False, "error": error_msg, "target": f"({x}, {y})", "reason": "unreachable"}

        nav_reason_text = f"Navigating from ({start_x}, {start_y}) to ({x}, {y})"
        if nav_reason:
            nav_reason_text += f": {nav_reason}"
        if variance_level:
            nav_reason_text += f" (variance={variance_level})"

        logger.info(f"🗺️ {nav_reason_text} - Buttons calculated: {len(buttons)}")

        # Same coordinate frame as the requested target (map-local on override maps).
        local_end = planned_end(start, buttons, pushes)
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
        result.update(push_plan(start, buttons, pushes))
        result["buttons"] = jump_presses(start, result["buttons"], map_data.get("raw_tiles"))
        if pathfinder.last_scene is not None:
            # The walk goes through a story scene that clears the way: the game stops it
            # there, and the caller plans again once the scene has played.
            result["scene"] = list(pathfinder.last_scene)
        if map_data.get("rotating_gates") is not None:
            # The gates as the plan leaves them: the walk checks the game agrees.
            result["gates_after"] = list(pathfinder.last_gate_orientations or ())
            result["gates_before"] = list(map_data.get("gate_orientations") or ())

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
                if map_data.get("boulders") and not map_data.get("strength"):
                    result["note"] += (
                        "; boulders ('R') may be in the way, and pushing them needs a party Pokémon that "
                        "knows STRENGTH and the Heat Badge"
                    )

        return result
    except Exception as e:
        logger.error(f"Failed to navigate: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


def list_map_exits(json_map: dict, porymap_map_name: str, coord_offset=None, live_tile=None) -> list:
    """Every exit the EXITS prompt line lists, with the coordinates go_to_exit needs.

    Warp coordinates come from the map's raw warp_events (an override map's json_map has
    its own few). Each warp's "how" is the EXITS note on how it fires when stepping on it
    is not enough (porymap_state.warp_entry: an exit mat is stood on and walked off, a
    door is walked into from below), and "entry" that (tile to stand on, button), or None.

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
        warp_entry,
        warp_entry_text,
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
            "open_span": connection_open_span(conn, surf=can_surf()),
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
    ox, oy = coord_offset or (0, 0)
    arrival_only = arrival_only_warps(porymap_map_name or "")
    sign_doors = sign_door_warps(porymap_map_name or "")
    for i, warp in enumerate(raw_warps):
        if i in arrival_only and not _fires_now(live_tile, warp):
            continue  # a spot warps land on: stepping on it goes nowhere
        name = _readable_map_name(warp.get("dest_map", "?"))
        wx, wy = int(warp.get("x", 0)), int(warp.get("y", 0))
        entry = warp_entry(porymap_map_name or "", i, wx, wy)
        exits.append({
            "kind": "warp",
            "x": wx - ox,
            "y": wy - oy,
            "name": name,
            "label": f"door/warp at ({wx - ox}, {wy - oy}) -> {name}",
            "how": warp_entry_text(entry, wx, wy, (ox, oy)),
            "entry": ((entry[0][0] - ox, entry[0][1] - oy), entry[1]) if entry else None,
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


def map_file_tiles(map_dir: str) -> Dict[int, Tuple[int, int]]:
    """{local id: map.json tile} of a map's object events (pret's localId: the place in
    object_events, from 1)."""
    return {
        number: (int(obj.get("x", -1)), int(obj.get("y", -1)))
        for number, obj in enumerate(_raw_map_json(map_dir).get("object_events") or [], start=1)
    }


def statue_at(reader, map_dir: str, x: int, y: int) -> bool:
    """Whether a statue (porymap_state.STATUE_GRAPHICS) stands on (x, y) now, by where
    each object stands (field_control.object_places)."""
    from server.field_control import object_places
    from utils.mapping.porymap_state import STATUE_GRAPHICS

    objects = _raw_map_json(map_dir).get("object_events") or []
    return any(
        tile == (x, y) and 0 < local_id <= len(objects) and objects[local_id - 1].get("graphics_id") in STATUE_GRAPHICS
        for local_id, tile in object_places(reader).items()
    )


# A go_to route step that presses a rotating tile puzzle's floor switch (puzzle_route):
# "FLOOR SWITCH (x, y)", walked onto by server/app.py _press_floor_switch.
SWITCH_STEP = "FLOOR SWITCH"


@lru_cache(maxsize=None)
def puzzle_switches(map_dir: str) -> Dict[Tuple[int, int], int]:
    """{tile: color} of a map's rotating tile puzzle switches: coord-event triggers whose
    script runs `moverotatingtileobjects <color>` (pret MossdeepCity_Gym scripts.inc
    MossdeepCity_Gym_EventScript_YellowFloorSwitch: moverotatingtileobjects 0). {} on
    every other map."""
    out = {}
    for event in _raw_map_json(map_dir).get("coord_events") or []:
        if event.get("type") != "trigger":
            continue
        for line in _script_lines(map_dir, str(event.get("script", ""))):
            command, _, arg = line.partition(" ")
            if command == "moverotatingtileobjects" and arg.strip().isdigit():
                out[(int(event.get("x", -1)), int(event.get("y", -1)))] = int(arg.strip())
                break
    return out


def _puzzle_is_trick_house(map_dir: str) -> bool:
    """pret `initrotatingtilepuzzle TRUE` (the Trick House's tiles) in a switch's script."""
    for event in _raw_map_json(map_dir).get("coord_events") or []:
        for line in _script_lines(map_dir, str(event.get("script", ""))):
            command, _, arg = line.partition(" ")
            if command == "initrotatingtilepuzzle":
                return arg.strip() == "TRUE"
    return False


def puzzle_objects(reader, map_dir: str) -> Tuple[Tuple[int, int], ...]:
    """Where each shown object of the map stands now (field_control.object_places), in
    object_events order; objects whose hide flag is set are left out. Read under the
    emulator lock; the plan made from it (puzzle_route) is not."""
    from server.field_control import object_places
    from utils.mapping.flag_hide_constants import FLAG_HIDE_IDS

    places = object_places(reader)
    out = []
    for local_id, obj in enumerate(_raw_map_json(map_dir).get("object_events") or [], start=1):
        flag_id = FLAG_HIDE_IDS.get(str(obj.get("flag", "0")))
        try:
            hidden = flag_id is not None and reader._is_event_flag_set(flag_id)
        except Exception:  # unreadable: the object counts as there
            hidden = False
        if not hidden and local_id in places:
            out.append(places[local_id])
    return tuple(out)


def puzzle_plan(places, map_dir: str, position: Tuple[int, int], target_tiles) -> Optional[list]:
    """rotating_tile_puzzle.plan from the player at position (map tiles) to any of
    target_tiles, with the objects at places (puzzle_objects); None on a map without a
    puzzle or when no way is found.

    Steps are the tile planner's own (pathfinding.walk_steps) on the static layout, less
    the objects' tiles. Other coord-event triggers and warps to other maps are never
    stepped on; warps to another spot of this map (MOSSDEEP CITY GYM's pads) are rides.
    """
    return _puzzle_plan(tuple(places), map_dir, tuple(position), frozenset(tuple(t) for t in target_tiles))


@lru_cache(maxsize=64)
def _puzzle_plan(places: tuple, map_dir: str, position: tuple, goal: frozenset) -> Optional[tuple]:
    """puzzle_plan, kept per (places, position, goal): a walk asks again after every step
    and arrived() asks for the same one again (a search can take ~2 s)."""
    from utils.mapping import rotating_tile_puzzle as puzzle
    from utils.mapping.pathfinding import walk_steps
    from utils.mapping.porymap_state import _EDGE_BLOCKED, _static_grid, _static_tiles, arrival_only_warps

    switches = puzzle_switches(map_dir)
    grid, tiles = _static_grid(map_dir), _static_tiles(map_dir)
    if not switches or not grid or not tiles:
        return None
    raw = _raw_map_json(map_dir)
    pads, closed = {}, set()
    landing = arrival_only_warps(map_dir)
    for i, warp in enumerate(raw.get("warp_events") or []):
        tile = (int(warp.get("x", -1)), int(warp.get("y", -1)))
        arrival = warp_arrival(warp)
        if i in landing:
            continue
        # A pad that is itself the goal (go_to_exit to "MOSSDEEP CITY GYM (1, 23)") is
        # walked to, not ridden: plan() never counts a pad's tile as reached.
        if warp.get("dest_map") == raw.get("id") and arrival is not None and tile not in goal:
            pads[tile] = arrival
        else:
            closed.add(tile)
    closed |= {
        (int(e.get("x", -1)), int(e.get("y", -1))) for e in raw.get("coord_events") or []
        if e.get("type") == "trigger"
    } - set(switches)
    closed -= goal
    walls = set(_EDGE_BLOCKED)

    def steps(tile, blocked):
        for nx, ny, _ in walk_steps(tiles, grid, tile[0], tile[1], None, False, walls):
            if (nx, ny) not in blocked and (nx, ny) not in closed:
                yield nx, ny

    found = puzzle.plan(
        position, places, goal, steps, switches, pads, puzzle.arrows_of(tiles, _puzzle_is_trick_house(map_dir)),
    )
    return tuple(found) if found is not None else None


def puzzle_route(places, map_dir: str, position: Tuple[int, int], target_tiles) -> Optional[List[Tuple[str, str]]]:
    """puzzle_plan as go_to route steps, [(step, name of the map), ...]: a switch press is
    (SWITCH_STEP + " (x, y)"), a pad the go_to_exit argument "MAP NAME (x, y)". [] when a
    target tile can be walked to now, None when no way is found or the map has no
    puzzle. Planned again after every step, from the objects' places then (places:
    puzzle_objects, read live)."""
    from utils.mapping.porymap_state import _readable_map_name

    actions = puzzle_plan(places, map_dir, position, target_tiles)
    if actions is None:
        return None
    name = _readable_map_name(str(_raw_map_json(map_dir).get("id", map_dir)))
    out = []
    for kind, (x, y) in actions:
        if kind == "press":
            out.append((f"{SWITCH_STEP} ({x}, {y})", name))
        elif kind == "warp":
            out.append((f"{name} ({x}, {y})", name))
    return out


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
def _flag_ids_resolved() -> dict:
    """Every flag in include/constants/flags.h, the "(BASE + 0xN)" ones too (FLAG_BADGE07_GET
    is (SYSTEM_FLAGS + 0xD)), where _flag_ids keeps only plain numbers. SYSTEM_FLAGS is
    seeded: flags.h derives it from MAX_TRAINERS_COUNT, defined elsewhere."""
    from pokemon_env.porymap_paths import get_porymap_root

    root = get_porymap_root()
    path = Path(root) / "include" / "constants" / "flags.h" if root else None
    ids: dict = {"SYSTEM_FLAGS": SYSTEM_FLAGS}
    if path is None or not path.exists():
        return ids

    def number(text: str) -> Optional[int]:
        text = text.strip().strip("()").strip()
        try:
            return int(text, 16) if text.lower().startswith("0x") else int(text)
        except ValueError:
            return ids.get(text)

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or parts[0] != "#define" or not parts[1].startswith("FLAG_"):
            continue
        value = parts[2].split("//")[0].strip()
        if value.startswith("(") and value.count("+") == 1 and "-" not in value:
            base, _, offset = value.strip("()").partition("+")
            left, right = number(base), number(offset)
            if left is not None and right is not None:
                ids[parts[1]] = left + right
        else:
            found = number(value)
            if found is not None:
                ids[parts[1]] = found
    return ids


def flag_id(name: str) -> Optional[int]:
    """The id of the flag pret's flags.h calls name (e.g. FLAG_BADGE07_GET), else None."""
    return _flag_ids_resolved().get(str(name).strip())


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
    return [
        (int(event.get("x", -1)), int(event.get("y", -1)))
        for event in _raw_map_json(map_dir).get("coord_events") or []
        if event.get("type") == "trigger" and _trigger_fires(env, event, unknown=True)
    ]


def _trigger_fires(env, event: dict, unknown: bool) -> bool:
    """Whether a coord-event trigger runs its script when stepped on now (pret: when
    VarGet(var) == var_value); `unknown` when its var can't be resolved or read."""
    var_id = _var_ids().get(str(event.get("var", "")))
    try:
        wanted = int(str(event.get("var_value", "0")), 0)
    except ValueError:
        return unknown
    if var_id is None or env is None:
        return unknown
    try:
        value = env.memory_reader.read_game_var(var_id)
    except Exception as e:  # unreadable var: the caller's guess
        logger.debug(f"trigger var {event.get('var')} unreadable: {e}")
        return unknown
    return unknown if value is None else value == wanted


def scene_clears(env, map_dir: str, objects: list, coord_offset=None) -> List[dict]:
    """[{"tiles": its trigger tiles, "clears": the tiles it frees}, ...] for each
    coord-event script that plays when stepped on now and `removeobject`s people of this
    map: the tiles are where those people (objects, placed where they stand and in walk
    coordinates, as load_pathfinding_state has them) stand now. The planner walks into
    such a scene when those people wall the only way (pathfinding _through_scene).

    pret MossdeepCity_EventScript_TeamMagmaEnterSpaceCenter: Maxie and four grunts stand
    across the street until the scene in front of them walks them into the Space Center
    and removes them. A var that can't be read counts as not playing here: a person
    planned away who then stays would end the walk at them.
    """
    ox, oy = coord_offset or (0, 0)
    by_script: Dict[str, List[Tuple[int, int]]] = {}
    for event in _raw_map_json(map_dir).get("coord_events") or []:
        if event.get("type") == "trigger" and _trigger_fires(env, event, unknown=False):
            tile = (int(event.get("x", -1)) - ox, int(event.get("y", -1)) - oy)
            by_script.setdefault(str(event.get("script", "")), []).append(tile)
    out = []
    for script, tiles in by_script.items():
        removed = {
            line.partition(" ")[2].split(",")[0].strip()
            for line in _script_lines(map_dir, script)
            if line.partition(" ")[0] == "removeobject"
        } - set(_PLAYER_IDS)
        clears = [
            (int(obj.get("x", -1)), int(obj.get("y", -1)))
            for obj in objects
            if (obj.get("local_id") and obj.get("local_id") in removed) or str(obj.get("event_id", "")) in removed
        ]
        if clears:
            out.append({"tiles": tiles, "clears": clears})
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
# Smash the Dynamo Badge (FLAG_BADGE03_GET). Strength boulders are not cleared but pushed,
# and a push needs room behind the boulder, so they are planned tile by tile
# (pathfinding _push_target, boulder_tiles below), not here.
FLAG_BADGE01_GET = 0x867  # include/constants/flags.h SYSTEM_FLAGS + 0x7
FIELD_MOVE_OBSTACLES = {
    "OBJ_EVENT_GFX_CUTTABLE_TREE": ("CUT", 1),
    "OBJ_EVENT_GFX_BREAKABLE_ROCK": ("ROCK_SMASH", 3),
}
# Surf (outside battle) needs the Balance Badge: pret src/field_control_avatar.c
# GetInteractedWaterScript checks FlagGet(FLAG_BADGE05_GET) and PartyHasMonWithSurf; the
# script it runs (data/scripts/surf.inc EventScript_UseSurf, checkpartymove MOVE_SURF)
# skips eggs.
SURF = "SURF"
# Fly (outside battle) needs the Feather Badge: pret src/party_menu.c CursorCb_FieldMove
# refuses a field move up to FIELD_MOVE_WATERFALL until FlagGet(FLAG_BADGE01_GET +
# fieldMove), and FIELD_MOVE_FLY is 5, so FLAG_BADGE06_GET (pokebot-gen3's fly_to checks
# badge 3, which is wrong for Emerald). The party menu only lists it for a Pokémon that
# knows the move (SetPartyMonFieldSelectionActions), and never for an egg.
FLY = "FLY"
# Strength (outside battle) needs the Heat Badge: pret data/scripts/field_move_scripts.inc
# EventScript_StrengthBoulder, goto_if_unset FLAG_BADGE04_GET, then checkpartymove
# MOVE_STRENGTH; YES to "Would you like to use STRENGTH?" sets FLAG_SYS_USE_STRENGTH, and
# from then on a step into a boulder pushes it (src/field_player_avatar.c TryPushBoulder)
# until the next map load clears the flag (src/event_data.c ClearTempFieldEventData).
STRENGTH = "STRENGTH"
PUSHABLE_BOULDER = "OBJ_EVENT_GFX_PUSHABLE_BOULDER"
# Every field move the walks use, by the badge number that allows it outside battle.
FIELD_MOVE_BADGES = {
    **{move: badge for move, badge in FIELD_MOVE_OBSTACLES.values()}, SURF: 5, FLY: 6, STRENGTH: 4,
}
# () -> set of field moves usable now (known by a party Pokémon, badge set), installed by
# the server. None: no obstacle is clearable and no water crossable.
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


def can_surf() -> bool:
    """A party Pokémon (not an egg) knows SURF and the Balance Badge is set: the game
    offers "Would you like to SURF?" facing water from the shore."""
    return SURF in _usable_field_moves()


def can_push_boulders() -> bool:
    """A party Pokémon (not an egg) knows STRENGTH and the Heat Badge is set: A at a
    boulder offers "Would you like to use STRENGTH?" and the walks push boulders."""
    return STRENGTH in _usable_field_moves()


# () -> (map dir the player is on, {local id: tile} of its boulders loaded now
# (field_control.loaded_boulders)), installed by the server. None: every boulder stands on
# its map-file tile.
boulders_reader = None


def boulder_spawns(map_dir: str) -> dict:
    """{local id: map-file tile} of the map's Strength boulders. pret gives each object
    event the local id of its place in map.json's object_events, counting from 1."""
    return {
        i + 1: (int(obj.get("x", -1)), int(obj.get("y", -1)))
        for i, obj in enumerate(_raw_map_json(map_dir).get("object_events") or [])
        if obj.get("graphics_id") == PUSHABLE_BOULDER
    }


def boulder_places(map_dir: str, loaded: Optional[dict] = None) -> dict:
    """{map-file tile: tile it stands on now} for the map's boulders: a loaded boulder's
    live tile, any other's map-file tile (field_control.loaded_boulders says why). loaded:
    {local id: tile} for the map the player is on; None asks boulders_reader, which
    answers only for that map."""
    spawns = boulder_spawns(map_dir)
    if not spawns:
        return {}
    if loaded is None and boulders_reader is not None:
        try:
            here, live = boulders_reader()
            loaded = live if here == map_dir else None
        except Exception as e:  # unreadable: the map-file tiles, as after a map load
            logger.debug(f"boulders unreadable: {e}")
    return {tile: (loaded or {}).get(local_id, tile) for local_id, tile in spawns.items()}


def boulder_tiles(map_dir: str, loaded: Optional[dict] = None) -> frozenset:
    """The tiles the map's boulders stand on now (boulder_places)."""
    return frozenset(boulder_places(map_dir, loaded).values())


def surf_start_tile(map_dir: str, tile, layout: str = "") -> bool:
    """Whether the map's (x, y) is water Surf can be started onto (pathfinding
    SURF_START_BEHAVIORS), by the decomp's layout: `layout` (a LAYOUT_* id) when a script
    swapped one in (dynamic_map_overlay.swapped_layout), else the map file's. Seen in a
    test: SHOAL CAVE's entrance room at high tide, (8,22) is water there but cave floor in
    the map file's low-tide layout, and the walk to its cave floor stopped at the shore."""
    from utils.mapping.pathfinding import is_surf_start
    from utils.mapping.porymap_state import layout_tiles

    tiles = layout_tiles(map_dir, layout) if map_dir else ()
    x, y = int(tile[0]), int(tile[1])
    return 0 <= y < len(tiles) and 0 <= x < len(tiles[y]) and is_surf_start(tiles[y][x])


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
    the turn_back_gates the bag does not open.

    Strength boulders block where they stand now (boulder_tiles), and not at all when the
    party can push them: which pushes open a way is the tile planner's search
    (pathfinding _push_target), too fine for these areas. Seen live in MAGMA HIDEOUT 1F:
    heal_at_pokecenter found no way out ("no nurse is on this map"). The boulders counted
    as walls on their map-file tiles, pushed aside or not, with STRENGTH in the party."""
    blocked = _shut_gates(map_dir)
    usable = _usable_field_moves()
    if STRENGTH not in usable:
        blocked |= boulder_tiles(map_dir)
    for obj in _raw_map_json(map_dir).get("object_events") or []:
        if obj.get("graphics_id") not in _OBSTACLE_GRAPHICS or obj.get("graphics_id") == PUSHABLE_BOULDER:
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

    When the party can Surf (can_surf), water is part of the areas too: it joins the
    shores it can be surfed from and landed on (pathfinding.water_step).
    """
    return _walk_areas_without(map_dir, _obstacle_tiles(map_dir) if map_dir else frozenset(), True, can_surf())


@lru_cache(maxsize=None)
def _walk_areas_without(
    map_dir: str, obstacles: frozenset, ledges_are_walls: bool = False, surf: bool = False
) -> dict:
    """The areas are flooded with the tile planner's own steps (pathfinding.walk_steps):
    collision bits, and the walker's elevation carried from tile to tile (pret
    IsElevationMismatchAt, ObjectEventUpdateElevation), so floors at different elevations
    are different areas unless a tile at elevation 0 (stairs, a ramp) or a bridge joins
    them. Seen live: FORTREE CITY's heal walk was routed from the gym door down onto the
    treetop walkway at (20,14), elevation 4, from the ground beside it, elevation 3.

    surf: water joins the areas, with the steps pathfinding.water_step allows: from a
    shore onto water Surf can start on and back, water to water, and under a bridge
    (elevation 15, which keeps the surfer's elevation). A bridge tile is labelled with the
    land it carries when a walker reaches it; a surfer passes under it in the water's area.
    """
    from utils.mapping.pathfinding import (
        BRIDGE_ELEVATION, PASSABLE_WITH_COLLISION, WATER_ELEVATION, is_surf_water, walk_steps,
    )
    from utils.mapping.porymap_state import _EDGE_BLOCKED, _static_grid, _static_tiles, fall_tiles

    grid = _static_grid(map_dir) if map_dir else None
    if not grid:
        return {}
    # A tile that drops the walker to the floor below is no floor to walk across; the
    # drop is a way of its own (_fall_exits).
    obstacles = obstacles | fall_tiles(map_dir, _static_tiles(map_dir))
    grid = [
        "".join("#" if (x, y) in obstacles else cell for x, cell in enumerate(row))
        for y, row in enumerate(grid)
    ] if obstacles else grid
    walls = set(_EDGE_BLOCKED) | (set(_LEDGE_STEPS) if ledges_are_walls else set())
    tiles = _static_tiles(map_dir)
    if len(tiles) != len(grid) or any(len(a) != len(b) for a, b in zip(tiles, grid)):
        tiles = ()  # no tiles to check: walls by symbol only, no elevation

    def tile(x: int, y: int):
        return tiles[y][x] if tiles else None

    def standable(x: int, y: int) -> bool:
        """A tile someone can stand on: the walk_steps rules for the tile itself."""
        here = tile(x, y)
        if here is not None and is_surf_water(here):
            return surf
        if grid[y][x] in walls:
            return False
        return here is None or len(here) < 3 or not here[2] or grid[y][x] in PASSABLE_WITH_COLLISION

    area: dict = {}
    states: dict = {}
    by_surfer = set()  # bridge tiles labelled only from under them so far

    def label(x: int, y: int, walker, number: int) -> None:
        here = tile(x, y)
        bridge = here is not None and len(here) > 3 and here[3] == BRIDGE_ELEVATION
        if (x, y) not in area or ((x, y) in by_surfer and walker != WATER_ELEVATION):
            area[(x, y)] = number
            if bridge and walker == WATER_ELEVATION:
                by_surfer.add((x, y))
            else:
                by_surfer.discard((x, y))

    def flood(x0: int, y0: int, walker, number: int) -> None:
        states[(x0, y0, walker)] = number
        label(x0, y0, walker, number)
        stack = [(x0, y0, walker)]
        while stack:
            x, y, here = stack.pop()
            for nx, ny, there in walk_steps(tiles, grid, x, y, here, surf, walls):
                if (nx, ny, there) in states:
                    continue
                states[(nx, ny, there)] = number
                label(nx, ny, there, number)
                stack.append((nx, ny, there))

    number = -1
    # Bridges last: whoever stands on one carries the elevation of the floor they came
    # from, so a bridge is reached from its ends; one no end reaches is flooded unknown.
    for bridges in (False, True):
        for y0, row in enumerate(grid):
            for x0, cell in enumerate(row):
                here = tile(x0, y0)
                elevation = here[3] if here is not None and len(here) > 3 else None
                if (x0, y0) in area or (elevation == BRIDGE_ELEVATION) != bridges or not standable(x0, y0):
                    continue
                number += 1
                flood(x0, y0, None if bridges else elevation, number)
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


def _edge_dead_ends(map_dir: str, dest_const: str, direction: str, offset: int) -> frozenset:
    """Positions along this map's edge (x for up/down, y for left/right, this map's frame)
    whose arrival tile next door is in an area with no way on but back here.

    Seen live: go_to_exit("east") from VERDANTURF TOWN (19,6) took the nearest edge tile,
    y=6, and landed on ROUTE 117 (0,6): a one-tile notch walled in on three sides, so
    every walk and train call from there failed. Route 117 goes on from y=8.
    """
    return _edge_dead_ends_with(map_dir, dest_const, direction, offset, can_surf())


@lru_cache(maxsize=256)
def _edge_dead_ends_with(map_dir: str, dest_const: str, direction: str, offset: int, surf: bool) -> frozenset:
    """_edge_dead_ends, kept per Surf ability: the areas it reads (_route_areas) change with it."""
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


def _ride_exits(map_dir: str) -> tuple:
    """_walk_exits' shape for a ride (RIDE_EXIT) from this map, or () if it has none.

    emerald-new (2026-11-04): from the Lavaridge Pokémon Center, go_to("Jagged Pass")
    walked into Jagged Pass's bottom (14,40), below one-way ledges, where the tall grass
    can't be reached; the way to it is the Cable Car up to Mt. Chimney.
    """
    return _ride_exits_with(map_dir, can_surf())


@lru_cache(maxsize=None)
def _ride_exits_with(map_dir: str, surf: bool) -> tuple:
    """_ride_exits, kept per Surf ability: the areas it names (_route_areas) change with it."""
    from utils.mapping.porymap_state import _readable_map_name, _map_dirs_by_const

    ride = CABLE_CAR_STATIONS.get(map_dir)
    if ride is None:
        return ()
    (x, y), dest = ride
    const = next((c for c, d in _map_dirs_by_const().items() if d == dest), dest)
    mine = _areas_at(map_dir, x, y + 1, _route_areas(map_dir))
    theirs = _areas_at(dest, x, y + 1, _route_areas(dest))
    return tuple((m, RIDE_EXIT, dest, t, _readable_map_name(const)) for m in mine for t in theirs)


# A route step that drops through the floor (_fall_exits): "FALL THROUGH (x, y)", walked
# by stepping onto that tile (server/app.py _fall_through).
FALL_STEP = "FALL THROUGH"


def _fall_exits(map_dir: str) -> tuple:
    """_walk_exits' shape for each tile of this map that drops the walker to the floor
    below (porymap_state.fall_tiles), at the same tile there (pret ScrCmd_warphole).

    pret data/maps/SkyPillar_4F: before Rayquaza wakes (VAR_SKY_PILLAR_STATE < 2) the 5F
    stairs at (3,1) are reached only from 3F's stairs at (7,1), which stand in a pocket of
    3F that nothing walks into: the way in is a drop through 4F's cracked floor at (6,4).
    """
    return _fall_exits_with(map_dir, can_surf())


@lru_cache(maxsize=None)
def _fall_exits_with(map_dir: str, surf: bool) -> tuple:
    """_fall_exits, kept per Surf ability: the areas it names (_route_areas) change with it."""
    from utils.mapping.porymap_state import _readable_map_name, _static_tiles, fall_tiles, hole_warp

    dest = hole_warp(map_dir)[0]
    if not dest:
        return ()
    name = _readable_map_name(str(_raw_map_json(dest).get("id", dest)))
    here, there = _route_areas(map_dir), _route_areas(dest)
    out = set()
    for x, y in sorted(fall_tiles(map_dir, _static_tiles(map_dir))):
        for mine in _areas_at(map_dir, x, y, here):
            for theirs in _areas_at(dest, x, y, there):
                out.add((mine, f"{FALL_STEP} ({x}, {y})", dest, theirs, name))
    return tuple(sorted(out))


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
    The same as go_to_exit's exit names (field_control.normalize_name).
    """
    from server.field_control import normalize_name

    return normalize_name(text)


_HALVES = {"NORTH": (1, -1), "SOUTH": (1, 1), "WEST": (0, -1), "EAST": (0, 1)}


def _is_floor_or_variant(word: str) -> bool:
    """'1F', 'B2F', '1R': parts of a map name that only say which floor or room it is.

    Seen live: on METEOR FALLS 1F 1R, inside the Route 114 door, "Head west through Route
    114 to Meteor Falls" (meteor_falls_reached) never showed as done, and the model walked
    back out twice. The decomp numbers the rooms of a floor 1R, 2R (Meteor Falls, Magma
    Hideout).
    """
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


# The most words in a row (_words_in_order) that may stand for one word written together
# on the other side: "HOUSE 3" for HOUSE3, "ROUTE104" for ROUTE 104.
_JOINED_WORDS_MAX = 3


def _words_in_order(have: Tuple[str, ...], want: Tuple[str, ...], to_the_end: bool = False) -> bool:
    """True when want's words (_place_words) appear in have's in order, spaces aside: a
    run of words on one side may stand for one word written together on the other.
    to_the_end: want's last word must stand for have's last.

    Seen live: go_to("MOSSDEEP CITY HOUSE 3"), standing in Mossdeep, answered "can't be
    reached on foot or by Surf": the map is MOSSDEEP CITY HOUSE3, and word by word no map
    matched."""

    @lru_cache(maxsize=None)
    def from_(i: int, j: int) -> bool:
        if i == len(want):
            return not to_the_end or j == len(have)
        for start in range(j, len(have)):
            for k in range(1, min(_JOINED_WORDS_MAX, len(want) - i) + 1):
                wanted = "".join(want[i:i + k])
                for m in range(1, min(_JOINED_WORDS_MAX, len(have) - start) + 1):
                    if k > 1 and m > 1:
                        continue  # a run on one side stands for a single word on the other
                    had = "".join(have[start:start + m])
                    same = _same_word(had, wanted) if k == m == 1 else had == wanted
                    if same and from_(i + k, start + m):
                        return True
        return False

    return from_(0, 0)


def _same_place_words(name: str, words: List[str]) -> bool:
    """A map name written as the words, spaces, case and punctuation aside ("MOSSDEEP CITY
    HOUSE 3" is MOSSDEEP CITY HOUSE3)."""
    return "".join(_place_words(name)) == "".join(words)


def _names_same_place(location_name: str, target: str) -> bool:
    """True when a loosely written target names this map.

    Seen live: the objective list writes "Dewford Gym" for DEWFORD TOWN GYM and "Granite
    Cave" for GRANITE CAVE 1F. The target's words must appear in the map name in order
    (_words_in_order, spaces aside), and the map name, without a floor or room suffix, must
    end with the target's last word. So "Rustboro City" does not match RUSTBORO CITY GYM.
    """
    have = _place_words(location_name)
    while have and _is_floor_or_variant(have[-1]):
        have = have[:-1]
    want = _place_words(target)
    if not have or not want:
        return False
    return _words_in_order(tuple(have), tuple(want), to_the_end=True)


@lru_cache(maxsize=None)
def _all_place_names() -> Tuple[str, ...]:
    """Every map's name as the game's state shows it (MOSSDEEP CITY HOUSE3), sorted."""
    from utils.mapping.porymap_state import _map_dirs_by_const, _readable_map_name

    return tuple(sorted({_readable_map_name(const) for const in _map_dirs_by_const()}))


# How many real names an unknown place name is answered with (unknown_place).
UNKNOWN_PLACE_SUGGESTIONS = 3


def _target_words(target: str) -> List[str]:
    """The target's words as go_to matches them: a trailing North/South/East/West (a half
    of a route, arrived_at) dropped."""
    words = _place_words(target)
    if len(words) > 1 and words[-1] in _HALVES:
        words = words[:-1]
    return words


def unknown_place(target: str) -> Optional[List[str]]:
    """None when some map is named by target the way go_to matches names (place_route:
    the name itself, or its words in order, spaces, case and punctuation aside); else the
    UNKNOWN_PLACE_SUGGESTIONS real names closest to it (difflib), for an answer that says
    the name is unknown rather than that the place can't be reached."""
    import difflib

    words = _target_words(target)
    names = _all_place_names()
    if words and any(
        _same_place_words(name, words) or _words_in_order(tuple(_place_words(name)), tuple(words)) for name in names
    ):
        return None
    by_squash = {}
    for name in names:
        by_squash.setdefault(_squash_name(name), name)
    close = difflib.get_close_matches(
        "".join(words) or _squash_name(target), list(by_squash), n=UNKNOWN_PLACE_SUGGESTIONS, cutoff=0.0
    )
    return [by_squash[squashed] for squashed in close]


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
        # The areas joined across water when the party can Surf (_route_areas).
        how = "on foot (and by Surf over water)" if can_surf() else "on foot"
        lines.append(
            f"{path[-1][1]} is not next to this map; the way {how} is {way}. "
            f'go_to_exit("{exit_arg}") takes the first step.'
        )
    return " ".join(lines)


def _walk_paths(
    map_dir: str, matches, position: Optional[Tuple[int, int]] = None, other_parts_of_here: bool = False,
    rides: bool = False, goal: Optional[frozenset] = None, falls: bool = False,
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

    falls: a drop through the floor to the floor below counts as a step too (_fall_exits).
    Callers ask for that only when no way without one is found (_walk_paths_or_fall).
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
            for from_area, exit_arg, dest, dest_area, name in (
                _walk_exits(node[0]) + (_ride_exits(node[0]) if rides else ()) + (_fall_exits(node[0]) if falls else ())
            ):
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
    paths = _walk_paths_or_fall(here_dir or map_dir, lambda name: False, position, goal=goal)
    return paths[0] if paths else None


def _walk_paths_or_fall(map_dir: str, matches, position: Optional[Tuple[int, int]] = None, **kwargs):
    """_walk_paths, and only when it finds no way, _walk_paths with drops through the
    floor (falls=True): a drop can't be walked back, so it is taken only when it is the
    way, never as a shortcut, and only on the way to the goal."""
    return _walk_paths(map_dir, matches, position, **kwargs) or _walk_paths(
        map_dir, matches, position, falls=True, **kwargs
    )


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
    paths = _walk_paths_or_fall(
        ROM_TO_PORYMAP_MAP.get(location_name, ""), lambda name: _squash_name(name).endswith(target), position
    )
    return paths[0] if paths else []


def way_to_pokemon_center(
    location_name: str, position: Optional[Tuple[int, int]], toward: Optional[str] = None
) -> str:
    """Where to heal: "" when already inside a Pokémon Center 1F (the nurse's floor),
    else the way on foot to the nearest one, or, when none can be reached, to keep going
    (toward the objective's place `toward`, naming the next Center on the way).

    Map names end in "POKEMON CENTER 1F" (RUSTBORO CITY POKEMON CENTER 1F); a 2F's way
    down the stairs is found the same way as any other.
    """
    if _squash_name(location_name).endswith(_squash_name(POKEMON_CENTER_1F)):
        return ""
    path = pokemon_center_route(location_name, position)
    if not path:
        return no_pokemon_center_note(location_name, toward)
    # heal_at_pokecenter walks the way itself (server _walk_to_pokemon_center); the way
    # is named so the model knows where it is going.
    way = " → ".join(name for _, name in path)
    return f"call heal_at_pokecenter(): it walks to {path[-1][1]} ({way}) and heals."


def no_pokemon_center_note(location_name: str, toward: Optional[str] = None) -> str:
    """What to do when no Pokémon Center can be reached from here: keep going, and heal
    at the next one on the way (next_pokemon_center), never "go back the way you came".

    Seen live: one fainted CAMERUPT and five healthy teammates in ROUTE 119 WEATHER
    INSTITUTE; the advice to heal sent the model back over water and ledges it could not
    cross (heal_at_pokecenter refused, go_to Mauville), away from the objective.
    """
    how = "on foot or by Surf" if can_surf() else "on foot"
    head = f"no Pokémon Center can be reached {how} from here, so don't turn back for one: keep going"
    center = next_pokemon_center(location_name, toward)
    if center:
        return f"{head} toward {toward} and heal at {center}, the next one on the way."
    return f"{head} (none is reachable from here)."


def _map_neighbours(map_dir: str) -> List[str]:
    """Maps this one leads to by its edges and its warps, whatever lies between: the
    decomp's map links (map.json connections, warp_events), not a walk."""
    from utils.mapping.porymap_state import _map_dirs_by_const

    raw, dirs = _raw_map_json(map_dir), _map_dirs_by_const()
    consts = [c.get("map") for c in raw.get("connections") or []]
    consts += [w.get("dest_map") for w in raw.get("warp_events") or []]
    return [dirs[c] for c in consts if c in dirs]


def _center_of(map_dir: str, names: Dict[str, str]) -> Optional[str]:
    """The readable name of the Pokémon Center 1F a map has a door to, or None. `names`
    maps a map dir to its readable name."""
    target = _squash_name(POKEMON_CENTER_1F)
    for neighbour in _map_neighbours(map_dir):
        name = names.get(neighbour, "")
        if _squash_name(name).endswith(target):
            return name
    return None


def next_pokemon_center(location_name: str, toward: Optional[str]) -> Optional[str]:
    """The Pokémon Center the player comes to next on the way to `toward` (an objective's
    place, matched like go_to's place_route): the first map with one along the fewest
    maps from here to there, counting map links only (what blocks the way now, a ledge
    or water without Surf, is the player's to get past by going on). None without a
    `toward`, when it names no map, or when no Center lies on the way."""
    from collections import deque

    from utils.mapping.porymap_state import _map_dirs_by_const, _readable_map_name
    from utils.state_formatter import ROM_TO_PORYMAP_MAP

    start = ROM_TO_PORYMAP_MAP.get(location_name, "")
    words = _place_words(toward or "")
    if not start or not words:
        return None
    names = {d: _readable_map_name(c) for c, d in _map_dirs_by_const().items()}

    def is_goal(map_dir: str) -> bool:
        return _words_in_order(tuple(_place_words(names.get(map_dir, ""))), tuple(words))

    parent = {start: None}
    queue = deque([start])
    goal = None
    while queue:
        here = queue.popleft()
        if is_goal(here):
            goal = here
            break
        for there in _map_neighbours(here):
            if there not in parent:
                parent[there] = here
                queue.append(there)
    if goal is None:
        return None
    way = []
    while goal is not None:
        way.append(goal)
        goal = parent[goal]
    for map_dir in reversed(way[:-1]):  # the maps after this one, in walking order
        center = _center_of(map_dir, names)
        if center:
            return center
    return None


# ---------------------------------------------------------------------------
# Wild encounter tiles (train)
# ---------------------------------------------------------------------------

WILD_WALKABLE = (".", "~")


def _land_encounter_behaviors() -> frozenset:
    """Behaviors a step on foot rolls a wild Pokémon on: pokeemerald
    MetatileBehavior_IsLandWildEncounter, TILE_FLAG_HAS_ENCOUNTERS without
    TILE_FLAG_SURFABLE (sTileBitAttributes). MB_LONG_GRASS_SOUTH_EDGE and MB_SAND carry no
    encounter flag; a step there rolls nothing."""
    from pokemon_env.enums import MetatileBehavior as MB

    return frozenset(
        int(b)
        for b in (
            MB.TALL_GRASS, MB.LONG_GRASS, MB.UNUSED_05, MB.DEEP_SAND, MB.CAVE,
            MB.INDOOR_ENCOUNTER, MB.ASHGRASS, MB.FOOTPRINTS,
        )
    )


# The ROM's own table of wild Pokémon per map (US Emerald, pret .sym gWildMonHeaders;
# src/wild_encounter.c, src/data/wild_encounters.h). struct WildPokemonHeader {u8 mapGroup,
# mapNum, pad[2]; const struct WildPokemonInfo *landMonsInfo, *waterMonsInfo,
# *rockSmashMonsInfo, *fishingMonsInfo} (20 bytes), ended by MAP_UNDEFINED (group 0xFF).
# struct WildPokemonInfo {u8 encounterRate, pad[3]; const struct WildPokemon *wildPokemon};
# struct WildPokemon {u8 minLevel, maxLevel; u16 species}: 12 land slots, 5 water slots,
# 10 fishing slots.
# StandardWildEncounter rolls on land only when the map's header has landMonsInfo: the
# old "a cave has wild Pokémon" guess knew neither that nor their levels.
G_WILD_MON_HEADERS = 0x08552D48
WILD_HEADER_SIZE = 20
WILD_HEADER_COUNT_MAX = 200  # the table has 124 headers; a bound in case the end is missed
LAND_WILD_COUNT, WATER_WILD_COUNT, FISHING_WILD_COUNT = 12, 5, 10
# The fishing slots each rod draws from (pret ChooseWildMonIndex_Fishing): OLD_ROD 0-1,
# GOOD_ROD 2-4, SUPER_ROD 5-9; the rod is the item's secondaryId (menu_state.rod_index).
FISHING_ROD_SLOTS = {0: (0, 2), 1: (2, 5), 2: (5, 10)}


@dataclass(frozen=True)
class WildArea:
    """One kind of wild encounter on a map (land or water)."""

    rate: int  # encounterRate: WildEncounterCheck rolls rate * 16 in 2880 per step
    min_level: int
    max_level: int
    species: Tuple[int, ...]  # species id per slot

    @property
    def levels(self) -> str:
        return f"Lv{self.min_level}" if self.min_level == self.max_level else f"Lv{self.min_level}-{self.max_level}"


@dataclass(frozen=True)
class MapWild:
    land: Optional[WildArea]
    water: Optional[WildArea]
    # (rate, ((min level, max level, species) per fishing slot)), () without fishingMonsInfo
    fishing: Tuple = ()


_WILD_TABLE: Optional[Dict[str, MapWild]] = None


def _read_wild_slots(reader, pointer: int, slots: int) -> Tuple[int, Tuple[Tuple[int, int, int], ...]]:
    """(encounterRate, ((min level, max level, species) per slot)) of a WildPokemonInfo."""
    info = bytes(reader._read_bytes(pointer, 8))
    mons = bytes(reader._read_bytes(int.from_bytes(info[4:8], "little"), 4 * slots))
    entries = tuple(
        (mons[4 * i], mons[4 * i + 1], int.from_bytes(mons[4 * i + 2:4 * i + 4], "little")) for i in range(slots)
    )
    return info[0], entries


def _wild_area(rate: int, entries) -> WildArea:
    return WildArea(
        rate=rate,
        min_level=min(lo for lo, _, _ in entries),
        max_level=max(hi for _, hi, _ in entries),
        species=tuple(s for _, _, s in entries),
    )


def _read_wild_area(reader, pointer: int, slots: int) -> Optional[WildArea]:
    if not pointer:
        return None
    return _wild_area(*_read_wild_slots(reader, pointer, slots))


def wild_table(reader) -> Dict[str, MapWild]:
    """{map directory: MapWild} for every map in gWildMonHeaders, read from the ROM once.

    `reader` is the emulator's memory reader (call it under the memory lock). Altering
    Cave has 9 headers for one map; GetCurrentMapWildMonHeaderId takes the first plus
    VAR_ALTERING_CAVE_WILD_SET, which only Mystery Gift changes, so the first is kept.
    """
    global _WILD_TABLE
    if _WILD_TABLE is not None:
        return _WILD_TABLE
    from utils.mapping.porymap_state import decomp_map_at

    table: Dict[str, MapWild] = {}
    for i in range(WILD_HEADER_COUNT_MAX):
        header = bytes(reader._read_bytes(G_WILD_MON_HEADERS + i * WILD_HEADER_SIZE, WILD_HEADER_SIZE))
        if header[0] == 0xFF:
            break
        found = decomp_map_at(header[0], header[1])
        if found is None or found[0] in table:
            continue
        land = int.from_bytes(header[4:8], "little")
        water = int.from_bytes(header[8:12], "little")
        fishing = int.from_bytes(header[16:20], "little")
        table[found[0]] = MapWild(
            land=_read_wild_area(reader, land, LAND_WILD_COUNT),
            water=_read_wild_area(reader, water, WATER_WILD_COUNT),
            fishing=_read_wild_slots(reader, fishing, FISHING_WILD_COUNT) if fishing else (),
        )
    _WILD_TABLE = table
    return table


def fishing_wild(map_dir: str, rod: int) -> Optional[WildArea]:
    """What a rod (0 OLD, 1 GOOD, 2 SUPER) hooks on a map, from its fishingMonsInfo; None
    when the map has none (pret DoesCurrentMapHaveFishingMons: "Not even a nibble…" at
    every cast). wild_table must be read first."""
    found = _loaded_wild_table().get(map_dir)
    if found is None or not found.fishing or rod not in FISHING_ROD_SLOTS:
        return None
    rate, entries = found.fishing
    first, end = FISHING_ROD_SLOTS[rod]
    return _wild_area(rate, entries[first:end])


def _loaded_wild_table() -> Dict[str, MapWild]:
    if _WILD_TABLE is None:
        raise RuntimeError("wild_table(reader) must be read before the wild Pokémon of other maps are looked up")
    return _WILD_TABLE


def land_wild(map_dir: str) -> Optional[WildArea]:
    """The land Pokémon of a map (data/maps directory), None when it has none."""
    found = _loaded_wild_table().get(map_dir)
    return found.land if found is not None else None


def encounter_tiles(grid, raw_tiles, land_mons: bool) -> set:
    """(x, y) of every walkable tile where a step on foot can start a wild battle.

    land_mons: the map's wild header has land Pokémon (land_wild); without them no tile
    counts. raw_tiles are (tile id, behavior, collision, elevation) per cell, live ones on
    a map a script redrew (dynamic_map_overlay); without them (an ASCII override map)
    only '~' counts.
    """
    if not land_mons:
        return set()
    behaviors = _land_encounter_behaviors()
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

    {"pairs": [((x, y), (x, y)), ...], "coord_offset": (ox, oy) or None, "map_dir": str,
    "land": WildArea or None}; "pairs" is empty when no tile here can start a battle on
    foot, and "land" is None when the map has no land Pokémon at all (gWildMonHeaders).
    """
    return _tile_pairs(env, lambda grid, raw_tiles, land: encounter_tiles(grid, raw_tiles, land is not None))


def pacing_spots(env) -> dict:
    """The nearest pairs of side-by-side plain tiles on the player's map (walkable, and no
    step there can meet a wild Pokémon), in navigate_to's frame, as wild_encounter_spots
    returns them. rematch_loop steps between them to count the rematch step counter up
    (pret IncrementRematchStepCounter: any step on foot) without wild battles."""

    def plain(grid, raw_tiles, land) -> set:
        walkable = {(x, y) for y, row in enumerate(grid or []) for x, cell in enumerate(row) if cell == "."}
        return walkable - encounter_tiles(grid, raw_tiles, land is not None)

    return _tile_pairs(env, plain)


def _tile_pairs(env, pick_tiles) -> dict:
    """wild_encounter_spots and pacing_spots: pairs of side-by-side tiles among
    pick_tiles(grid, raw_tiles, land WildArea or None), nearest first, kept only when the
    pathfinder reaches them."""
    from utils.mapping.porymap_state import decomp_map_at

    reader = env.memory_reader
    table = wild_table(reader)
    # The map by its live group and number, as GetCurrentMapWildMonHeaderId looks it up.
    found = decomp_map_at(*reader._read_current_map())
    land = table[found[0]].land if found is not None and found[0] in table else None
    state, coord_offset, source = load_pathfinding_state(env)
    porymap = (state.get("map") or {}).get("porymap") or {}
    grid = porymap.get("grid") or []
    position = (state.get("player") or {}).get("position") or {}
    ox, oy = coord_offset or (0, 0)
    start = (int(position.get("x", 0)) - ox, int(position.get("y", 0)) - oy)
    map_dir = found[0] if found is not None else source.get("porymap_map_name", "")
    tiles = pick_tiles(grid, porymap.get("raw_tiles"), land)
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
        "land": land,
    }


@lru_cache(maxsize=None)
def map_has_wild_encounters(map_dir: str) -> bool:
    """Whether a map (data/maps directory) has land Pokémon (gWildMonHeaders) and tiles a
    step on foot rolls them on, in its map file's layout. wild_table must be read first."""
    from pokemon_env.porymap_paths import get_porymap_root
    from utils.mapping.porymap_json_builder import build_json_map

    root = get_porymap_root()
    if not root or not map_dir or land_wild(map_dir) is None:
        return False
    try:
        built = build_json_map(map_dir, root, include_grid=True, include_ascii=False) or {}
    except Exception as e:  # a broken map must not take the hint down with it
        logger.warning(f"Could not read {map_dir} for wild encounters: {e}")
        return False
    return bool(encounter_tiles(built.get("grid"), built.get("raw_tiles"), True))


@lru_cache(maxsize=1)
def _map_dirs_by_name() -> dict:
    """'ROUTE 116' -> 'Route116', for every map the decomp has."""
    from utils.mapping.porymap_state import _map_dirs_by_const, _readable_map_name

    return {_readable_map_name(const): map_dir for const, map_dir in _map_dirs_by_const().items()}


def way_to_wild_encounters(location_name: str, position: Optional[Tuple[int, int]]) -> str:
    """The way on foot to the nearest map with wild Pokémon on land, and their levels.
    wild_table must be read first."""
    from utils.state_formatter import ROM_TO_PORYMAP_MAP

    paths = _walk_paths(
        ROM_TO_PORYMAP_MAP.get(location_name, ""),
        lambda name: map_has_wild_encounters(_map_dirs_by_name().get(name, "")),
        position,
        other_parts_of_here=True,
    )
    if not paths:
        return "no map with wild Pokémon on land can be reached on foot from here."
    # Equally near, the player's own map wins: its Pokémon are the ones an objective
    # there wants (Marill on Route 117, not Rusturf Tunnel's Whismur).
    path = min(paths, key=lambda p: (len(p), p[-1][1] != location_name))
    place = path[-1][1]
    land = land_wild(_map_dirs_by_name().get(place, ""))
    levels = f" ({land.levels})" if land is not None else ""
    if len(path) == 1:
        return f'{place} has wild Pokémon{levels}: go_to_exit("{path[0][0]}"), then call train again.'
    way = " → ".join(name for _, name in path)
    return (
        f"the nearest wild Pokémon are on {place}{levels}; the way on foot: {way}. "
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
    words = _target_words(target)
    if not words:
        return ""

    def matches(name: str) -> bool:
        return _words_in_order(tuple(_place_words(name)), tuple(words))

    if matches(location_name):
        return ""  # on the named map already (e.g. the other half of a route)
    map_dir = ROM_TO_PORYMAP_MAP.get(location_name, "")
    # A map named exactly as the target wins. Seen live: "Route 116" matched ROUTE 116
    # TUNNELERS REST HOUSE too, and the hint (which shows the farthest match per first
    # exit) routed into that house instead of onto the route.
    exact = _far_map_hint(map_dir, lambda name: _same_place_words(name, words), position, nearest=True)
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
    words = _target_words(target)
    if not words:
        return None

    def matches(name: str) -> bool:
        return _words_in_order(tuple(_place_words(name)), tuple(words))

    map_dir = ROM_TO_PORYMAP_MAP.get(location_name, "")
    for matcher in (lambda name: _same_place_words(name, words), matches):
        # go_to walks this, so the Cable Car is a step it can take (RIDE_EXIT).
        paths = _walk_paths_or_fall(map_dir, matcher, position, rides=True)
        if paths:
            return paths[0]
    return None


# ---------------------------------------------------------------------------
# Fly (go_to, fly_to)
# ---------------------------------------------------------------------------

# Where Fly works: pret src/party_menu.c SetUpFieldMove_Fly allows it only when
# Overworld_MapTypeAllowsTeleportAndFly(gMapHeader.mapType): routes, towns, cities and
# ocean routes, never in a building, cave or underwater. The server reads the live map
# type from gMapHeader; the planner reads the map's map.json.
FLY_MAP_TYPES = ("MAP_TYPE_TOWN", "MAP_TYPE_CITY", "MAP_TYPE_ROUTE", "MAP_TYPE_OCEAN_ROUTE")
# A flight's cost in maps walked: START → POKéMON → FLY → the region map → the flight
# and the landing, ~1200 game frames from Lavaridge to Mauville in local_tests/test_fly.py.
# Counted as two maps (a judgement, not measured against walks), a flight is taken only
# when it saves three maps of walking or more, or the walk can't be found at all.
FLY_COST_MAPS = 2

# ROM tables the fly map works from (US Emerald, pret .sym; src/region_map.c).
# gRegionMapEntries[mapsec]: struct RegionMapLocation {u8 x, y, width, height; const u8
# *name}; the cursor's cell is that plus (MAPCURSOR_X_MIN, MAPCURSOR_Y_MIN)
# (InitMapBasedOnPlayerLocation). sMapHealLocations[mapsec]: {map group, map num, heal
# location id}; CB_ExitFlyMap warps to that heal location, sHealLocations[id - 1]
# (src/data/heal_locations.h: {s8 group, s8 num, s16 x, s16 y}).
G_REGION_MAP_ENTRIES = 0x085A147C
REGION_MAP_ENTRY_SIZE = 8
S_MAP_HEAL_LOCATIONS = 0x085A1E3C
MAP_HEAL_LOCATION_SIZE = 3
S_HEAL_LOCATIONS = 0x0859F53C
HEAL_LOCATION_SIZE = 8
MAPCURSOR_X_MIN, MAPCURSOR_Y_MIN = 1, 2
# MAPSEC_LITTLEROOT_TOWN (0) .. MAPSEC_EVER_GRANDE_CITY (15): the towns with a
# FLAG_VISITED_* flag, which GetMapsecType makes MAPSECTYPE_CITY_CANFLY once it is set.
TOWN_MAPSECS = 16
MAPSEC_LITTLEROOT_TOWN, MAPSEC_EVER_GRANDE_CITY = 0, 15
# CB_ExitFlyMap's two exceptions to sMapHealLocations. Littleroot lands outside the
# player's own house: HEAL_LOCATION_LITTLEROOT_TOWN_BRENDANS_HOUSE or _MAYS_HOUSE by
# gSaveBlock2Ptr->playerGender (u8 at +8, after the 8-byte name). Ever Grande lands at the
# Pokémon League from the mapsec's top cell (posWithinMapSec 0) once
# FLAG_LANDMARK_POKEMON_LEAGUE is set, so its bottom cell, the city's own Pokémon Center,
# is the one used.
HEAL_LOCATION_BRENDANS_HOUSE, HEAL_LOCATION_MAYS_HOUSE = 12, 13
G_SAVE_BLOCK2_PTR = 0x03005D90
SAVE_BLOCK2_PLAYER_GENDER = 8
FEMALE = 1


@dataclass(frozen=True)
class FlyDestination:
    town: str                 # the town's map name, as read_location gives it (LILYCOVE CITY)
    mapsec: int
    cell: Tuple[int, int]     # the fly map cursor cell (cursorPosX, cursorPosY) to press A on
    landing_map: str          # the map the flight lands on
    landing: Tuple[int, int]  # the tile it lands on
    visited: bool             # FLAG_VISITED_<town>: the fly map only takes A on a visited town


def _map_name(group: int, num: int) -> str:
    from pokemon_env.enums import MapLocation

    try:
        return MapLocation((group << 8) | num).name.replace("_", " ")
    except ValueError:
        return ""


def fly_destinations(reader) -> List[FlyDestination]:
    """Every town on the fly map with its cursor cell and landing tile, from the ROM tables
    (so no cell or landing is typed in here), and whether it was visited, from the save's
    flags. `reader` is the emulator's memory reader."""
    entries = bytes(reader._read_bytes(G_REGION_MAP_ENTRIES, REGION_MAP_ENTRY_SIZE * TOWN_MAPSECS))
    heals = bytes(reader._read_bytes(S_MAP_HEAL_LOCATIONS, MAP_HEAL_LOCATION_SIZE * TOWN_MAPSECS))
    visited_flags = _first_visit_flags()
    out = []
    for mapsec in range(TOWN_MAPSECS):
        x, y, _, height = entries[mapsec * REGION_MAP_ENTRY_SIZE:mapsec * REGION_MAP_ENTRY_SIZE + 4]
        group, num, heal_id = heals[mapsec * MAP_HEAL_LOCATION_SIZE:(mapsec + 1) * MAP_HEAL_LOCATION_SIZE]
        town = _map_name(group, num)
        row = y + height - 1 if mapsec == MAPSEC_EVER_GRANDE_CITY else y
        if mapsec == MAPSEC_LITTLEROOT_TOWN:
            save2 = reader._read_u32(G_SAVE_BLOCK2_PTR)
            female = reader._read_u8(save2 + SAVE_BLOCK2_PLAYER_GENDER) == FEMALE
            heal_id = HEAL_LOCATION_MAYS_HOUSE if female else HEAL_LOCATION_BRENDANS_HOUSE
        heal = bytes(reader._read_bytes(S_HEAL_LOCATIONS + (heal_id - 1) * HEAL_LOCATION_SIZE, HEAL_LOCATION_SIZE))
        flag = visited_flags.get(town)
        out.append(
            FlyDestination(
                town=town,
                mapsec=mapsec,
                cell=(x + MAPCURSOR_X_MIN, row + MAPCURSOR_Y_MIN),
                landing_map=_map_name(heal[0], heal[1]),
                landing=(int.from_bytes(heal[2:4], "little", signed=True), int.from_bytes(heal[4:6], "little", signed=True)),
                visited=bool(flag) and bool(reader._is_event_flag_set(flag)),
            )
        )
    return out


def fly_destination_named(destinations: List[FlyDestination], name: str) -> Optional[FlyDestination]:
    """The town `name` names: the town itself ("Lilycove", "LILYCOVE CITY") or a place in
    it ("Lilycove City Pokemon Center")."""
    wanted = _squash_name(name)
    if not wanted:
        return None
    for d in destinations:
        town = _squash_name(d.town)
        if town.startswith(wanted) or wanted.startswith(town) or _names_same_place(d.town, name):
            return d
    return None


@lru_cache(maxsize=None)
def visit_triggers(map_dir: str, flag: str) -> Tuple[bool, Tuple[Tuple[int, int], ...]]:
    """How the game sets a town's FLAG_VISITED_* flag: (a map script sets it on arrival,
    the coord_events tiles whose script sets it).

    In pret most towns `setflag FLAG_VISITED_<town>` in their OnTransition map script
    (DewfordTown_OnTransition), some through a call (SlateportCity_OnTransition →
    SlateportCity_EventScript_EnterSlateport). Mossdeep and Ever Grande set it only from
    coord events: MossdeepCity_EventScript_VisitedMossdeep at (25,25), (26,25), (32,27)
    and (33,27). Seen live: MOSSDEEP CITY was refused by fly_to after the player came in
    from Route 124's west edge, past none of those tiles. Map scripts that run in tables
    (MAP_SCRIPT_ON_FRAME_TABLE, ..._WARP_INTO_MAP_TABLE) run only when a var matches, so
    only the plain ones (ON_TRANSITION, ON_LOAD, ON_RESUME, ...) count as arriving.
    """

    def sets(label: str) -> bool:
        # _script_lines follows the label's calls, gotos and fall-throughs.
        return any(line.split() == ["setflag", flag] for line in _script_lines(map_dir, label))

    on_arrival = False
    for line in _script_labels(map_dir)[0].get(f"{map_dir}_MapScripts", ()):
        command, _, args = line.partition(" ")
        kind, _, script = args.partition(",")
        if command == "map_script" and not kind.strip().endswith("_TABLE") and sets(script.strip()):
            on_arrival = True
    tiles = []
    for event in _raw_map_json(map_dir).get("coord_events") or []:
        tile = (int(event.get("x", -1)), int(event.get("y", -1)))
        if event.get("type") == "trigger" and tile not in tiles and sets(str(event.get("script", ""))):
            tiles.append(tile)
    return on_arrival, tuple(sorted(tiles, key=lambda t: (t[1], t[0])))  # row by row, as on screen


def visit_hint(town: str) -> str:
    """A sentence on what makes fly_to's `town` count as visited, from its map's scripts
    (visit_triggers), or "" when nothing there sets its FLAG_VISITED_* flag."""
    from utils.state_formatter import ROM_TO_PORYMAP_MAP

    on_arrival, tiles = visit_triggers(ROM_TO_PORYMAP_MAP.get(town, ""), "FLAG_VISITED_" + town.replace(" ", "_"))
    ways = ["as soon as you arrive there"] if on_arrival else []
    if tiles:
        cells = [f"({x},{y})" for x, y in tiles]
        listed = cells[0] if len(cells) == 1 else ", ".join(cells[:-1]) + " or " + cells[-1]
        ways.append(f"once you walk over {listed} there")
    if not ways:
        return ""
    return f"{town} counts as visited {', or '.join(ways)}."


def can_fly() -> bool:
    """A party Pokémon (not an egg) knows FLY and the Feather Badge is set."""
    return FLY in _usable_field_moves()


def map_allows_fly(map_dir: str) -> bool:
    return _raw_map_json(map_dir).get("map_type") in FLY_MAP_TYPES


def map_allows_escaping(map_dir: str) -> bool:
    """The map header's allowEscaping (map.json allow_escaping): pret src/item_use.c
    CanUseDigOrEscapeRopeOnCurMap lets an Escape Rope (or Dig) work only there, in caves
    and dungeons."""
    return bool(_raw_map_json(map_dir).get("allow_escaping"))


def fly_plan(
    location_name: str, position: Optional[Tuple[int, int]], target: str, destinations: List[FlyDestination]
) -> Optional[Tuple[FlyDestination, List[Tuple[str, str]]]]:
    """(town to fly to, the walk from where it lands) when flying there and walking on
    reaches `target` clearly sooner than walking all the way, else None.

    David (2026-11-04): "use TROPIUS + Fly instead of long walks". Both ways are counted
    in maps walked (place_route), a flight as FLY_COST_MAPS more. A walk place_route can't
    find at all (water without Surf, the Route 111 desert without the Go-Goggles, or more
    than _FAR_MAP_MAX_STEPS maps) loses to any flight that gets there. Only visited towns
    count (the fly map refuses the others), and only when the party can fly from here.
    """
    from utils.state_formatter import ROM_TO_PORYMAP_MAP

    usable = [d for d in destinations if d.visited and d.landing_map and d.landing_map != location_name]
    if not usable or not can_fly() or not map_allows_fly(ROM_TO_PORYMAP_MAP.get(location_name, "")):
        return None
    walk = place_route(location_name, position, target)
    if walk is not None and len(walk) <= FLY_COST_MAPS:
        return None  # no flight can beat a walk this short
    best = None
    for d in usable:
        rest = place_route(d.landing_map, d.landing, target)
        if rest is not None and (best is None or len(rest) < len(best[1])):
            best = (d, rest)
    if best is None or (walk is not None and len(best[1]) + FLY_COST_MAPS >= len(walk)):
        return None
    return best


def note_live_layout(reader) -> None:
    """Tell the route planner which layout the player's map is drawn from now
    (porymap_state.note_layout; dynamic_map_overlay.swapped_layout reads it from the save),
    and forget the areas planned from another one. Caller holds the emulator lock."""
    from utils.mapping.dynamic_map_overlay import swapped_layout
    from utils.mapping.porymap_state import decomp_map_at, note_layout

    found = decomp_map_at(*reader._read_current_map())
    if found is None:
        return
    if note_layout(found[0], (swapped_layout(reader) or {}).get("id", "")):
        for cached in (_walk_areas_without, _edge_dead_ends_with, _ride_exits_with, _fall_exits_with, _puzzle_plan):
            cached.cache_clear()


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
    lands on when it leads to another spot of this same map, else None. Every plan has
    "to", the name of the map the exit leads to (the gym level gate reads it).

    avoid: map tiles (raw map coordinates) to treat as blocked, e.g. where a person
    stopped the last walk.
    """
    from server.field_control import (
        DIRECTION_DELTAS,
        EDGE_BUTTON,
        edge_candidates,
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
    surf = bool(state.get("map", {}).get("porymap", {}).get("surf"))
    # The boulder pushes of each route found (push_plan), by its end tile.
    pushes_to: dict = {}
    # The story scene each route found walks into to clear the way (pathfinding
    # _through_scene), by its end tile.
    scenes_to: dict = {}

    def route_to(tile):
        if tile == start:
            return []
        buttons = pathfinder.find_path(
            start, tile, state, consider_npcs=True, allow_partial=False, blocked_coords=blocked
        )
        pushes = list(pathfinder.last_pushes)
        if buttons and planned_end(start, buttons, pushes) == tile:
            pushes_to[tile] = pushes
            scenes_to[tile] = pathfinder.last_scene
            return buttons
        return None

    best = None
    tried = []
    # The tiles each matched exit is taken from. When no plain walk reaches one on a
    # rotating tile puzzle map, a failed plan hands them back ("puzzle_targets"), and
    # go_to_exit walks through the floor switches to them (server/app.py _cross_to).
    targets: list = []
    for exit_entry in matches:
        if exit_entry["kind"] == "edge":
            candidates = edge_candidates(grid, exit_entry["direction"], surf=surf)
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
            targets.extend(candidates[:max_edge_candidates])
            for tile in candidates[:max_edge_candidates]:
                buttons = route_to(tile)
                if buttons is not None:
                    if best is None or len(buttons) < len(best["buttons"]):
                        best = {
                            "kind": "edge",
                            "label": exit_entry["label"],
                            "to": exit_entry["name"],
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
                    targets.append(front)
                    buttons = route_to(front)
                    if buttons is not None:
                        fronts.append((len(buttons), front, button, buttons))
                if not fronts:
                    tried.append(f"{exit_entry['label']}: no walkable path to a tile in front of the door at {tile}")
                    continue
                _, front, button, buttons = min(fronts)
                way.update({"kind": "sign_door", "target": front, "face": button})
            else:
                targets.append(tile)
                buttons = route_to(tile)
                if buttons is None:
                    tried.append(f"{exit_entry['label']}: no walkable path to the warp at {tile}")
                    continue
                way.update({"kind": "warp", "target": tile})
            if best is None or len(buttons) < len(best["buttons"]):
                best = {**way, "label": exit_entry["label"], "to": exit_entry["name"], "buttons": buttons, "final_button": None}

    if best is None:
        # A plain walk never steps on a puzzle's floor switch (find_path): a press moves the
        # statues. go_to_exit plans the presses itself from the targets (puzzle_route).
        puzzle = bool(puzzle_switches(source.get("porymap_map_name") or ""))
        failed = {
            "success": False,
            "error": (
                f"Cannot reach '{exit_query}' from ({pos.get('x')}, {pos.get('y')}) on {location}: "
                + "; ".join(tried)
                # With Surf the plan crosses water already: water is no reason then.
                + (". A ledge (one-way) or a person may be in the way." if surf
                   else ". A ledge (one-way), water (needs Surf) or a person may be in the way.")
                + (" This map's floor switches move its statues, and no way through them was found either."
                   if puzzle else "")
            ),
        }
        # A floor that drops the walker to the one below (hole_warp): the exit may be
        # reached through the floor below and back up (tile_route with falls, _fall_exits).
        from utils.mapping.porymap_state import hole_warp

        drops = bool(hole_warp(source.get("porymap_map_name") or "")[0])
        if (puzzle or drops) and targets:
            failed["puzzle_targets"] = list(dict.fromkeys(targets))
            failed["crossing"] = "through the floor switches" if puzzle else "by a drop to the floor below"
        return failed
    # A route that pushes boulders is walked to its first push only (push_plan): the
    # caller pushes, then plans again. Its edge press is kept for the last plan.
    best.update(push_plan(start, best["buttons"], pushes_to.get(tuple(best["target"])) or []))
    if scenes_to.get(tuple(best["target"])) is not None:
        # The walk stops in a story scene that clears the way; the caller plans again.
        best["scene"] = list(scenes_to[tuple(best["target"])])
    raw_tiles = state.get("map", {}).get("porymap", {}).get("raw_tiles")
    best["buttons"] = jump_presses(start, best["buttons"], raw_tiles)
    best.update({
        "success": True,
        "grid": grid,
        "raw_tiles": raw_tiles,
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
