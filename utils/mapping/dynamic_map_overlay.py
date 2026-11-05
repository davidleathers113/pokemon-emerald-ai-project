#!/usr/bin/env python3
"""
Dynamic map overlay helpers for runtime-changing maps.

For selected locations, and for any map a script has redrawn from another layout
(Shoal Cave's tides), this module reads LIVE metatiles from emulator memory
and rebuilds the pathfinding grid so dynamic collision changes (e.g. gym
barriers) are reflected in navigation.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from utils.mapping.map_formatter import format_tile_to_symbol

logger = logging.getLogger(__name__)

# Expand this allowlist as additional dynamic maps are supported.
DYNAMIC_MAP_LOCATIONS = {
    "MAUVILLE CITY GYM",
    # Rusturf Tunnel's scripts rewrite its metatiles as the story opens the tunnel.
    # Seen live (2026-11-03): at (6,10) the static grid showed (6,11) as open floor but
    # the game blocked the step, and navigate_to failed 5 times in a row.
    "RUSTURF TUNNEL",
    # "FORTREE CITY GYM",
}

# GBA map buffers include a 7-tile border around the real map.
GBA_MAP_BORDER_OFFSET = 7


def is_dynamic_location(location_name: str) -> bool:
    """Return True if the given location needs live metatile overlays."""
    return bool(location_name) and location_name.upper() in DYNAMIC_MAP_LOCATIONS


# pret overworld.c: a map script's setmaplayoutindex (SetCurrentMapLayout) puts another
# layout in gSaveBlock1Ptr->mapLayoutId (SaveBlock1 +0x32), and the map is drawn and walked
# from gMapLayouts[mapLayoutId - 1] (GetMapLayout). The map itself (group/num, its scripts,
# its wild Pokémon) stays the same. Seen live (2026-11-05): SHOAL CAVE LOW TIDE ENTRANCE ROOM
# at high tide is drawn from LAYOUT_SHOAL_CAVE_HIGH_TIDE_ENTRANCE_ROOM (its OnTransition
# script). The map file's low-tide cave floor at (18,16) and (14,24) was shallow water, and
# train stepped there 300 times without a wild battle.
SAVE_BLOCK1_MAP_LAYOUT_ID = 0x32


@lru_cache(maxsize=1)
def _layouts() -> Tuple[Dict[str, Any], ...]:
    """data/layouts/layouts.json's layouts in file order: the LAYOUT_* constants are
    generated from it, so layout id N is entry N - 1."""
    from pokemon_env.porymap_paths import get_porymap_root

    root = get_porymap_root()
    if not root:
        return ()
    try:
        with open(root / "data" / "layouts" / "layouts.json", encoding="utf-8") as fh:
            return tuple(json.load(fh).get("layouts") or ())
    except (OSError, ValueError) as e:
        logger.warning("layouts.json unreadable: %s", e)
        return ()


@lru_cache(maxsize=None)
def _map_file_layout(map_dir: str) -> str:
    """The layout a map's map.json names, e.g. 'LAYOUT_SHOAL_CAVE_LOW_TIDE_ENTRANCE_ROOM'."""
    from pokemon_env.porymap_paths import get_porymap_root

    root = get_porymap_root()
    if not root or not map_dir:
        return ""
    try:
        with open(root / "data" / "maps" / map_dir / "map.json", encoding="utf-8") as fh:
            return str(json.load(fh).get("layout") or "")
    except (OSError, ValueError):
        return ""


def swapped_layout(memory_reader: Any) -> Optional[Dict[str, Any]]:
    """The layouts.json entry the player's map is drawn from now, when a script swapped it
    for another than the map file's; None when it is the map file's own (or unreadable)."""
    from utils.mapping.porymap_state import decomp_map_at

    if memory_reader is None:
        return None
    try:
        group, num = memory_reader._read_current_map()
        save_block1 = memory_reader._read_u32(memory_reader.addresses.SAVE_BLOCK1_PTR)
        layout_id = memory_reader._read_u16(save_block1 + SAVE_BLOCK1_MAP_LAYOUT_ID)
    except Exception as e:  # no swap known: the map file's tiles stay
        logger.debug("live layout id unreadable: %s", e)
        return None
    found = decomp_map_at(group, num)
    layouts = _layouts()
    if found is None or not 1 <= layout_id <= len(layouts):
        return None
    live = layouts[layout_id - 1]
    own = _map_file_layout(found[0])
    if not own or live.get("id") == own:
        return None
    return live


def _live_dimensions(memory_reader: Any, width: int, height: int) -> Tuple[int, int, bool]:
    """(width, height, swapped): a swapped layout's own size, else the map file's."""
    live = swapped_layout(memory_reader)
    if live is None:
        return width, height, False
    return int(live.get("width") or width), int(live.get("height") or height), True


def _read_live_tiles(memory_reader: Any, width: int, height: int, location_name: str) -> List[List[Any]] | None:
    """Read live metatiles from the emulator, returning None on failure."""
    if memory_reader is None:
        logger.warning("Dynamic overlay skipped for %s: memory_reader unavailable", location_name)
        return None

    if not getattr(memory_reader, "_map_buffer_addr", None):
        find_buffer_fn = getattr(memory_reader, "_find_map_buffer_addresses", None)
        if not callable(find_buffer_fn) or not find_buffer_fn():
            logger.warning("Dynamic overlay skipped for %s: failed to locate map buffer", location_name)
            return None

    live_raw_tiles = memory_reader.read_map_metatiles(
        x_start=GBA_MAP_BORDER_OFFSET,
        y_start=GBA_MAP_BORDER_OFFSET,
        width=width,
        height=height,
    )
    if not live_raw_tiles:
        logger.warning("Dynamic overlay skipped for %s: live metatile read returned empty", location_name)
        return None
    return live_raw_tiles


def _build_live_grid(live_raw_tiles: List[List[Any]], location_name: str) -> List[List[str]]:
    """Convert raw metatile data into an ASCII symbol grid."""
    live_grid: List[List[str]] = []
    for y, row in enumerate(live_raw_tiles):
        grid_row: List[str] = []
        for x, tile in enumerate(row):
            grid_row.append(format_tile_to_symbol(tile, x=x, y=y, location_name=location_name))
        live_grid.append(grid_row)
    return live_grid


# ------------------------------------------------------------------
# Public API: overlay on a json_map dict (used inside _format_porymap_info)
# ------------------------------------------------------------------

def apply_live_overlay_to_json_map(json_map: Dict[str, Any], memory_reader: Any, location_name: str) -> bool:
    """Replace grid/raw_tiles/ascii in a json_map dict with live emulator data.

    This is the preferred entry-point for code paths that work with the
    ``json_map`` dictionary produced by ``build_json_map_for_llm``.

    Returns True if the overlay was applied. Applied on the allowlisted maps, and on any
    map a script has redrawn from another layout (swapped_layout).
    """
    dims = json_map.get("dimensions") or {}
    width, height, swapped = _live_dimensions(
        memory_reader, int(dims.get("width", 0) or 0), int(dims.get("height", 0) or 0)
    )
    if not (swapped or is_dynamic_location(location_name)):
        return False
    if width <= 0 or height <= 0:
        logger.warning("Dynamic overlay skipped for %s: missing json_map dimensions", location_name)
        return False

    live_raw_tiles = _read_live_tiles(memory_reader, width, height, location_name)
    if live_raw_tiles is None:
        return False

    live_grid = _build_live_grid(live_raw_tiles, location_name)

    json_map["raw_tiles"] = live_raw_tiles
    json_map["grid"] = live_grid
    json_map["ascii"] = "\n".join("".join(row) for row in live_grid)
    json_map["dimensions"] = {"width": width, "height": height}

    logger.info("Dynamic overlay applied to json_map for %s (%sx%s)", location_name, width, height)
    return True


# ------------------------------------------------------------------
# Public API: overlay on the full game state dict (used in navigate_to)
# ------------------------------------------------------------------

def apply_live_metatile_overlay(state: Dict[str, Any], env: Any, location_name: str) -> bool:
    """Replace porymap grid/raw_tiles in the game *state* dict with live data.

    Returns True if overlay was applied, otherwise False. Applied on the allowlisted maps,
    and on any map a script has redrawn from another layout (swapped_layout).
    """
    porymap = state.get("map", {}).get("porymap", {})
    dimensions = porymap.get("dimensions", {}) or {}
    memory_reader = getattr(env, "memory_reader", None)
    width, height, swapped = _live_dimensions(
        memory_reader, int(dimensions.get("width", 0) or 0), int(dimensions.get("height", 0) or 0)
    )
    if not (swapped or is_dynamic_location(location_name)):
        return False
    if width <= 0 or height <= 0:
        logger.warning("Dynamic overlay skipped for %s: missing porymap dimensions", location_name)
        return False

    live_raw_tiles = _read_live_tiles(memory_reader, width, height, location_name)
    if live_raw_tiles is None:
        return False

    live_grid = _build_live_grid(live_raw_tiles, location_name)

    state.setdefault("map", {}).setdefault("porymap", {})
    state["map"]["porymap"]["raw_tiles"] = live_raw_tiles
    state["map"]["porymap"]["grid"] = live_grid
    state["map"]["porymap"]["dimensions"] = {"width": width, "height": height}

    logger.info(
        "Dynamic overlay applied for %s using live metatiles (%sx%s)",
        location_name,
        width,
        height,
    )
    return True
