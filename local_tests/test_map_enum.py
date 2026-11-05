"""Every map is named and numbered as the ROM numbers it (pret's map_groups.json).

Seen live: entering MT PYRE 1F (map group 0x18, number 0x0F) the state read "Map_18_0F",
with no ASCII map and no exits, and go_to failed. pokemon_env.enums.MapLocation was a
hand-written table with 217 wrong values and 64 maps missing (239 of the 518 map ids had
no member); some caves only read right through _ALT aliases ("RUSTURF TUNNEL ALT") and a
fix-up in read_location. MapLocation is now generated from map_groups.json; these tests
fail if it drifts again, or if a location name stops leading to its porymap map.

  docker run --rm --entrypoint /opt/venv/bin/python -v "$PWD":/app \
      -v ~/continual-harness/Emerald-GBAdvance:/app/Emerald-GBAdvance:ro -w /app \
      emerald-harness:local local_tests/test_map_enum.py
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent
MAPS = REPO / "pokemon_env" / "porymap" / "data" / "maps"
ROM = REPO / "Emerald-GBAdvance" / "rom.gba"


def _pret_maps():
    """[(group, num, data/maps dir, MAP_ constant)] in the ROM's order (map_groups.json)."""
    groups = json.loads((MAPS / "map_groups.json").read_text())
    out = []
    for group, group_name in enumerate(groups["group_order"]):
        for num, map_dir in enumerate(groups[group_name]):
            const = json.loads((MAPS / map_dir / "map.json").read_text())["id"]
            out.append((group, num, map_dir, const))
    return out


def test_enum_is_map_groups_json():
    from pokemon_env.enums import MapLocation

    maps = _pret_maps()
    assert len(maps) == 518, len(maps)
    # No aliases: two names for one id would make MapLocation(id) pick one silently.
    assert len(MapLocation) == len(MapLocation.__members__) == len(maps), (
        len(MapLocation), len(MapLocation.__members__))
    for group, num, _, const in maps:
        location = MapLocation((group << 8) | num)
        # The name is pret's constant; only ROUTE<n> is written ROUTE_<n>.
        assert location.name.replace("ROUTE_", "ROUTE") == const[len("MAP_"):], (location, const)


def test_location_name_is_the_exit_name_and_finds_its_map():
    """read_location's name, the name exits are listed under and ROM_TO_PORYMAP_MAP's key
    must be one string per map: GRANITE CAVE B1F was listed as an exit "GRANITE CAVE B 1F"."""
    from pokemon_env.enums import MapLocation
    from utils.mapping.porymap_state import ROM_TO_PORYMAP_MAP, _readable_map_name, decomp_map_at

    for group, num, map_dir, const in _pret_maps():
        name = MapLocation((group << 8) | num).name.replace("_", " ")
        assert _readable_map_name(const) == name, (const, name)
        assert decomp_map_at(group, num) == (map_dir, name), (group, num)
        assert ROM_TO_PORYMAP_MAP.get(name) == map_dir, (name, ROM_TO_PORYMAP_MAP.get(name), map_dir)
    names = {MapLocation(v).name.replace("_", " ") for v in MapLocation}
    # Every other key is a name for a map that isn't its ROM name; none may be a stale alias.
    extra = {k: v for k, v in ROM_TO_PORYMAP_MAP.items() if k not in names}
    assert extra == {
        "MOVING_VAN": "InsideOfTruck",  # read_location's name for MAP_INSIDE_OF_TRUCK
        "PROFESSOR BIRCHS LAB": "LittlerootTown_ProfessorBirchsLab",
        "FIERY PATH INTERIOR": "FieryPath",
    }, extra


def test_mt_pyre_1f_reads_by_name_with_its_map_and_exits():
    """No savestate reaches Mt. Pyre, so a real one (Rusturf Tunnel) is loaded and the map
    read from memory is made (0x18, 0x0F): the id that read as Map_18_0F live."""
    from utils.mapping.porymap_state import ROM_TO_PORYMAP_MAP, _format_porymap_info

    reader = _emulator(HERE / "test_rusturf_tunnel.state").memory_reader
    real = reader._read_current_map
    try:
        reader._read_current_map = lambda: (0x18, 0x0F)
        location = reader.read_location()
    finally:
        reader._read_current_map = real
    assert location == "MT PYRE 1F", location
    assert ROM_TO_PORYMAP_MAP[location] == "MtPyre_1F"
    # (17, 17): just inside the door to Route 122 (warps at (17, 18)/(18, 18) in map.json).
    text = "\n".join(_format_porymap_info(location, (17, 17)).context_parts)
    assert "ASCII Map:" in text and "EXITS (where each way out" in text, text[:500]
    for leads_to in ("ROUTE 122", "MT PYRE EXTERIOR", "MT PYRE 2F"):
        assert f"-> {leads_to}" in text, (leads_to, text)


def test_every_map_id_reads_a_name_that_has_a_map():
    """All 518 ids through read_location itself: none may come back Map_XX_YY."""
    from utils.mapping.porymap_state import ROM_TO_PORYMAP_MAP

    reader = _emulator(HERE / "test_rusturf_tunnel.state").memory_reader
    real = reader._read_current_map
    try:
        for group, num, map_dir, _ in _pret_maps():
            reader._read_current_map = lambda g=group, n=num: (g, n)
            location = reader.read_location()
            assert ROM_TO_PORYMAP_MAP.get(location) == map_dir, (group, num, location, map_dir)
    finally:
        reader._read_current_map = real


def test_cave_savestates_read_their_map_with_ascii_and_exits():
    """Real savestates in caves. Rusturf Tunnel used to read RUSTURF TUNNEL ALT."""
    from utils.mapping.dynamic_map_overlay import is_dynamic_location
    from utils.mapping.porymap_state import _format_porymap_info

    for state, expected in (
        ("test_rusturf_tunnel.state", "RUSTURF TUNNEL"),
        ("test_meteor_falls_before_scene.state", "METEOR FALLS 1F 1R"),
        ("test_mt_chimney_busy_grunt.state", "MT CHIMNEY"),
        ("test_jagged_pass_top.state", "JAGGED PASS"),
    ):
        reader = _emulator(HERE / state).memory_reader
        location = reader.read_location()
        assert location == expected, (state, location)
        text = "\n".join(_format_porymap_info(location, reader.read_coordinates()).context_parts)
        assert "ASCII Map:" in text and "EXITS (where each way out" in text, (state, text[:500])
    # Its live-tile overlay is keyed by the location name.
    assert is_dynamic_location("RUSTURF TUNNEL")


_EMU = None


def _emulator(state):
    global _EMU
    if _EMU is None:
        from pokemon_env.emulator import EmeraldEmulator

        os.chdir(tempfile.mkdtemp())  # the emulator writes milestone/map files into cwd
        _EMU = EmeraldEmulator(str(ROM), headless=True, sound=False)
        _EMU.initialize()
    _EMU.load_state(str(state))
    _EMU.tick(5)
    return _EMU


def _run_all():
    failures = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        started = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - started:.1f}s)")
        except Exception as e:  # report every failure, not just the first
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
