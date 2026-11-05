"""Rotating gates (FORTREE CITY GYM, the Trick House's sixth puzzle), modelled from pret
pokeemerald src/rotating_gate.c so walks can plan through them.

A gate is not on the map grid: it is a sprite whose arms lie on the grid lines through a
pivot point, the corner shared by tiles (x-1, y-1), (x, y-1), (x-1, y) and (x, y), where
(x, y) is the gate's entry in the puzzle config. A step whose way crosses an arm pushes
that arm: the gate turns a quarter and the player walks on, unless the arms' sweep would
pass over a wall, and then the step is blocked (pret CheckForRotatingGatePuzzleCollision,
called from field_player_avatar.c CheckForObjectEventCollision). So which tiles join
which depends on the gates' orientations, and every push changes them: a walk is planned
over (tile, orientations), see pathfinding.Pathfinder._astar.

Seen live: the 8B model stalled in FORTREE CITY GYM; the pathfinder planned through the
arms as if they were floor.
"""
from collections.abc import Callable, Sequence

# pret rotating_gate.c: enum { GATE_ARM_NORTH, GATE_ARM_EAST, GATE_ARM_SOUTH, GATE_ARM_WEST }
# and enum { ROTATE_NONE, ROTATE_ANTICLOCKWISE, ROTATE_CLOCKWISE }.
NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
ANTICLOCKWISE, CLOCKWISE = 1, 2

# pret sRotatingGate_ArmLayout: per shape, (short, long) segment of the north, east, south
# and west arm when the gate is at GATE_ORIENTATION_0. Index = the GATE_SHAPE_* enum.
ARM_LAYOUT: tuple[tuple[int, ...], ...] = (
    (1, 0, 1, 0, 0, 0, 0, 0),  # GATE_SHAPE_L1
    (1, 1, 1, 0, 0, 0, 0, 0),  # GATE_SHAPE_L2
    (1, 0, 1, 1, 0, 0, 0, 0),  # GATE_SHAPE_L3
    (1, 1, 1, 1, 0, 0, 0, 0),  # GATE_SHAPE_L4
    (1, 0, 1, 0, 1, 0, 0, 0),  # GATE_SHAPE_T1
    (1, 1, 1, 0, 1, 0, 0, 0),  # GATE_SHAPE_T2
    (1, 0, 1, 1, 1, 0, 0, 0),  # GATE_SHAPE_T3
    (1, 0, 1, 0, 1, 1, 0, 0),  # GATE_SHAPE_T4
    (1, 1, 1, 1, 1, 0, 0, 0),  # GATE_SHAPE_UNUSED_T1
    (1, 1, 1, 0, 1, 1, 0, 0),  # GATE_SHAPE_UNUSED_T2
    (1, 0, 1, 1, 1, 1, 0, 0),  # GATE_SHAPE_UNUSED_T3
    (1, 1, 1, 1, 1, 1, 0, 0),  # GATE_SHAPE_UNUSED_T4
)
SHAPES = {
    name: i
    for i, name in enumerate(
        ("L1", "L2", "L3", "L4", "T1", "T2", "T3", "T4", "UNUSED_T1", "UNUSED_T2", "UNUSED_T3", "UNUSED_T4")
    )
}

# pret sRotatingGate_RotationInfo{North,South,West,East}: for a step in that direction onto
# the tile (x - gate.x + 2, y - gate.y + 2) of the gate's 4x4 box, the arm segment the step
# crosses (world direction, long?) and which way pushing it turns the gate. Only the
# entries that are not GATE_ROT_NONE are listed, as (column, row): (turn, arm, long).
RotationInfo = tuple[int, int, int]
ROTATION_INFO: dict[str, dict[tuple[int, int], RotationInfo]] = {
    "UP": {
        (0, 1): (CLOCKWISE, WEST, 1), (1, 1): (CLOCKWISE, WEST, 0),
        (2, 1): (ANTICLOCKWISE, EAST, 0), (3, 1): (ANTICLOCKWISE, EAST, 1),
    },
    "DOWN": {
        (0, 2): (ANTICLOCKWISE, WEST, 1), (1, 2): (ANTICLOCKWISE, WEST, 0),
        (2, 2): (CLOCKWISE, EAST, 0), (3, 2): (CLOCKWISE, EAST, 1),
    },
    "LEFT": {
        (1, 0): (ANTICLOCKWISE, NORTH, 1), (1, 1): (ANTICLOCKWISE, NORTH, 0),
        (1, 2): (CLOCKWISE, SOUTH, 0), (1, 3): (CLOCKWISE, SOUTH, 1),
    },
    "RIGHT": {
        (2, 0): (CLOCKWISE, NORTH, 1), (2, 1): (CLOCKWISE, NORTH, 0),
        (2, 2): (ANTICLOCKWISE, SOUTH, 0), (2, 3): (ANTICLOCKWISE, SOUTH, 1),
    },
}

# pret sRotatingGate_ArmPositions{Clockwise,AntiClockwise}Rotation: the tile (relative to
# the gate's x, y) each arm segment sweeps over when it turns, indexed 2 * world arm + long.
ARM_SWEEP: dict[int, tuple[tuple[int, int], ...]] = {
    CLOCKWISE: ((0, -1), (1, -2), (0, 0), (1, 0), (-1, 0), (-1, 1), (-1, -1), (-2, -1)),
    ANTICLOCKWISE: ((-1, -1), (-1, -2), (0, -1), (1, -1), (0, 0), (0, 1), (-1, 0), (-2, 0)),
}

# pret sRotatingGate_FortreePuzzleConfig / sRotatingGate_TrickHousePuzzleConfig, in gate id
# order (the order CheckForRotatingGatePuzzleCollision tries them): x, y, shape and the
# orientation (quarter turns clockwise) RotatingGate_InitPuzzle resets it to on entering.
# Keyed by the map, as GetCurrentMapRotatingGatePuzzleType picks the puzzle by map group
# and number (map.json "id").
PUZZLES: dict[str, tuple[tuple[int, int, str, int], ...]] = {
    "MAP_FORTREE_CITY_GYM": (
        (6, 7, "T2", 1), (9, 15, "T2", 2), (3, 19, "T2", 1), (2, 6, "T1", 1),
        (9, 12, "T1", 0), (6, 23, "T1", 0), (12, 22, "T1", 0), (6, 3, "L4", 2),
    ),
    "MAP_ROUTE110_TRICK_HOUSE_PUZZLE6": (
        (14, 5, "T1", 1), (10, 6, "L2", 2), (6, 6, "L4", 1), (14, 8, "T1", 1),
        (3, 10, "L3", 3), (9, 14, "L1", 1), (3, 15, "T3", 0), (2, 17, "L2", 2),
        (12, 18, "T3", 3), (5, 18, "L4", 1), (10, 19, "L3", 2),
    ),
}

# pret include/constants/vars.h VAR_TEMP_0: the puzzle keeps gate i's orientation in byte i
# of the vars from VAR_TEMP_0 on (RotatingGate_GetGateOrientation casts GetVarPointer(VAR_TEMP_0)
# to u8 *); the map scripts note "makes use of VAR_TEMP_0 - VAR_TEMP_3" (Fortree).
VAR_TEMP_0 = 0x4000

DELTAS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

Orientations = tuple[int, ...]


class Gate:
    __slots__ = ("x", "y", "layout", "initial")

    def __init__(self, x: int, y: int, shape: str, initial: int):
        self.x, self.y, self.layout, self.initial = x, y, ARM_LAYOUT[SHAPES[shape]], initial

    def has_arm(self, orientation: int, arm: int, long: int) -> bool:
        """pret RotatingGate_HasArm: is there an arm segment at world direction arm now."""
        return bool(self.layout[((arm - orientation + 4) % 4) * 2 + long])

    def can_turn(self, orientation: int, turn: int, collision_at: Callable[[int, int], int]) -> bool:
        """pret RotatingGate_CanRotate: no tile any arm segment sweeps over is a wall.

        The retail game tests MapGridGetCollisionAt(...) == 1, not != 0 (the BUGFIX branch),
        so only collision value 1 stops a gate; that is what the cartridge does."""
        sweep = ARM_SWEEP[turn]
        for arm in range(4):
            for long in range(2):
                if self.layout[2 * arm + long]:
                    dx, dy = sweep[2 * ((orientation + arm) % 4) + long]
                    if collision_at(self.x + dx, self.y + dy) == 1:
                        return False
        return True


class RotatingGates:
    """One map's gate puzzle: its gates and the map's collision for their sweeps."""

    def __init__(self, gates: Sequence[Gate], collision_at: Callable[[int, int], int]):
        self.gates = tuple(gates)
        self.collision_at = collision_at

    @property
    def initial(self) -> Orientations:
        return tuple(g.initial for g in self.gates)

    def step(self, orientations: Orientations, button: str, dest: tuple[int, int]) -> Orientations | None:
        """The gates' orientations after a step in direction button onto dest, or None when
        a gate blocks it (pret CheckForRotatingGatePuzzleCollision). Only called for a step
        the map itself allows: the game checks gates only when GetCollisionAtCoords found
        nothing in the way (COLLISION_NONE).

        The first gate (in config order) with an arm in the way decides: it turns or it
        blocks, and no later gate is looked at."""
        table = ROTATION_INFO[button]
        x, y = dest
        for i, gate in enumerate(self.gates):
            if not (gate.x - 2 <= x <= gate.x + 1 and gate.y - 2 <= y <= gate.y + 1):
                continue
            info = table.get((x - gate.x + 2, y - gate.y + 2))
            if info is None:
                continue
            turn, arm, long = info
            orientation = orientations[i]
            if not gate.has_arm(orientation, arm, long):
                continue
            if not gate.can_turn(orientation, turn, self.collision_at):
                return None
            # pret RotatingGate_RotateInDirection.
            turned = (orientation + (1 if turn == CLOCKWISE else 3)) % 4
            return orientations[:i] + (turned,) + orientations[i + 1:]
        return orientations



def puzzle_for_map(map_id: str, raw_tiles, offset: tuple[int, int] = (0, 0)) -> RotatingGates | None:
    """The gate puzzle of the map whose map.json "id" is map_id, or None. raw_tiles is the
    walking grid's [y][x] (metatile, behavior, collision, elevation); offset is what the
    walking grid's coordinates are shifted by (local = map - offset)."""
    config = PUZZLES.get(map_id or "")
    if not config:
        return None
    ox, oy = offset
    tiles = raw_tiles or []

    def collision_at(x: int, y: int) -> int:
        # pret fieldmap.c GetMapGridBlockAt: off the map the border block is read with
        # MAPGRID_IMPASSABLE set, collision 3, which the retail == 1 test lets through.
        if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]) and tiles[y][x] is not None and len(tiles[y][x]) > 2:
            return int(tiles[y][x][2])
        return 3

    return RotatingGates([Gate(x - ox, y - oy, shape, o) for x, y, shape, o in config], collision_at)


def read_orientations(reader, count: int) -> Orientations | None:
    """The gates' live orientations (caller holds the emulator lock), or None when they
    can't be read or are not orientations (0-3): then the puzzle is not modelled."""
    values = []
    for var in range((count + 1) // 2):
        word = reader.read_game_var(VAR_TEMP_0 + var)
        if word is None:
            return None
        values.extend((word & 0xFF, word >> 8))  # GBA memory is little-endian
    values = values[:count]
    if any(v > 3 for v in values):
        return None
    return tuple(values)
