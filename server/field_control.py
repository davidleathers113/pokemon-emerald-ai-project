"""Overworld state reads and decisions behind movement feedback, exits and dialogue.

The harness reads these straight from RAM rather than through PokemonEmeraldReader
because the reader's helpers for them are wrong on this ROM (checked on real
savestates):
- ``read_player_facing`` reads SaveBlock1+4, which is the current map group, not the
  facing direction (it reported "South" on Route 101 while the player faced north).
- ``is_in_dialog`` returned True in the plain overworld (its script-context addresses
  are not pokeemerald's), and ``read_dialog`` keeps returning the last message long
  after the box closed.

Addresses are pokeemerald (US Emerald) symbols. Every value below was checked against
savestates: walking, bumping into trees, a ledge jump, crossing a map edge, Mom's
dialogue, the START menu and the save YES/NO prompt.

Nothing here touches the emulator except through the ``memory_reader`` passed in,
so the decisions can be unit-tested with plain snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# gMain.callback2; CB2_Overworld is a Thumb function, so the stored pointer has bit 0 set.
G_MAIN_CALLBACK2 = 0x030022C4
CB2_OVERWORLD = 0x08085E5C
# gMain.inBattle (bit 1 of the byte at gMain+0x439). PokemonEmeraldReader.is_in_battle
# also treats a nonzero u16 at 0x02022AAE as "in battle", but that address lies inside
# sSavedLinkPlayers, not gBattleTypeFlags (0x02022FEC), so it is not used here.
G_MAIN_IN_BATTLE_BYTE = 0x030026F9
IN_BATTLE_MASK = 0x02
# struct PlayerAvatar: +3 tileTransitionState, +5 objectEventId.
G_PLAYER_AVATAR = 0x02037590
T_TILE_TRANSITION = 1
G_OBJECT_EVENTS = 0x02037350
OBJECT_EVENT_SIZE = 0x24
OBJECT_EVENT_COUNT = 16
# ObjectEvent: +0x10 currentCoords (s16 x, y, map coords + MAP_OFFSET), +0x18 low nibble facingDirection.
MAP_OFFSET = 7
# Set while a script, the START menu or a warp owns the player (LockPlayerFieldControls).
S_LOCK_FIELD_CONTROLS = 0x03000F2C
# sFieldMessageBoxMode: 2 while a field message is printing (including the pause for A
# between paragraphs). It drops back to 0 as soon as the last character is printed,
# while the box stays on screen until the script closes it, so it alone misses the
# final "press A" of every message and the question in front of a YES/NO.
S_FIELD_MESSAGE_BOX_MODE = 0x020375BC
# sGlobalScriptContext: +1 mode (2 = native function), +4 nativePtr. A script sitting
# in `waitbuttonpress` runs the native WaitForAorBPress: the box is up and needs A.
S_GLOBAL_SCRIPT_CONTEXT = 0x03000E40
SCRIPT_MODE_NATIVE = 2
WAIT_FOR_A_OR_B_PRESS = 0x0809AC98
# gWindows[32]: 12-byte struct Window; bg == 0xFF marks a free slot. Window 0 is the
# field message box and is always allocated in the overworld. Menus, the YES/NO box,
# multichoice lists and the map-name popup each add another window.
G_WINDOWS = 0x02020004
WINDOW_STRUCT_SIZE = 12
WINDOW_COUNT = 32
FIELD_MESSAGE_WINDOW = 0
# Field messages are expanded into gStringVar4 before they are printed.
G_STRING_VAR4 = 0x02021FC4
STRING_VAR4_SIZE = 1000

DIRECTION_DELTAS: Dict[str, Tuple[int, int]] = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}
# pokeemerald DIR_SOUTH=1, DIR_NORTH=2, DIR_WEST=3, DIR_EAST=4.
_FACING_TO_BUTTON = {1: "DOWN", 2: "UP", 3: "LEFT", 4: "RIGHT"}

# A press that started while the player was still mid-step is judged only once the
# step (or a ledge jump, ~34 frames) is over; this caps that wait.
SETTLE_MAX_FRAMES = 48


@dataclass
class FieldSnapshot:
    x: int
    y: int
    map_id: Tuple[int, int]
    facing: Optional[str]
    in_transition: bool
    controls_locked: bool
    message_box: int
    in_battle: bool
    overworld: bool
    extra_windows: List[int] = field(default_factory=list)
    waiting_for_button: bool = False
    printer_active: bool = False  # a text printer is printing or on its arrow (menu_state.text_printing)
    phone_call: bool = False  # a POKéNAV call is playing (menu_state.phone_call_active)

    @property
    def position(self) -> Tuple[int, int, Tuple[int, int]]:
        return (self.x, self.y, self.map_id)

    @property
    def field_free(self) -> bool:
        """The player is walking around: direction presses move or turn the player."""
        return self.overworld and not self.in_battle and not self.controls_locked and self.message_box == 0


def _player_object_base(reader) -> int:
    obj_id = reader._read_u8(G_PLAYER_AVATAR + 5)
    if obj_id >= OBJECT_EVENT_COUNT:
        obj_id = 0
    return G_OBJECT_EVENTS + obj_id * OBJECT_EVENT_SIZE


def active_windows(reader) -> List[int]:
    return [
        i for i in range(WINDOW_COUNT)
        if reader._read_u8(G_WINDOWS + i * WINDOW_STRUCT_SIZE) != 0xFF
    ]


def read_field_snapshot(reader) -> FieldSnapshot:
    """One consistent read of everything the movement and dialogue logic needs."""
    x, y = reader.read_coordinates()
    base = _player_object_base(reader)
    facing = _FACING_TO_BUTTON.get(reader._read_u16(base + 0x18) & 0xF)
    in_battle = bool(reader._read_u8(G_MAIN_IN_BATTLE_BYTE) & IN_BATTLE_MASK)
    from server.menu_state import phone_call_active, text_printing  # menu_state imports this module

    return FieldSnapshot(
        x=int(x),
        y=int(y),
        map_id=tuple(reader._read_current_map()),
        facing=facing,
        in_transition=reader._read_u8(G_PLAYER_AVATAR + 3) == T_TILE_TRANSITION,
        controls_locked=reader._read_u8(S_LOCK_FIELD_CONTROLS) != 0,
        message_box=reader._read_u8(S_FIELD_MESSAGE_BOX_MODE),
        in_battle=in_battle,
        overworld=(reader._read_u32(G_MAIN_CALLBACK2) | 1) == (CB2_OVERWORLD | 1),
        extra_windows=[w for w in active_windows(reader) if w != FIELD_MESSAGE_WINDOW],
        waiting_for_button=(
            reader._read_u8(S_GLOBAL_SCRIPT_CONTEXT + 1) == SCRIPT_MODE_NATIVE
            and (reader._read_u32(S_GLOBAL_SCRIPT_CONTEXT + 4) | 1) == (WAIT_FOR_A_OR_B_PRESS | 1)
        ),
        printer_active=text_printing(reader),
        phone_call=phone_call_active(reader),
    )


# struct PlayerAvatar +0 flags (pret include/global.fieldmap.h): set when the Surf field
# effect starts (pret src/field_effect.c SurfFieldEffect_Init), cleared when the landing
# jump starts (src/field_player_avatar.c CreateStopSurfingTask, which also locks the
# field controls until the jump is over).
PLAYER_AVATAR_FLAG_SURFING = 1 << 3


def player_surfing(reader) -> bool:
    return bool(reader._read_u8(G_PLAYER_AVATAR) & PLAYER_AVATAR_FLAG_SURFING)


def npc_positions(reader) -> List[Tuple[int, int]]:
    """Map tiles where a visible object event other than the player stands right now.

    Only objects the game has loaded (near the camera) are listed; the map file's static
    objects cover the rest.
    """
    player_base = _player_object_base(reader)
    spots = []
    for i in range(OBJECT_EVENT_COUNT):
        base = G_OBJECT_EVENTS + i * OBJECT_EVENT_SIZE
        if base == player_base:
            continue
        flags = reader._read_u8(base) | (reader._read_u8(base + 1) << 8)
        active, invisible = flags & 0x1, flags & (1 << 13)
        if not active or invisible:
            continue
        spots.append((reader._read_s16(base + 0x10) - MAP_OFFSET, reader._read_s16(base + 0x12) - MAP_OFFSET))
    return spots


def npc_at(reader, x: int, y: int) -> bool:
    """True if a visible object event other than the player stands on (x, y)."""
    return (x, y) in npc_positions(reader)


def moved_people(reader) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """(map-file tile, where they stand now) of this map's loaded people who stand off
    their map-file tile.

    pret include/global.fieldmap.h struct ObjectEvent: +0x09 mapNum, +0x0A mapGroup,
    +0x0C initialCoords (the template's x, y + MAP_OFFSET: map.json's tile), +0x10
    currentCoords. Seen live in FORTREE CITY GYM: JARED walked over to battle and stayed
    at (4,16), while the map (map.json) still put him at (4,14). Walks and talk_to take
    people from here, not from the map file.
    """
    player_base = _player_object_base(reader)
    map_group, map_num = reader._read_current_map()
    moved = []
    for i in range(OBJECT_EVENT_COUNT):
        base = G_OBJECT_EVENTS + i * OBJECT_EVENT_SIZE
        if base == player_base:
            continue
        flags = reader._read_u8(base) | (reader._read_u8(base + 1) << 8)
        if not flags & 0x1 or flags & (1 << 13):  # inactive or invisible
            continue
        if (reader._read_u8(base + 0x0A), reader._read_u8(base + 0x09)) != (map_group, map_num):
            continue  # a neighbouring map's person: not in this map's file
        spawn = (reader._read_s16(base + 0x0C) - MAP_OFFSET, reader._read_s16(base + 0x0E) - MAP_OFFSET)
        now = (reader._read_s16(base + 0x10) - MAP_OFFSET, reader._read_s16(base + 0x12) - MAP_OFFSET)
        if spawn != now:
            moved.append((spawn, now))
    return moved


# ---------------------------------------------------------------------------
# Movement verdicts
# ---------------------------------------------------------------------------

BLOCKED = "blocked"
INTERRUPTED = "interrupted"


def judge_direction_press(
    button: str,
    before: FieldSnapshot,
    after: FieldSnapshot,
    previous_button: Optional[str],
) -> Optional[str]:
    """Classify one finished direction press: BLOCKED, INTERRUPTED or None (fine).

    BLOCKED needs proof that the press should have moved the player: it started in the
    free overworld from a standing position, the player already faced that way (or the
    previous press in the sequence was the same direction), and neither the tile nor
    the map changed. A press that only turned the player is not blocked.

    INTERRUPTED means the game took the player out of free walking on this map (a wild
    battle, a trainer's dialogue, a cutscene): later presses in the sequence would land
    in a battle menu or a text box instead of walking.
    """
    if button not in DIRECTION_DELTAS:
        return None
    if not before.field_free or before.in_transition:
        return None
    if after.map_id != before.map_id:
        return None  # crossed an edge or took a warp: arrival, not a failure
    if not after.field_free:
        return INTERRUPTED
    moved = (after.x, after.y) != (before.x, before.y)
    if moved:
        return None
    if before.facing == button or previous_button == button:
        return BLOCKED
    return None


def interruption_cause(after: FieldSnapshot) -> str:
    if after.in_battle:
        return "a battle started"
    if not after.overworld:
        return "the game left the overworld screen (battle intro, menu or warp)"
    if after.message_box or after.waiting_for_button:
        return "a text box opened (someone spoke to you or an event started)"
    return "the game took control of the player (cutscene, warp or menu)"


def blocked_target(button: str, before: FieldSnapshot) -> Tuple[int, int]:
    dx, dy = DIRECTION_DELTAS[button]
    return before.x + dx, before.y + dy


# ---------------------------------------------------------------------------
# Dialogue
# ---------------------------------------------------------------------------

DIALOGUE_NONE = "none"          # free overworld, no text box
DIALOGUE_PLAIN = "plain"        # a text box that only needs A
DIALOGUE_CHOICE = "choice"      # YES/NO, multichoice, START menu or another menu window
DIALOGUE_BUSY = "busy"          # a script holds the player with no box up (healing jingle, cutscene)
DIALOGUE_OTHER = "other"        # battle, naming screen, bag... (not the field)


def dialogue_status(snapshot: FieldSnapshot) -> str:
    """What kind of text interaction is on screen, judged conservatively.

    Any window besides the field message box while a box is up or a script holds the
    player makes this a CHOICE, so A is never pressed into a menu (the nurse's YES/NO
    shows with sFieldMessageBoxMode already back to 0). The map-name popup is also a
    window, but it shows while the player walks freely, so it does not count. A box is
    PLAIN while its text prints or the script waits in `waitbuttonpress`.

    A box whose text is still printing (or on its down arrow) is PLAIN even with another
    window up: a YES/NO or menu opens only after its question's printer has finished.
    Seen live: a POKéNAV call (Dad, on Mr. Briney's boat) draws the caller's icon in a
    second window; read as a CHOICE, nothing ever pressed A and the boat sat at sea.
    After its last line the call waits for A to hang up with no printer running, so a
    call is PLAIN throughout (calls never ask a question).
    """
    if snapshot.in_battle or not snapshot.overworld:
        return DIALOGUE_OTHER
    if snapshot.phone_call:
        return DIALOGUE_PLAIN
    if snapshot.printer_active and snapshot.message_box:
        return DIALOGUE_PLAIN
    if snapshot.extra_windows and (snapshot.message_box or snapshot.controls_locked):
        return DIALOGUE_CHOICE
    if snapshot.message_box or snapshot.waiting_for_button:
        return DIALOGUE_PLAIN
    if snapshot.controls_locked:
        return DIALOGUE_BUSY
    return DIALOGUE_NONE


# EXT_CTRL_CODE (0xFC) argument counts, from pokeemerald include/characters.h.
_EXT_CTRL_ARGS = {
    0x01: 1, 0x02: 1, 0x03: 1, 0x04: 3, 0x05: 1, 0x06: 1, 0x07: 0, 0x08: 1, 0x09: 0,
    0x0A: 0, 0x0B: 2, 0x0C: 1, 0x0D: 1, 0x0E: 1, 0x0F: 0, 0x10: 2, 0x11: 1, 0x12: 1,
    0x13: 1, 0x14: 1, 0x15: 0, 0x16: 0, 0x17: 0, 0x18: 0,
}


def decode_field_message(raw: bytes) -> str:
    """Decode an expanded field message: line/paragraph breaks become spaces."""
    from pokemon_env.emerald_utils import EmeraldCharmap

    charmap = EmeraldCharmap().charmap
    out: List[str] = []
    i = 0
    while i < len(raw):
        byte = raw[i]
        if byte == 0xFF:
            break
        if byte in (0xFA, 0xFB, 0xFE):  # scroll, new box, newline
            out.append(" ")
        elif byte == 0xFC:
            i += 1 + _EXT_CTRL_ARGS.get(raw[i + 1] if i + 1 < len(raw) else 0, 0)
        elif byte == 0xFD:  # unexpanded placeholder
            i += 1
        elif byte < len(charmap):
            out.append(charmap[byte])
        i += 1
    return " ".join("".join(out).split())


def read_field_message(reader) -> str:
    return decode_field_message(reader._read_bytes(G_STRING_VAR4, STRING_VAR4_SIZE))


@dataclass
class DialogueAdvanceResult:
    presses: int
    texts: List[str]
    final_status: str
    capped: bool

    def summary(self) -> str:
        said = " / ".join(self.texts) if self.texts else "(no text read)"
        if self.final_status == DIALOGUE_CHOICE:
            end = "A choice (YES/NO or a menu) is open now: answer it yourself."
        elif self.final_status == DIALOGUE_PLAIN:
            end = f"Stopped after {self.presses} presses; the text box is still open."
        elif self.final_status == DIALOGUE_OTHER:
            end = "The game left the overworld (battle or another screen)."
        elif self.final_status == DIALOGUE_BUSY:
            end = "An event is still running (no text box right now)."
        else:
            end = "The text box closed."
        return f"Dialogue read (auto-advanced with {self.presses} A presses): {said} {end}"


def auto_advance_dialogue(
    status: Callable[[], str],
    read_text: Callable[[], str],
    press_a: Callable[[], None],
    settle: Callable[[], None],
    max_presses: int = 15,
    max_busy_waits: int = 30,
) -> DialogueAdvanceResult:
    """Press A through a plain text box, never into a YES/NO or menu.

    Before every A the status must read PLAIN twice, ``settle`` apart: a YES/NO box
    is created a frame or two after its question finishes printing, and the second
    look catches it. An A that is already held when the box appears does not answer
    it (menus react to a new press only). While a script holds the player with no
    box up (BUSY: the nurse's healing jingle after YES, or between two boxes) it waits
    up to ``max_busy_waits`` settles for the next box instead of handing back to the
    model mid-event.
    """
    texts: List[str] = []
    presses = 0
    busy_waits = 0
    looks = 0

    def note_text():
        text = read_text()
        if text and text not in texts:
            texts.append(text)

    while presses < max_presses and looks < (max_presses + 1) * (3 + max_busy_waits):
        looks += 1
        current = status()
        if current == DIALOGUE_BUSY and busy_waits < max_busy_waits:
            busy_waits += 1
            settle()
            continue
        if current != DIALOGUE_PLAIN:
            break
        settle()
        if status() != DIALOGUE_PLAIN:
            continue  # re-classify: a YES/NO may have just appeared
        note_text()
        press_a()
        presses += 1
        busy_waits = 0  # the wait budget is per gap between two boxes
    final = status()
    if final in (DIALOGUE_PLAIN, DIALOGUE_CHOICE):
        note_text()  # the question in front of a YES/NO, or the box we stopped on
    return DialogueAdvanceResult(
        presses=presses,
        texts=texts,
        final_status=final,
        capped=presses >= max_presses and final == DIALOGUE_PLAIN,
    )


# ---------------------------------------------------------------------------
# Exits
# ---------------------------------------------------------------------------

EDGE_NAMES = {"up": "north", "down": "south", "left": "west", "right": "east"}
EDGE_BUTTON = {"up": "UP", "down": "DOWN", "left": "LEFT", "right": "RIGHT"}
_DIRECTION_WORDS = {
    "NORTH": "up", "UP": "up", "N": "up", "NORTHEDGE": "up",
    "SOUTH": "down", "DOWN": "down", "S": "down", "SOUTHEDGE": "down",
    "WEST": "left", "LEFT": "left", "W": "left", "WESTEDGE": "left",
    "EAST": "right", "RIGHT": "right", "E": "right", "EASTEDGE": "right",
}
# Tiles a player can't stand on (same rule the EXITS line uses for edge openings).
EDGE_UNWALKABLE = {"#", "X", "W"}


def normalize_name(text: str) -> str:
    """'Professor Birch's Lab' -> 'PROFESSORBIRCHSLAB'; spacing/punctuation-insensitive."""
    return "".join(ch for ch in str(text).upper() if ch.isalnum())


def _integers_in(text: str) -> List[int]:
    numbers, current = [], ""
    for ch in str(text):
        if ch.isdigit():
            current += ch
        elif current:
            numbers.append(int(current))
            current = ""
    if current:
        numbers.append(int(current))
    return numbers


def match_exits(exits: List[Dict], query: str) -> List[Dict]:
    """Exits that ``query`` names, best matches first.

    Accepts a compass/edge word ("north", "UP", "north edge"), a destination name
    or part of one ("OLDALE TOWN", "oldale", "Professor Birch's Lab"), or the
    coordinates shown in the EXITS list ("(8, 9)"), or both ("GRANITE CAVE 1F (4, 21)":
    the route hint's form where several warps lead to the same map).
    """
    wanted = normalize_name(query)
    if not wanted:
        return []
    # An EXITS line copied whole ("door/warp at (6, 12) -> ROUTE 112", "west edge ->
    # ROUTE 113"). Seen live in ROUTE 112 CABLE CAR STATION: that exact line answered "No
    # exit … matches", and the model sent it again and again.
    labelled = [e for e in exits if normalize_name(e.get("label", "")) == wanted]
    if labelled:
        return labelled
    text = query.strip()
    open_at = text.rfind("(")
    if open_at > 0 and text.endswith(")"):
        parts = [p.strip() for p in text[open_at + 1:-1].split(",")]
        if len(parts) == 2 and all(p.lstrip("-").isdigit() for p in parts):
            coords = (int(parts[0]), int(parts[1]))
            named = match_exits(exits, text[:open_at])
            at = [e for e in named if e["kind"] == "warp" and coords in (e.get("listed_xy"), (e["x"], e["y"]))]
            if at:
                return at
    stripped = wanted
    for word in ("EDGEOFTHEMAP", "EDGE", "THE", "GOTO", "EXIT"):
        stripped = stripped.replace(word, "")
    if stripped in _DIRECTION_WORDS:
        wanted_dir = _DIRECTION_WORDS[stripped]
        return [e for e in exits if e["kind"] == "edge" and e["direction"] == wanted_dir]

    exact = [e for e in exits if normalize_name(e["name"]) == wanted]
    if exact:
        return exact
    partial = [e for e in exits if wanted in normalize_name(e["name"])]
    if partial:
        # "LITTLEROOT" should prefer LITTLEROOT TOWN over LITTLEROOT TOWN BRENDANS HOUSE 2F.
        return sorted(partial, key=lambda e: len(normalize_name(e["name"])))

    numbers = _integers_in(query)
    if len(numbers) == 2:
        coords = tuple(numbers)
        return [e for e in exits if e["kind"] == "warp" and coords in (e.get("listed_xy"), (e["x"], e["y"]))]
    return []


def edge_candidates(grid: List, direction: str, surf: bool = False) -> List[Tuple[int, int]]:
    """Standable tiles on one edge of the grid (local coordinates); with surf (the party
    can Surf) water too, which a surfer crosses the edge on."""
    unwalkable = EDGE_UNWALKABLE - {"W"} if surf else EDGE_UNWALKABLE
    height = len(grid)
    if not height:
        return []
    if direction in ("up", "down"):
        y = 0 if direction == "up" else height - 1
        return [(x, y) for x, cell in enumerate(grid[y]) if cell not in unwalkable]
    width = max(len(row) for row in grid)
    x = 0 if direction == "left" else width - 1
    return [
        (x, y) for y, row in enumerate(grid)
        if x < len(row) and row[x] not in unwalkable
    ]


_ARROW_WORDS = (("SOUTH", "DOWN"), ("NORTH", "UP"), ("WEST", "LEFT"), ("EAST", "RIGHT"))


def warp_push_directions(
    grid: List, raw_tiles: Optional[List], x: int, y: int, last_move: Optional[str]
) -> List[str]:
    """Directions to press on a warp tile that did not fire when stepped on.

    Exit mats in houses and cave mouths are arrow warps (e.g. MB_SOUTH_ARROW_WARP):
    they fire when the player walks *off* them toward the arrow. The tile's metatile
    behavior names that direction; otherwise try directions whose neighbour is not a
    real map tile or is unwalkable. The ASCII grid alone is not enough: it draws a
    display-only door marker below a house's exit mats.
    """
    out: List[str] = []
    tile = None
    if raw_tiles and 0 <= y < len(raw_tiles) and 0 <= x < len(raw_tiles[y]):
        tile = raw_tiles[y][x]
    if tile is not None and len(tile) > 1:
        behavior = str(getattr(tile[1], "name", tile[1]))
        if "ARROW_WARP" in behavior:
            for word, button in _ARROW_WORDS:
                if word in behavior:
                    out.append(button)

    def unwalkable(nx, ny):
        if raw_tiles is not None and not (0 <= ny < len(raw_tiles) and 0 <= nx < len(raw_tiles[ny])):
            return True
        if ny < 0 or ny >= len(grid) or nx < 0 or nx >= len(grid[ny]):
            return True
        return grid[ny][nx] in EDGE_UNWALKABLE

    order = ([last_move] if last_move else []) + ["DOWN", "UP", "LEFT", "RIGHT"]
    for button in order:
        if button in out or button not in DIRECTION_DELTAS:
            continue
        dx, dy = DIRECTION_DELTAS[button]
        if unwalkable(x + dx, y + dy):
            out.append(button)
    return out


def end_of_path(start: Tuple[int, int], buttons: List[str]) -> Tuple[int, int]:
    x, y = start
    for button in buttons:
        dx, dy = DIRECTION_DELTAS.get(button, (0, 0))
        x, y = x + dx, y + dy
    return x, y
