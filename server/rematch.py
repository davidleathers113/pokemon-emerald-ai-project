"""Trainer rematches through Match Call: who is registered, who wants a rematch now, and
where they stand.

Everything is read, not listed by hand: the trainers, their rematch teams and their maps
from the ROM's gRematchTable, the save's registration and rematch state from RAM, and
where each trainer stands and what their script does from pret's map data.

How a rematch comes about (pret src/battle_setup.c, src/overworld.c, src/match_call.c):
  - every step on foot counts up gSaveBlock1Ptr->trainerRematchStepCounter, to 255, but
    only with at least five badges (IncrementRematchStepCounter, HasAtLeastFiveBadges);
  - on each map load, crossing a map's edge or through a warp (overworld.c
    LoadMapFromCameraTransition and mli0_load_map both call
    TryUpdateRandomTrainerRematches), a maxed counter gives every registered trainer of
    that map who isn't ready yet a 31% chance (Random() % 100 <= 30) to want a rematch
    (UpdateRandomTrainerRematches). The counter goes back to 0 only when one of that
    map's registered trainers is ready (one already was, or one rolled it); otherwise it
    stays at 255 and the next map load rolls again;
  - a POKéNAV call from a trainer on another route can also make them ready
    (SelectMatchCallMessage → UpdateRematchIfDefeated), with five badges in a row;
  - trainerRematches[i] is then the tier of the team they bring (SetRematchIdForTrainer:
    the first of trainerIds[1..4] not yet beaten), and after the last tier is beaten that
    one repeats (GetRematchTrainerIdFromTable). Winning clears it (HandleRematchVarsOnBattleEnd);
  - a trainer is registered right after the first win when the player has Match Call
    (CB2_EndTrainerBattle → RegisterTrainerInMatchCall, and the trainer's after-battle
    script's register_matchcall). Only trainers whose script asks IsTrainerRegistered
    (the early routes', beaten before the POKéNAV) offer it again when talked to later.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# pret .sym gRematchTable. struct RematchTrainer {u16 trainerIds[REMATCHES_COUNT]; u16
# mapGroup; u16 mapNum;} (include/battle_setup.h) is 14 bytes in C but 16 in the ROM:
# agbcc pads structs to 4 bytes. The .sym gives the table 0x4E0 bytes (78 x 16), and
# ROSE's entry reads {37, 40, 41, 42, 43, 0, 33, 0}: her five tiers, ROUTE 118 (0, 33).
G_REMATCH_TABLE = 0x085500A4
REMATCH_ENTRY_SIZE = 16
REMATCHES_COUNT = 5
MAP_GROUP_OFFSET = 2 * REMATCHES_COUNT
MAP_NUM_OFFSET = MAP_GROUP_OFFSET + 2
# struct SaveBlock1 (pret include/global.h): /*0x9C8*/ u16 trainerRematchStepCounter;
# /*0x9CA*/ u8 trainerRematches[MAX_REMATCH_ENTRIES].
SAVE_BLOCK1_STEP_COUNTER = 0x9C8
SAVE_BLOCK1_REMATCHES = 0x9CA
STEP_COUNTER_MAX = 255  # battle_setup.c STEP_COUNTER_MAX
# pret .sym gTrainerBattleOpponent_A (u16): the trainer id of the battle in progress or
# the last one; for a rematch, the tier's own id (GetRematchTrainerId).
G_TRAINER_BATTLE_OPPONENT_A = 0x02038BCA
REMATCH_CHANCE_PERCENT = 31  # UpdateRandomTrainerRematches: Random() % 100 <= 30
MIN_BADGES = 5  # HasAtLeastFiveBadges
BADGE_COUNT = 8


def _constants_dir() -> Optional[Path]:
    from pokemon_env.porymap_paths import get_porymap_root

    root = get_porymap_root()
    return Path(root) / "include" / "constants" if root else None


@lru_cache(maxsize=1)
def rematch_ids() -> Dict[str, int]:
    """REMATCH_ROSE -> 0 ... REMATCH_TABLE_ENTRIES -> 78, plus the two #defines on it
    (REMATCH_SPECIAL_TRAINER_START, REMATCH_ELITE_FOUR_ENTRIES), from pret's
    include/constants/rematches.h (copied into the repo with the other constants)."""
    folder = _constants_dir()
    path = folder / "rematches.h" if folder else None
    if path is None or not path.exists():
        return {}
    ids: Dict[str, int] = {}
    in_enum = False
    for line in path.read_text(encoding="utf-8").splitlines():
        code = line.split("//")[0].strip()
        if code.startswith("enum"):
            in_enum = True
            continue
        if in_enum:
            if code.startswith("}"):
                in_enum = False
                continue
            name = code.rstrip(",").strip()
            if name.startswith("REMATCH_"):
                ids[name] = len(ids)
            continue
        words = code.split()
        if len(words) == 3 and words[0] == "#define" and words[2] in ids:
            ids[words[1]] = ids[words[2]]
    return ids


@lru_cache(maxsize=1)
def _flag_bases() -> Dict[str, int]:
    """The flag ranges flags.h defines by number: TRAINER_FLAGS_START (0x500, a trainer's
    "beaten" flag is it + trainer id) and TRAINER_REGISTERED_FLAGS_START (0x15C, a
    trainer's Match Call flag is it + its REMATCH_ id)."""
    folder = _constants_dir()
    path = folder / "flags.h" if folder else None
    bases: Dict[str, int] = {}
    if path is None or not path.exists():
        return bases
    for line in path.read_text(encoding="utf-8").splitlines():
        words = line.split("//")[0].split()
        if len(words) == 3 and words[0] == "#define" and words[1] in ("TRAINER_FLAGS_START", "TRAINER_REGISTERED_FLAGS_START"):
            try:
                bases[words[1]] = int(words[2], 0)
            except ValueError:
                continue
    return bases


@dataclass(frozen=True)
class RematchEntry:
    """One gRematchTable entry, with where its trainer stands (pret map data)."""

    index: int  # its REMATCH_ id: the trainerRematches[] slot and the registered flag's offset
    name: str  # "FERNANDO", "ANNA AND MEG"
    trainer_ids: Tuple[int, ...]  # the teams, first battle first
    map_dir: str  # "Route123"
    map_name: str  # "ROUTE 123"
    spots: Tuple[Tuple[int, int], ...]  # map-file tiles of the trainer's object(s)
    registers_on_talk: bool  # the script asks IsTrainerRegistered: talking offers Match Call
    double: bool  # a double battle (two trainers or one with two Pokémon out)
    rolls: bool  # the step counter's map-load roll can make them ready (not a gym leader)


@dataclass(frozen=True)
class RematchTrainer:
    entry: RematchEntry
    beaten: bool  # their first battle was won
    registered: bool  # in the POKéNAV's Match Call list
    ready_tier: int  # trainerRematches[]: 0 = not ready, else the team (trainer_ids index) they bring
    forbidden: bool  # pret IsRematchForbidden: the Elite Four, and Victory Road Wally until beaten

    @property
    def ready(self) -> bool:
        return self.ready_tier > 0 and not self.forbidden


@dataclass(frozen=True)
class Rematches:
    has_match_call: bool
    badges: int
    step_counter: int
    trainers: Tuple[RematchTrainer, ...]

    @property
    def counting(self) -> bool:
        """The step counter counts (five badges)."""
        return self.badges >= MIN_BADGES

    def on_map(self, map_dir: str) -> List[RematchTrainer]:
        return [t for t in self.trainers if t.entry.map_dir == map_dir and not t.forbidden]

    def ready(self) -> List[RematchTrainer]:
        return [t for t in self.trainers if t.ready]


def _map_rematch_objects(map_dir: str) -> Dict[int, List[Tuple[int, int, bool, bool]]]:
    """{first-battle trainer id: [(x, y, registers on talk, double)]} for the objects on a
    map whose script starts a trainer battle (porymap_state's own reading of pret's
    scripts.inc: the first trainerbattle line names the trainer)."""
    from server.game_tools import _raw_map_json
    from utils.mapping.porymap_state import _battle_trainer, _map_script_labels, _trainer_ids

    ids = _trainer_ids()
    labels = _map_script_labels(map_dir)
    out: Dict[int, List[Tuple[int, int, bool, bool]]] = {}
    for obj in _raw_map_json(map_dir).get("object_events") or []:
        body = labels.get(str(obj.get("script") or ""))
        if not body:
            continue
        for line in body:
            trainer = _battle_trainer(line, ids)
            if trainer is None:
                continue
            # Route104_EventScript_Cindy: `specialvar VAR_RESULT, IsTrainerRegistered`, then
            # Route104_EventScript_TryRegisterCindy offers Match Call. Most trainers' scripts
            # (Route123_EventScript_Fernando) register only in the after-battle script.
            asks = any("IsTrainerRegistered" in other.replace(",", " ").split() for other in body)
            double = line.split()[0].startswith("trainerbattle_double")
            out.setdefault(ids[trainer], []).append((int(obj.get("x", 0)), int(obj.get("y", 0)), asks, double))
            break
    return out


_ENTRIES: Optional[Tuple[RematchEntry, ...]] = None


def rematch_entries(reader) -> Tuple[RematchEntry, ...]:
    """Every gRematchTable entry, read from the ROM once (call under the memory lock)."""
    global _ENTRIES
    if _ENTRIES is not None:
        return _ENTRIES
    from utils.mapping.porymap_state import _trainer_ids, decomp_map_at, trainer_name

    enum = rematch_ids()
    count = enum.get("REMATCH_TABLE_ENTRIES", 0)
    special = enum.get("REMATCH_SPECIAL_TRAINER_START", -1)
    consts = {value: name for name, value in _trainer_ids().items()}
    entries = []
    objects_by_map: Dict[str, Dict[int, List[Tuple[int, int, bool, bool]]]] = {}
    for i in range(count):
        raw = bytes(reader._read_bytes(G_REMATCH_TABLE + i * REMATCH_ENTRY_SIZE, REMATCH_ENTRY_SIZE))
        tiers = [int.from_bytes(raw[2 * j:2 * j + 2], "little") for j in range(REMATCHES_COUNT)]
        group = int.from_bytes(raw[MAP_GROUP_OFFSET:MAP_GROUP_OFFSET + 2], "little")
        num = int.from_bytes(raw[MAP_NUM_OFFSET:MAP_NUM_OFFSET + 2], "little")
        found = decomp_map_at(group, num)
        map_dir, map_name = found if found is not None else ("", f"MAP {group}.{num}")
        if map_dir not in objects_by_map:
            objects_by_map[map_dir] = _map_rematch_objects(map_dir) if map_dir else {}
        spots = objects_by_map[map_dir].get(tiers[0], [])
        # The tiers end at the first 0 (GetRematchTrainerIdFromTable: "previous entry was
        # this trainer's last one").
        teams = tuple(t for t in tiers[: next((j for j, t in enumerate(tiers) if t == 0), len(tiers))])
        entries.append(RematchEntry(
            index=i,
            name=trainer_name(consts.get(tiers[0], f"TRAINER_{tiers[0]}")),
            trainer_ids=teams,
            map_dir=map_dir,
            map_name=map_name,
            spots=tuple((x, y) for x, y, _, _ in spots),
            registers_on_talk=any(asks for _, _, asks, _ in spots),
            double=any(double for _, _, _, double in spots),
            # UpdateRandomTrainerRematches loops i <= REMATCH_SPECIAL_TRAINER_START: the
            # gym leaders after it have their own rematches (UpdateGymLeaderRematch).
            rolls=i <= special,
        ))
    _ENTRIES = tuple(entries)
    return _ENTRIES


def _flag(reader, flag: int) -> bool:
    return bool(flag >= 0 and reader._is_event_flag_set(flag))


def read_rematches(reader) -> Rematches:
    """The save's Match Call and rematch state (call under the memory lock)."""
    from server import game_tools

    enum = rematch_ids()
    bases = _flag_bases()
    registered_base = bases.get("TRAINER_REGISTERED_FLAGS_START", -1)
    beaten_base = bases.get("TRAINER_FLAGS_START", -1)
    sb1 = reader._read_u32(reader.addresses.SAVE_BLOCK1_PTR)
    badges = sum(
        _flag(reader, game_tools.flag_id(f"FLAG_BADGE0{n}_GET") or -1) for n in range(1, BADGE_COUNT + 1)
    )
    elite_four = enum.get("REMATCH_ELITE_FOUR_ENTRIES", 1 << 16)
    wally = enum.get("REMATCH_WALLY_VR", -1)
    wally_beaten = _flag(reader, game_tools.flag_id("FLAG_DEFEATED_WALLY_VICTORY_ROAD") or -1)
    trainers = []
    for entry in rematch_entries(reader):
        trainers.append(RematchTrainer(
            entry=entry,
            beaten=beaten_base >= 0 and bool(entry.trainer_ids) and _flag(reader, beaten_base + entry.trainer_ids[0]),
            registered=registered_base >= 0 and _flag(reader, registered_base + entry.index),
            ready_tier=reader._read_u8(sb1 + SAVE_BLOCK1_REMATCHES + entry.index),
            forbidden=entry.index >= elite_four or (entry.index == wally and not wally_beaten),
        ))
    return Rematches(
        has_match_call=_flag(reader, game_tools.flag_id("FLAG_HAS_MATCH_CALL") or -1),
        badges=badges,
        step_counter=reader._read_u16(sb1 + SAVE_BLOCK1_STEP_COUNTER),
        trainers=tuple(trainers),
    )


def trainer_beaten(reader, trainer_id: int) -> bool:
    """pret HasTrainerBeenFought: TRAINER_FLAGS_START + trainer id."""
    base = _flag_bases().get("TRAINER_FLAGS_START", -1)
    return base >= 0 and _flag(reader, base + trainer_id)


def battle_opponent(reader) -> Tuple[int, str]:
    """(trainer id, name) of gTrainerBattleOpponent_A: who the trainer battle is with."""
    from utils.mapping.porymap_state import _trainer_ids, trainer_name

    trainer_id = reader._read_u16(G_TRAINER_BATTLE_OPPONENT_A)
    const = next((name for name, value in _trainer_ids().items() if value == trainer_id), f"TRAINER_{trainer_id}")
    return trainer_id, trainer_name(const)


def entry_for_trainer(reader, trainer_id: int) -> Optional[RematchEntry]:
    """The rematch entry any of whose teams is trainer_id (pret TrainerIdToRematchTableId)."""
    return next((e for e in rematch_entries(reader) if trainer_id in e.trainer_ids), None)


def tier_text(trainer: RematchTrainer) -> str:
    """'rematch 2 of 4': which of the trainer's rematch teams comes next."""
    rematches = len(trainer.entry.trainer_ids) - 1
    return f"rematch {trainer.ready_tier} of {rematches}" if rematches > 0 else "rematch"


def _spot_text(entry: RematchEntry) -> str:
    return " / ".join(f"({x}, {y})" for x, y in entry.spots) or "(not on the map file)"


def ready_line(state: Rematches, here: str = "") -> str:
    """One line for the state: who wants a rematch and where, the trainers of the map the
    player is on (`here`, a map directory) first; short when nobody does."""
    if not state.has_match_call:
        return ""
    ready = sorted(state.ready(), key=lambda t: (t.entry.map_dir != here, t.entry.map_name, t.entry.name))
    if not ready:
        registered = sum(1 for t in state.trainers if t.registered and t.entry.rolls and not t.forbidden)
        if not state.counting:
            return f"REMATCHES: none ready ({registered} registered; the step counter starts at {MIN_BADGES} badges)."
        return f"REMATCHES: none ready ({registered} registered; step counter {state.step_counter}/{STEP_COUNTER_MAX})."
    by_map: Dict[str, List[str]] = {}
    for t in ready:
        by_map.setdefault(t.entry.map_name, []).append(f"{t.entry.name} at {_spot_text(t.entry)}")
    places = " | ".join(f"{name}: {', '.join(who)}" for name, who in by_map.items())
    return f"REMATCH READY: {places} (rematch_loop(map) fights them; trainer battles pay 1.5x EXP)."


def registration_hint(state: Rematches, map_dir: str) -> str:
    """How the map's rematch trainers who aren't registered can be, or "" when all are.

    pret: a trainer is registered right after the first win (with Match Call), so a
    registered trainer is never lost; only an early-route trainer beaten before the
    POKéNAV, whose script asks IsTrainerRegistered, offers it when talked to again."""
    if not state.has_match_call:
        return ""
    unbeaten, talk, never = [], [], []
    for t in state.on_map(map_dir):
        if t.registered or not t.entry.rolls:
            continue
        who = f"{t.entry.name} at {_spot_text(t.entry)}"
        if not t.beaten:
            unbeaten.append(who)
        elif t.entry.registers_on_talk:
            calls = ", ".join(f"talk_to({x}, {y})" for x, y in t.entry.spots[:1])
            talk.append(f"{t.entry.name} ({calls})" if calls else who)
        else:
            never.append(who)
    parts = []
    if talk:
        parts.append(f"beaten before you had Match Call, not registered: {'; '.join(talk)}; talked to, they offer to register")
    if unbeaten:
        parts.append(f"not beaten yet: {'; '.join(unbeaten)}; beat them once and they register themselves")
    if never:
        parts.append(f"beaten but not registered: {'; '.join(never)}; their script never offers it again")
    name = next((t.entry.map_name for t in state.on_map(map_dir)), map_dir)
    return (f"MATCH CALL REGISTRATION on {name}: " + ". ".join(parts) + ".") if parts else ""
