"""Medicine from the bag: what each item does, and which one a Pokémon needs.

Used by server/app.py (use_item, train's healing) and by the battle view's item line
(pokemon_env/memory_reader.py _battle_item_hint). Outside pokemon_env, whose package
import loads the emulator, so the server can import it at load time.
"""
from typing import Dict, List, Optional, Tuple

# pret include/constants/items.h ids, the HP each restores (src/data/items.h holdEffectParam;
# MEDICINE_FULL = all of it), the status1 bits it cures, whether it revives, and its Poké
# Mart price (used only to pick the cheapest that does the job). Herbs are left out: they
# lower friendship. STATUS1_TOXIC_POISON (0x80) is cured with POISON.
MEDICINE_FULL = 10 ** 6
STATUS_SLEEP, STATUS_POISON, STATUS_BURN, STATUS_FREEZE, STATUS_PARALYSIS = 0x7, 0x88, 0x10, 0x20, 0x40
STATUS_ANY = STATUS_SLEEP | STATUS_POISON | STATUS_BURN | STATUS_FREEZE | STATUS_PARALYSIS
MEDICINE: Dict[int, Tuple[int, int, bool, int]] = {
    # item id: (HP restored, status cured, revives, price)
    13: (20, 0, False, 300),                         # POTION
    14: (0, STATUS_POISON, False, 100),              # ANTIDOTE
    15: (0, STATUS_BURN, False, 250),                # BURN HEAL
    16: (0, STATUS_FREEZE, False, 250),              # ICE HEAL
    17: (0, STATUS_SLEEP, False, 250),               # AWAKENING
    18: (0, STATUS_PARALYSIS, False, 200),           # PARALYZE HEAL
    19: (MEDICINE_FULL, STATUS_ANY, False, 3000),    # FULL RESTORE
    20: (MEDICINE_FULL, 0, False, 2500),             # MAX POTION
    21: (200, 0, False, 1200),                       # HYPER POTION
    22: (50, 0, False, 700),                         # SUPER POTION
    23: (0, STATUS_ANY, False, 600),                 # FULL HEAL
    24: (0, 0, True, 1500),                          # REVIVE (to half HP)
    25: (0, 0, True, 4000),                          # MAX REVIVE (full HP)
    26: (50, 0, False, 200),                         # FRESH WATER
    27: (60, 0, False, 300),                         # SODA POP
    28: (80, 0, False, 350),                         # LEMONADE
    29: (100, 0, False, 500),                        # MOOMOO MILK
}


# PP items (pret include/constants/items.h ids; src/data/pokemon/item_effects.h): the PP
# each restores (MEDICINE_FULL = all of it) and whether it restores one move, picked in the
# party menu's "Restore which move?" list (ITEM4_HEAL_PP_ONE, ItemUseCB_PPRecovery), or
# every move. Seen live (2026-11-05, Elite Four): use_item("ETHER", "MANECTRIC",
# move="THUNDER") restored SPARK, the first move in that list.
PP_RESTORE: Dict[int, Tuple[int, bool]] = {
    34: (10, True),               # ETHER
    35: (MEDICINE_FULL, True),    # MAX ETHER
    36: (10, False),              # ELIXIR
    37: (MEDICINE_FULL, False),   # MAX ELIXIR
}
# PP UP and PP MAX raise one move's max PP, picked in the same list (ItemUseCB_PPUp): the
# number of PP Ups each applies (pret pokemon.c ExecuteTableBasedItemEffect: ITEM4_PP_UP
# adds one, ITEM5_PP_MAX sets three).
PP_BOOST: Dict[int, int] = {69: 1, 71: 3}  # PP UP, PP MAX
MAX_PP_UPS = 3


def pp_item_picks_a_move(item_id: int) -> bool:
    """The party menu asks which move: ETHER, MAX ETHER, PP UP, PP MAX."""
    return item_id in PP_BOOST or (item_id in PP_RESTORE and PP_RESTORE[item_id][1])


def pp_ups(pp_bonuses: int, index: int) -> int:
    """PP Ups applied to move slot `index` (pret gPPUpGetMask: two bits per move)."""
    return (pp_bonuses >> (2 * index)) & 3


def max_pp(base_pp: int, pp_bonuses: int, index: int) -> int:
    """pret pokemon.c CalculatePPWithBonus: base PP plus a fifth of it per PP Up."""
    return base_pp + base_pp * 20 * pp_ups(pp_bonuses, index) // 100


def pp_item_refusal(item_id: int, moves: List[Tuple[str, int, int, int]], index: Optional[int]) -> Optional[str]:
    """Why this PP item would do nothing (pret TryUsePPItem: "It won't have any effect.",
    nothing spent), else None. `moves` holds (name, PP, max PP, PP Ups) per move slot;
    `index` is the slot picked for an item that takes one."""
    if not pp_item_picks_a_move(item_id):
        if all(pp >= top for _, pp, top, _ in moves):
            return "every move has full PP"
        return None
    name, pp, top, ups = moves[index]
    if item_id in PP_BOOST:
        # ITEM4_PP_UP needs under 3 PP Ups and more than 4 max PP; ITEM5_PP_MAX, at least 5.
        if ups >= MAX_PP_UPS or top < 5:
            return f"{name}'s PP can't be raised any further ({ups} PP Ups, {top} max PP)"
        return None
    if pp >= top:
        return f"{name} has full PP ({pp}/{top})"
    return None


def most_pp_missing(moves: List[Tuple[str, int, int, int]]) -> Optional[int]:
    """The move slot with the most PP missing (the first of a tie), else None."""
    missing = [(top - pp, -i) for i, (_, pp, top, _) in enumerate(moves) if top > pp]
    return -max(missing)[1] if missing else None


def medicine_for(mon, bag: Dict[int, int]) -> Optional[int]:
    """The item a Pokémon needs most from what the bag holds ({item id: count}), else None.

    Fainted: the cheapest revive. A status: the cheapest item that cures it. Missing HP:
    the cheapest item that restores all of it; when none does, the one restoring the most
    (cheapest of those). emerald-new (2026-11-04), David's "Potions + Jagged Pass": train
    heals with the cheapest item that does the job before it walks to a Pokémon Center.
    """
    have = {i: MEDICINE[i] for i, n in bag.items() if n > 0 and i in MEDICINE}
    if mon.is_egg or not mon.max_hp:
        return None
    if mon.current_hp == 0:
        revives = [i for i, (_, _, revive, _) in have.items() if revive]
        return min(revives, key=lambda i: have[i][3], default=None)
    status = int(mon.status) & STATUS_ANY
    if status:
        cures = [i for i, (_, cured, _, _) in have.items() if cured & status]
        if cures:
            return min(cures, key=lambda i: have[i][3])
    missing = mon.max_hp - mon.current_hp
    if missing <= 0:
        return None
    heals = [i for i, (hp, _, _, _) in have.items() if hp > 0]
    enough = [i for i in heals if have[i][0] >= missing]
    if enough:
        return min(enough, key=lambda i: have[i][3])
    return min(heals, key=lambda i: (-have[i][0], have[i][3]), default=None)
