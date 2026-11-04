"""Menu, battle-flow and shop state behind the closed-loop macros (heal, run, catch,
switch, buy).

Each macro presses a button only after reading from RAM that the game shows exactly
the screen and cursor it expects, and stops with a reason when it doesn't. The state
is recognized by *which task or callback is running* (gTasks[].func,
gMain.callback2, gBattlerControllerFuncs) rather than by frame counts; the frame
budgets in server/app.py only guard against waiting forever. This follows the method
pokebot-gen3 documents for Emerald (same pret symbols); no code was copied from it.

Addresses are pret/pokeemerald symbols for the US ROM ("l" = file-static in the
.sym). Each one was checked on a savestate before use, by printing it next to a
screenshot: the nurse's YES/NO (Rustboro and Oldale centers), the Rustboro Mart's
BUY menu, quantity box and YES/NO, a wild battle's bag, ball throw, catch, nickname
prompt and Pokédex page, the battle party menu's SHIFT, and a successful run.

Function pointers are Thumb (bit 0 set), so they are compared with `| 1`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from server.field_control import (
    G_MAIN_CALLBACK2,
    G_OBJECT_EVENTS,
    MAP_OFFSET,
    OBJECT_EVENT_COUNT,
    OBJECT_EVENT_SIZE,
    decode_field_message,
)

# ---------------------------------------------------------------------------
# Generic engine state
# ---------------------------------------------------------------------------

# gTasks[16]: struct Task {func (4), isActive (1), prev, next, priority, s16 data[16]}.
G_TASKS = 0x03005E00
TASK_SIZE = 0x28
TASK_COUNT = 16
TASK_DATA = 8
# menu.c's sMenu (struct Menu): +2 cursorPos, +3 minCursorPos, +4 maxCursorPos. Every
# standard menu (YES/NO, BUY/SELL/QUIT, the bag's USE/CANCEL, the party SHIFT menu)
# moves this cursor; the .sym has two other static sMenu symbols in other files.
S_MENU = 0x0203CD90
# sTextPrinters[32] (text.c), one per window: +0x1B active, +0x1C state. States 1
# (RENDER_STATE_WAIT) and 2 (RENDER_STATE_CLEAR) wait for A/B with the down arrow.
# Seen live: every battle message ("Wild SHROOMISH appeared!", "Got away safely!")
# waits for A in a single-player battle, so the macros must press through them.
S_TEXT_PRINTERS = 0x020201B0
TEXT_PRINTER_SIZE = 0x24
TEXT_PRINTER_COUNT = 32
# RENDER_STATE_WAIT (1), _CLEAR (2) and _SCROLL_START (3) all show the down arrow and wait
# for a key (src/text.c). Seen in a test: "Wild SHROOMISH's EFFECT SPORE paralyzed
# TORCHIC! It may be unable to move!" is long enough to scroll, sat in state 3, and with
# only 1 and 2 counted use_move waited out its whole budget and reported a timeout.
TEXT_WAITING_STATES = (1, 2, 3)
# gPaletteFade: bit 7 of the byte at +7 is set while a screen fade runs.
G_PALETTE_FADE_FLAGS = 0x02037FD4 + 7
PALETTE_FADE_ACTIVE = 0x80

# Callbacks (gMain.callback2).
CB2_BATTLE_MAIN = 0x08038420       # BattleMainCB2
CB2_BAG_MENU_RUN = 0x081AAD5C      # CB2_BagMenuRun
CB2_UPDATE_PARTY_MENU = 0x081B01B0  # CB2_UpdatePartyMenu
CB2_BUY_MENU = 0x080DFD64          # CB2_BuyMenu

# Task functions.
TASK_FIELD_YES_NO = 0x080E215C          # Task_HandleYesNoInput (script yesnobox: the nurse)
TASK_SHOP_MENU = 0x080DFB88             # Task_ShopMenu (BUY / SELL / SEE YA!)
TASK_BUY_MENU = 0x080E0AC8              # Task_BuyMenu (the item list)
TASK_BUY_HOW_MANY = 0x080E0D88          # Task_BuyHowManyDialogueHandleInput
TASK_YES_NO_CALLBACK = 0x08121FDC       # Task_CallYesOrNoCallback (the purchase YES/NO)
TASK_AFTER_PURCHASE = 0x080E100C        # Task_ReturnToItemListAfterItemPurchase (waits for A)
TASK_BAG_INPUT = 0x081ABD28             # Task_BagMenu_HandleInput (the item list)
TASK_BAG_CONTEXT = (0x081ACC04, 0x081ACC78)  # Task_ItemContext_SingleRow / _MultipleRows
TASK_PARTY_CHOOSE = 0x081B1370          # Task_HandleChooseMonInput
TASK_PARTY_SELECTION = 0x081B3730       # Task_HandleSelectionMenuInput (SHIFT/SUMMARY/CANCEL)
TASK_CAUGHT_DEX_PAGE = 0x080C0088       # Task_HandleCaughtMonPageInput (waits for A/B)

# ---------------------------------------------------------------------------
# Battle
# ---------------------------------------------------------------------------

G_BATTLER_CONTROLLER_FUNCS = 0x03005D60
CONTROLLER_CHOOSE_ACTION = 0x08057588   # HandleInputChooseAction
# PlayerHandleYesNoInput. Seen on test_wild_two_party.state: "Use next POKéMON?" is the
# battle script's yesnobox (BATTLE_YES_NO_OPCODES), not this controller.
CONTROLLER_YES_NO = 0x080599D4
G_BATTLE_OUTCOME = 0x0202433A
B_OUTCOME_WON, B_OUTCOME_LOST, B_OUTCOME_DREW, B_OUTCOME_RAN = 1, 2, 3, 4
B_OUTCOME_PLAYER_TELEPORTED, B_OUTCOME_MON_FLED, B_OUTCOME_CAUGHT = 5, 6, 7
G_BATTLESCRIPT_CURR_INSTR = 0x02024214
G_BATTLE_COMMUNICATION = 0x02024332     # [0] MULTIUSE_STATE, [1] CURSOR_POSITION
G_DISPLAYED_STRING_BATTLE = 0x02022E2C
DISPLAYED_STRING_BATTLE_SIZE = 300
# Battle messages print in window 0 (B_WIN_MSG); sTextPrinters is indexed by window.
# The same string buffer also feeds the "What will TORCHIC do?" prompt (window 1), the
# move menu's PP/TYPE lines and the healthbox names ("TORCHIC♂"), so a buffer read only
# counts as a battle message while window 0's printer is active.
B_WIN_MSG = 0
# The nickname YES/NO runs `trygivecaughtmonnick` inside this script; its state 1
# waits for input, and B there means NO (Cmd_trygivecaughtmonnick).
BS_TRY_NICKNAME_CAUGHT_MON = 0x082DBDA5
BS_GIVE_CAUGHT_MON_END = 0x082DBDC2
BS_ASK_TO_LEARN_MOVE = 0x082DABE3
BS_ASK_TO_LEARN_MOVE_END = 0x082DAC10   # BattleScript_ForgotAndLearnedNewMove, the next label
# Battle script commands that put up a YES/NO and wait for the player: yesnoboxlearnmove
# (0x5A), yesnoboxstoplearningmove (0x5B) and yesnobox (0x67, "Will DAVID change
# POKéMON?" before a trainer's next Pokémon). Opcodes read from the ROM's
# gBattleScriptingCommandsTable (0x0831BD10) by matching the .sym's Cmd_* addresses.
# While one runs, gBattlescriptCurrInstr points at its opcode, and A would answer YES.
BATTLE_YES_NO_OPCODES = (0x5A, 0x5B, 0x67)
# drawlvlupbox: the stat panel after "X grew to LV. N!" waits for a key press of its
# own (gMain.newKeys), not through a text printer, so text_waiting stays False under it.
# Seen on test_wild_two_party.state: gBattlescriptCurrInstr 0x082DABC9 (in
# BattleScript_LevelUp) held opcode 0x6C until two A presses closed both panels; until
# then use_move sat out its whole 4800-frame budget after every level-up.
BATTLE_LEVEL_UP_BOX_OPCODE = 0x6C
ROM_START, ROM_END = 0x08000000, 0x0A000000
CB2_EVOLUTION_SCENE = 0x0813E3A4          # CB2_EvolutionSceneUpdate (after a battle)
G_BATTLER_PARTY_INDEXES = 0x0202406E     # u16[4]
G_BATTLE_MONS = 0x02024084
BATTLE_MON_SIZE = 0x58
BATTLE_MON_PERSONALITY = 0x48
ACTION_FIGHT, ACTION_BAG, ACTION_POKEMON, ACTION_RUN = 0, 1, 2, 3

# Party (gPlayerParty: 100-byte struct Pokemon; +0 personality, +0x56 hp, +0x58 maxHP).
G_PLAYER_PARTY_COUNT = 0x020244E9
G_PLAYER_PARTY = 0x020244EC
PARTY_MON_SIZE = 100

# Bag: gBagPosition +5 pocket, +8 cursorPosition[5], +0x12 scrollPosition[5].
G_BAG_POSITION = 0x0203CE58
BALLS_POCKET = 1
POCKET_NAMES = ("ITEMS", "POKé BALLS", "TMs & HMs", "BERRIES", "KEY ITEMS")
# gBagPockets[5]: struct BagPocket {struct ItemSlot *itemSlots; u8 capacity;}; the bag
# lists itemSlots in order, so the cursor index names the item under it.
G_BAG_POCKETS = 0x02039DD8
BAG_POCKET_SIZE = 8
G_SPECIAL_VAR_ITEM_ID = 0x0203CE7C       # set to the chosen item when its context menu opens

# Party menu: gPartyMenu +9 slotId; sPartyMenuInternal (pointer) +0xF actions[8], +0x17 numActions.
G_PARTY_MENU = 0x0203CEC8
S_PARTY_MENU_INTERNAL = 0x0203CEC4
MENU_SWITCH = 1  # party_menu.c MENU_SWITCH, the field selection menu's "SWITCH"
MENU_ITEM = 3    # party_menu.c MENU_ITEM: opens GIVE / TAKE / CANCEL
MENU_GIVE = 4
MENU_TAKE_ITEM = 5
ITEM_EXP_SHARE = 182                       # include/constants/items.h
FLAG_RECEIVED_EXP_SHARE = 0x110            # include/constants/flags.h (Mr. Stone, Devon 3F)
ITEMS_POCKET = 0                           # the bag's ITEMS pocket (where the Exp. Share goes)
# EXP to the next level (David's lead rule's tie-break): pret gSpeciesInfo[species]
# (28 bytes) +0x13 growthRate indexes gExperienceTables[growthRate][level] (u32[6][101]).
G_SPECIES_INFO = 0x083203CC
SPECIES_INFO_SIZE = 28
SPECIES_GROWTH_RATE = 0x13
G_EXPERIENCE_TABLES = 0x0831F72C
MAX_LEVEL = 100
MENU_SHIFT = 10  # party_menu.c MENU_SHIFT, the battle selection menu's first entry
# party_menu.c MENU_SEND_OUT: the same menu when a fainted Pokémon must be replaced.
MENU_SEND_OUT = 11

# Mart: sMartInfo +8 itemList (u16 *), +0xC itemCount; sShopData (pointer)
# +0x2006 selectedRow, +0x2008 scrollOffset, +0x200A maxQuantity (what money and bag
# room allow, set when the quantity box opens).
S_MART_INFO = 0x02039F60
S_SHOP_DATA = 0x02039F70
SHOP_SELECTED_ROW = 0x2006
SHOP_SCROLL_OFFSET = 0x2008
SHOP_MAX_QUANTITY = 0x200A
# The quantity task's data[1] is the count shown, data[5] the item being bought.
BUY_TASK_COUNT = 1
BUY_TASK_ITEM = 5
# gItems[]: 0x2C-byte struct Item, price (u16) at +0x10.
G_ITEMS = 0x085839A0
ITEM_SIZE = 0x2C
ITEM_PRICE = 0x10

# Object event graphics (include/constants/event_objects.h).
OBJ_EVENT_GFX_NURSE = 58
OBJ_EVENT_GFX_MART_EMPLOYEE = 83
# pokeemerald DIR_SOUTH=1, DIR_NORTH=2, DIR_WEST=3, DIR_EAST=4, as D-pad presses.
FACING_BUTTON = {1: "DOWN", 2: "UP", 3: "LEFT", 4: "RIGHT"}
OPPOSITE = {"DOWN": "UP", "UP": "DOWN", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def _is(func: int, address: int) -> bool:
    return (func | 1) == (address | 1)


def active_task_funcs(reader) -> List[int]:
    funcs = []
    for i in range(TASK_COUNT):
        base = G_TASKS + i * TASK_SIZE
        if reader._read_u8(base + 4):
            funcs.append(reader._read_u32(base))
    return funcs


def find_task(reader, func: int) -> Optional[int]:
    """Index of the active task running `func`, or None."""
    for i in range(TASK_COUNT):
        base = G_TASKS + i * TASK_SIZE
        if reader._read_u8(base + 4) and _is(reader._read_u32(base), func):
            return i
    return None


def task_data(reader, task_id: int, index: int) -> int:
    return reader._read_s16(G_TASKS + task_id * TASK_SIZE + TASK_DATA + 2 * index)


def task_running(reader, *funcs: int) -> bool:
    return any(find_task(reader, f) is not None for f in funcs)


@dataclass
class MenuCursor:
    cursor: int
    max_cursor: int


def read_menu_cursor(reader) -> MenuCursor:
    return MenuCursor(cursor=reader._read_u8(S_MENU + 2), max_cursor=reader._read_u8(S_MENU + 4))


def text_waiting_for_button(reader) -> bool:
    """True while any text printer shows the down arrow and waits for A/B."""
    for i in range(TEXT_PRINTER_COUNT):
        base = S_TEXT_PRINTERS + i * TEXT_PRINTER_SIZE
        if reader._read_u8(base + 0x1B) and reader._read_u8(base + 0x1C) in TEXT_WAITING_STATES:
            return True
    return False


# The task that plays a POKéNAV call in the field (src/match_call.c). A call is lines
# of text and an A to hang up; it never asks a question.
TASK_MATCH_CALL = 0x081960E0  # ExecuteMatchCall


def phone_call_active(reader) -> bool:
    return task_running(reader, TASK_MATCH_CALL)


def text_printing(reader) -> bool:
    """True while any text printer is still active: printing, or waiting on the arrow."""
    return any(
        reader._read_u8(S_TEXT_PRINTERS + i * TEXT_PRINTER_SIZE + 0x1B) for i in range(TEXT_PRINTER_COUNT)
    )


def fade_active(reader) -> bool:
    return bool(reader._read_u8(G_PALETTE_FADE_FLAGS) & PALETTE_FADE_ACTIVE)


def callback2_is(reader, address: int) -> bool:
    return _is(reader._read_u32(G_MAIN_CALLBACK2), address)


# ---------------------------------------------------------------------------
# Battle flow
# ---------------------------------------------------------------------------


@dataclass
class BattleFlow:
    """One read of where a battle stands, for deciding the next press."""

    in_battle: bool
    in_battle_screen: bool       # gMain.callback2 is BattleMainCB2 (not the bag/party/dex)
    choosing_action: bool        # FIGHT/BAG/POKEMON/RUN is waiting for the player
    yes_no_prompt: bool          # the battle's own YES/NO ("Use next POKéMON?")
    nickname_prompt: bool        # "Give a nickname to the captured X?" waiting for input
    learn_move_prompt: bool      # inside the "trying to learn X ... delete a move?" script
    dex_page: bool               # the caught Pokémon's Pokédex page, waiting for A/B
    text_waiting: bool
    outcome: int
    text: str
    tasks: List[int] = field(default_factory=list)
    script_question: bool = False  # a battle-script YES/NO is up (BATTLE_YES_NO_OPCODES)
    level_up_box: bool = False     # the level-up stat panel (BATTLE_LEVEL_UP_BOX_OPCODE)
    evolving: bool = False         # the evolution scene that follows a battle
    message_printing: bool = False  # the message window's text printer is active

    @property
    def asking(self) -> bool:
        """A YES/NO is up: A would answer it. "Use next POKéMON?" is a script yesnobox."""
        return self.yes_no_prompt or self.script_question or self.nickname_prompt


def read_battle_flow(reader) -> BattleFlow:
    from server.field_control import G_MAIN_IN_BATTLE_BYTE, IN_BATTLE_MASK

    controller = reader._read_u32(G_BATTLER_CONTROLLER_FUNCS)
    # The player's battlers are 0 and, in a double battle, 2 (B_POSITION_PLAYER_RIGHT).
    # Seen live: against PLUSLE and MINUN, after SABLEYE's move use_move waited for
    # battler 0's menu while COMBUSKEN (battler 2) was the one choosing, and timed out.
    partner_controller = reader._read_u32(G_BATTLER_CONTROLLER_FUNCS + 4 * 2)
    instr = reader._read_u32(G_BATTLESCRIPT_CURR_INSTR)
    multiuse_state = reader._read_u8(G_BATTLE_COMMUNICATION)
    opcode = reader._read_u8(instr) if ROM_START <= instr < ROM_END else None
    tasks = active_task_funcs(reader)
    return BattleFlow(
        in_battle=bool(reader._read_u8(G_MAIN_IN_BATTLE_BYTE) & IN_BATTLE_MASK),
        in_battle_screen=callback2_is(reader, CB2_BATTLE_MAIN),
        choosing_action=_is(controller, CONTROLLER_CHOOSE_ACTION) or _is(partner_controller, CONTROLLER_CHOOSE_ACTION),
        yes_no_prompt=_is(controller, CONTROLLER_YES_NO) or _is(partner_controller, CONTROLLER_YES_NO),
        nickname_prompt=BS_TRY_NICKNAME_CAUGHT_MON <= instr < BS_GIVE_CAUGHT_MON_END and multiuse_state == 1,
        learn_move_prompt=BS_ASK_TO_LEARN_MOVE <= instr < BS_ASK_TO_LEARN_MOVE_END,
        dex_page=any(_is(f, TASK_CAUGHT_DEX_PAGE) for f in tasks),
        text_waiting=text_waiting_for_button(reader),
        outcome=reader._read_u8(G_BATTLE_OUTCOME),
        text=decode_field_message(reader._read_bytes(G_DISPLAYED_STRING_BATTLE, DISPLAYED_STRING_BATTLE_SIZE)),
        tasks=tasks,
        script_question=opcode in BATTLE_YES_NO_OPCODES,
        level_up_box=opcode == BATTLE_LEVEL_UP_BOX_OPCODE,
        evolving=callback2_is(reader, CB2_EVOLUTION_SCENE),
        message_printing=bool(reader._read_u8(S_TEXT_PRINTERS + B_WIN_MSG * TEXT_PRINTER_SIZE + 0x1B)),
    )


def active_battler_personality(reader) -> int:
    return reader._read_u32(G_BATTLE_MONS + BATTLE_MON_PERSONALITY)


# ---------------------------------------------------------------------------
# Party
# ---------------------------------------------------------------------------


def exp_to_next_level(reader, species_id: int, level: int, experience: int) -> int:
    """EXP a Pokémon still needs for its next level (0 at level 100), from the ROM's
    tables: gExperienceTables[gSpeciesInfo[species].growthRate][level + 1] - experience."""
    if level >= MAX_LEVEL:
        return 0
    growth = reader._read_u8(G_SPECIES_INFO + SPECIES_INFO_SIZE * species_id + SPECIES_GROWTH_RATE)
    needed = reader._read_u32(G_EXPERIENCE_TABLES + 4 * ((MAX_LEVEL + 1) * growth + level + 1))
    return max(0, needed - int(experience or 0))


def party_personalities(reader) -> List[int]:
    count = min(reader._read_u8(G_PLAYER_PARTY_COUNT), 6)
    return [reader._read_u32(G_PLAYER_PARTY + i * PARTY_MON_SIZE) for i in range(count)]


def party_menu_slot(reader) -> int:
    slot = reader._read_u8(G_PARTY_MENU + 9)
    return slot - 256 if slot >= 128 else slot


def party_menu_switch_slot(reader) -> int:
    """gPartyMenu.slotId2 (+0xA): the cursor while SWITCH asks "Move to where?" (pret
    GetCurrentPartySlotPtr returns it for PARTY_ACTION_SWITCH; slotId stays on the
    Pokémon being moved)."""
    slot = reader._read_u8(G_PARTY_MENU + 0xA)
    return slot - 256 if slot >= 128 else slot


def party_menu_actions(reader) -> List[int]:
    internal = reader._read_u32(S_PARTY_MENU_INTERNAL)
    if not internal:
        return []
    count = min(reader._read_u8(internal + 0x17), 8)
    return [reader._read_u8(internal + 0xF + i) for i in range(count)]


# ---------------------------------------------------------------------------
# Bag
# ---------------------------------------------------------------------------


def bag_pocket(reader) -> int:
    return reader._read_u8(G_BAG_POSITION + 5)


def bag_cursor_index(reader, pocket: int) -> int:
    cursor = reader._read_u16(G_BAG_POSITION + 8 + 2 * pocket)
    scroll = reader._read_u16(G_BAG_POSITION + 0x12 + 2 * pocket)
    return cursor + scroll


def bag_pocket_item_ids(reader, pocket: int) -> List[int]:
    """Item ids in the order the bag lists them (up to the first empty slot)."""
    base = G_BAG_POCKETS + pocket * BAG_POCKET_SIZE
    slots, capacity = reader._read_u32(base), reader._read_u8(base + 4)
    ids = []
    for i in range(capacity):
        item_id = reader._read_u16(slots + 4 * i)
        if item_id == 0:
            break
        ids.append(item_id)
    return ids


def pocket_presses(current: int, goal: int) -> List[str]:
    """LEFT/RIGHT switch bag pockets one at a time (no wrap-around is assumed)."""
    if goal > current:
        return ["RIGHT"] * (goal - current)
    return ["LEFT"] * (current - goal)


def list_presses(current: int, goal: int) -> List[str]:
    if goal > current:
        return ["DOWN"] * (goal - current)
    return ["UP"] * (current - goal)


# ---------------------------------------------------------------------------
# START menu, TMs & HMs, the move-replace screen (teach_move). pret symbols (US) and
# struct layouts; each was watched in memory on test_marill_in_party.state.
# ---------------------------------------------------------------------------

S_START_MENU_CURSOR_POS = 0x0203760E      # sStartMenuCursorPos: index into the actions
S_NUM_START_MENU_ACTIONS = 0x0203760F
S_CURRENT_START_MENU_ACTIONS = 0x02037610  # u8[9], start_menu.c MENU_ACTION_*
TASK_SHOW_START_MENU = 0x0809FA34
MENU_ACTION_POKEMON = 1                    # start_menu.c MENU_ACTION_POKEMON
MENU_ACTION_BAG = 2
TMHM_POCKET = 2                            # POKé BALLS is 1 (BALLS_POCKET)
ITEM_TM01 = 289                            # TM01..TM50 = 289..338, HM01..HM08 = 339..346
G_TMHM_LEARNSETS = 0x0831E898             # u32[2] per species: bit n = TM/HM n
S_TMHM_MOVES = 0x08616040                  # u16 per TM/HM: the move it teaches
CB2_SUMMARY_SCREEN = 0x081BFAB4            # seen as gMain.callback2 on the forget screen
TASK_REPLACE_MOVE_INPUT = 0x081C174C       # Task_HandleReplaceMoveInput
S_MON_SUMMARY_SCREEN = 0x0203CF1C          # pointer to struct PokemonSummaryScreenData
SUMMARY_NEW_MOVE = 0x40C4                  # u16 newMove (0x70 + 0x4C summary + 4 pages x 0x1000)
SUMMARY_FIRST_MOVE_INDEX = 0x40C6          # u8 firstMoveIndex: the forget cursor (4 = the new move)


def start_menu_open(reader) -> bool:
    return task_running(reader, TASK_SHOW_START_MENU)


def start_menu_actions(reader) -> List[int]:
    count = min(reader._read_u8(S_NUM_START_MENU_ACTIONS), 9)
    return list(bytes(reader._read_bytes(S_CURRENT_START_MENU_ACTIONS, count)))


def start_menu_cursor(reader) -> int:
    return reader._read_u8(S_START_MENU_CURSOR_POS)


def tmhm_move(reader, item_id: int) -> int:
    """The move id a TM/HM item teaches (ROM sTMHMMoves)."""
    return reader._read_u16(S_TMHM_MOVES + 2 * (item_id - ITEM_TM01))


def can_learn_tmhm(reader, species_id: int, item_id: int) -> bool:
    """pret CanMonLearnTMHM: gTMHMLearnsets[species] has the TM/HM's bit."""
    index = item_id - ITEM_TM01
    word = reader._read_u32(G_TMHM_LEARNSETS + 8 * species_id + 4 * (index // 32))
    return bool(word >> (index % 32) & 1)


def replace_move_cursor(reader) -> Optional[Tuple[int, int]]:
    """(firstMoveIndex, newMove) on the forget-a-move screen, or None when it isn't up."""
    if not (callback2_is(reader, CB2_SUMMARY_SCREEN) and task_running(reader, TASK_REPLACE_MOVE_INPUT)):
        return None
    data = reader._read_u32(S_MON_SUMMARY_SCREEN)
    if not data:
        return None
    return reader._read_u8(data + SUMMARY_FIRST_MOVE_INDEX), reader._read_u16(data + SUMMARY_NEW_MOVE)


# ---------------------------------------------------------------------------
# Mart
# ---------------------------------------------------------------------------


def mart_stock(reader) -> List[int]:
    items_ptr = reader._read_u32(S_MART_INFO + 8)
    count = reader._read_u16(S_MART_INFO + 0xC)
    if not items_ptr or count > 64:
        return []
    return [reader._read_u16(items_ptr + 2 * i) for i in range(count)]


def shop_list_index(reader) -> Optional[int]:
    data = reader._read_u32(S_SHOP_DATA)
    if not data:
        return None
    return reader._read_u16(data + SHOP_SELECTED_ROW) + reader._read_u16(data + SHOP_SCROLL_OFFSET)


def shop_max_quantity(reader) -> Optional[int]:
    data = reader._read_u32(S_SHOP_DATA)
    return reader._read_u8(data + SHOP_MAX_QUANTITY) if data else None


def item_price(reader, item_id: int) -> int:
    return reader._read_u16(G_ITEMS + item_id * ITEM_SIZE + ITEM_PRICE)


def quantity_presses(current: int, goal: int) -> List[str]:
    """Shop quantity box: RIGHT/LEFT change by 10, UP/DOWN by 1 (goal within 1..max)."""
    diff = goal - current
    tens, ones = divmod(abs(diff), 10)
    if diff >= 0:
        return ["RIGHT"] * tens + ["UP"] * ones
    return ["LEFT"] * tens + ["DOWN"] * ones


def name_key(text: str) -> str:
    """'Poke Ball', 'POKé BALL' and 'poke_ball' all name the same item."""
    return "".join(ch for ch in str(text).upper().replace("É", "E") if ch.isalnum())


# ---------------------------------------------------------------------------
# People behind counters
# ---------------------------------------------------------------------------


@dataclass
class CounterPerson:
    x: int
    y: int
    facing: str               # the way they face (toward customers)
    stand: Tuple[int, int]    # the tile to talk from
    face: str                 # the way to face from there
    across_counter: bool


def find_objects_by_graphics(reader, graphics_id: int) -> List[Tuple[int, int, str]]:
    """(x, y, facing button) of every visible object event using this sprite.

    gObjectEvents holds the people loaded on the current map; graphicsId (+5) names
    the sprite, which is the same for every Pokémon Center nurse and Mart clerk.
    """
    found = []
    for i in range(OBJECT_EVENT_COUNT):
        base = G_OBJECT_EVENTS + i * OBJECT_EVENT_SIZE
        flags = reader._read_u32(base)
        if not flags & 0x1 or flags & (1 << 13):  # inactive or invisible
            continue
        if reader._read_u8(base + 5) != graphics_id:
            continue
        x = reader._read_s16(base + 0x10) - MAP_OFFSET
        y = reader._read_s16(base + 0x12) - MAP_OFFSET
        facing = FACING_BUTTON.get(reader._read_u16(base + 0x18) & 0xF, "DOWN")
        found.append((x, y, facing))
    return found


OBJ_EVENT_ID_PLAYER = 0xFF  # struct ObjectEvent localId (+8) of the player's own object


def people_near(reader, x: int, y: int, radius: int) -> List[Tuple[int, int]]:
    """(x, y) of visible object events other than the player within `radius` steps of
    (x, y), nearest first. For someone a script can move (the Cable Car attendant)."""
    found = []
    for i in range(OBJECT_EVENT_COUNT):
        base = G_OBJECT_EVENTS + i * OBJECT_EVENT_SIZE
        flags = reader._read_u32(base)
        if not flags & 0x1 or flags & (1 << 13) or reader._read_u8(base + 8) == OBJ_EVENT_ID_PLAYER:
            continue
        ox = reader._read_s16(base + 0x10) - MAP_OFFSET
        oy = reader._read_s16(base + 0x12) - MAP_OFFSET
        if abs(ox - x) + abs(oy - y) <= radius:
            found.append((ox, oy))
    return sorted(found, key=lambda p: abs(p[0] - x) + abs(p[1] - y))


def talk_spot(x: int, y: int, facing: str, counter_between: bool) -> CounterPerson:
    """Where to stand to talk to a person who faces customers, maybe across a counter.

    Every Pokémon Center nurse and Mart clerk stands behind a counter facing the
    customers' side; the game lets the player talk across a counter tile
    (MB_COUNTER), so the talking tile is two steps out when a counter is in between.
    """
    from server.field_control import DIRECTION_DELTAS

    dx, dy = DIRECTION_DELTAS[facing]
    steps = 2 if counter_between else 1
    return CounterPerson(
        x=x,
        y=y,
        facing=facing,
        stand=(x + dx * steps, y + dy * steps),
        face=OPPOSITE[facing],
        across_counter=counter_between,
    )


def battle_text(reader) -> str:
    return decode_field_message(reader._read_bytes(G_DISPLAYED_STRING_BATTLE, DISPLAYED_STRING_BATTLE_SIZE))


def outcome_name(outcome: int) -> str:
    return {
        B_OUTCOME_WON: "won",
        B_OUTCOME_LOST: "lost",
        B_OUTCOME_DREW: "drew",
        B_OUTCOME_RAN: "ran away",
        B_OUTCOME_PLAYER_TELEPORTED: "teleported away",
        B_OUTCOME_MON_FLED: "the wild Pokémon fled",
        B_OUTCOME_CAUGHT: "caught",
    }.get(outcome, f"outcome {outcome}")
