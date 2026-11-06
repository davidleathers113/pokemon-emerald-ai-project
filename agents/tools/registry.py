"""Centralized tool declaration registry with per-scaffold availability.

Each entry in ``TOOL_REGISTRY`` carries a ``scaffolds`` field that is either
the sentinel string ``"all"`` (available to every scaffold) or a ``set`` of
scaffold names where the tool should be declared.

``build_tools_for_scaffold(scaffold)`` returns the filtered list of Gemini-
compatible tool declarations for the requested scaffold.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from agents.subagents.utils.registry import (
    BUILTIN_SUBAGENT_TOOL_NAMES,
    LOCAL_SUBAGENT_SPECS,
)
from agents.subagents.planner import REPLAN_OBJECTIVES_TOOL_DECLARATION

# ---------------------------------------------------------------------------
# Scaffold tier constants
# ---------------------------------------------------------------------------

ALL_SCAFFOLDS = "all"

# Scaffolds that get the expert-level tools (pathfinding, walkthrough, wiki)
EXPERT_SCAFFOLDS = frozenset({"pokeagent", "autonomous_cli"})

# Local addition: also give the lightweight "simple" scaffold (used with small
# local models) pathfinding, the walkthrough and the wiki lookup.
GUIDE_SCAFFOLDS = EXPERT_SCAFFOLDS | {"simple"}

# Scaffolds where built-in subagents are excluded, but generic primitives
# (skills, code, subagent CRUD, custom subagents, etc.) are available.
NO_BUILTINS_SCAFFOLDS = frozenset({"simple", "continualharness"})

# Union of all scaffolds except simplest — used for tools that every scaffold
# except the bare-minimum one should have.
_ALL_EXCEPT_SIMPLEST = frozenset(
    {"pokeagent", "autonomous_cli", "simple", "continualharness", "simplest"}
)
_STANDARD_SCAFFOLDS = frozenset(
    {"pokeagent", "autonomous_cli", "simple", "continualharness"}
)


# Each decision costs a full model call (tens of seconds with a local model), while the
# server queues any number of buttons and the agent waits for all of them. Pressing a
# planned sequence per decision is therefore much faster than one button per decision.
_BUTTON_SEQUENCE_GUIDANCE = (
    " Press a whole planned SEQUENCE in one call whenever you can predict the result: "
    "the buttons run in order and you see the screen again when they finish. Examples: "
    "walk 4 tiles north ['UP','UP','UP','UP']; read through dialogue ['A','A','A']; "
    "to attack in a battle use use_move with the move's name instead (the cursor starts "
    "on the first move, which is often not the one you want). "
    "Stop the sequence (end the list) right BEFORE anything you need to look at: a "
    "YES/NO question, the result of a battle turn, a turn into tall grass or a "
    "trainer's line of sight, or a menu whose cursor position you are unsure of. "
    "For walking more than a few tiles use navigate_to; to go to another map use go_to_exit. "
    "If a step is BLOCKED the rest of the list is skipped and the result says so."
)


def _press_buttons_description() -> str:
    gt = os.environ.get("GAME_TYPE", "emerald").lower()
    if gt == "red":
        return (
            "Press Game Boy buttons to interact with the game. Available "
            "buttons: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, WAIT "
            "(Game Boy has no L/R shoulder buttons). Use WAIT to observe "
            "without pressing any button." + _BUTTON_SEQUENCE_GUIDANCE
        )
    return (
        "Press Game Boy Advance buttons to interact with the game. "
        "Available buttons: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, "
        "L, R, WAIT. Use WAIT to observe without pressing any button." + _BUTTON_SEQUENCE_GUIDANCE
    )


# ---------------------------------------------------------------------------
# Registry: each entry is {name, description, parameters, scaffolds}
# ---------------------------------------------------------------------------

TOOL_REGISTRY: List[Dict[str, Any]] = [
    # -- press_buttons: every scaffold --
    {
        "name": "press_buttons",
        "scaffolds": ALL_SCAFFOLDS,
        "description": _press_buttons_description(),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "buttons": {
                    "type_": "ARRAY",
                    "items": {"type_": "STRING"},
                    "description": (
                        "Buttons to press in order, e.g. ['UP','UP','UP'] to walk 3 tiles, "
                        "['A','A'] to advance two text boxes, ['B'] to close a menu"
                    ),
                },
                "reasoning": {
                    "type_": "STRING",
                    "description": (
                        "REQUIRED FORMAT: Must include 'ANALYZE: [game screen, "
                        "location, objective, situation]' and 'PLAN: [action, "
                        "reason, expected result]'. Example: 'ANALYZE: Dialogue "
                        "box visible. Location: Route 101 (5,8). Objective: Talk "
                        "to Prof Birch. Situation: Birch asking for help. PLAN: "
                        "Action: Press A. Reason: Advance dialogue. Expected: "
                        "Dialogue continues or battle starts.'"
                    ),
                },
            },
            "required": ["buttons", "reasoning"],
        },
    },
    # -- complete_direct_objective: all except simplest --
    {
        "name": "complete_direct_objective",
        "scaffolds": _STANDARD_SCAFFOLDS,
        "description": (
            "Mark an objective done once the screen or state shows you achieved it, "
            "and advance to the next one. Pass the objective's ID exactly as shown "
            "in OBJECTIVES (e.g. 'early_012'); the harness works out its category."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "reasoning": {
                    "type_": "STRING",
                    "description": (
                        "REQUIRED FORMAT: Must include 'ANALYZE: [current state, "
                        "objective requirements, completion evidence]' and "
                        "'PLAN: [confirm completion, next objective]'. Example: "
                        "'ANALYZE: Objective was to reach Route 101. Current "
                        "location shows Route 101 at (15,5). Evidence: Game text "
                        "shows \"Route 101\". PLAN: Objective complete, marking "
                        "as done. Next: Talk to Prof Birch.'"
                    ),
                },
                "objective_id": {
                    "type_": "STRING",
                    "description": (
                        "ID of the objective you achieved, copied from OBJECTIVES "
                        "(e.g. 'early_012' or 'battle_003')."
                    ),
                },
                "category": {
                    "type_": "STRING",
                    "enum": ["story", "battling", "dynamics"],
                    "description": (
                        "Optional; derived from objective_id. Only needed by callers "
                        "that complete by category without an ID."
                    ),
                },
            },
            "required": ["objective_id", "reasoning"],
        },
    },
    # -- process_memory: every scaffold --
    {
        "name": "process_memory",
        "scaffolds": ALL_SCAFFOLDS,
        "description": (
            "Manage long-term memory. The <long_term_memory> section of your "
            "prompt shows all entry IDs. Use 'read' to get full details, "
            "'add' to create, 'update' to modify, 'delete' to remove."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "action": {
                    "type_": "STRING",
                    "enum": ["read", "add", "update", "delete"],
                    "description": "The action to perform on memory entries.",
                },
                "entries": {
                    "type_": "ARRAY",
                    "items": {
                        "type_": "OBJECT",
                        "properties": {
                            "id": {
                                "type_": "STRING",
                                "description": (
                                    "Entry ID or name (required for read/update/"
                                    "delete). For add, optionally specify a "
                                    "descriptive ID (e.g. 'route_101_map'). You "
                                    "can look up entries by ID or title."
                                ),
                            },
                            "path": {
                                "type_": "STRING",
                                "description": (
                                    "Category path, e.g. 'navigation/routes'. "
                                    "Defaults to 'uncategorized'."
                                ),
                            },
                            "title": {
                                "type_": "STRING",
                                "description": "Short descriptive title (required for add).",
                            },
                            "content": {
                                "type_": "STRING",
                                "description": "Detailed content text (required for add).",
                            },
                            "importance": {
                                "type_": "INTEGER",
                                "description": "1-5 importance rating (default 3).",
                            },
                            "location": {
                                "type_": "STRING",
                                "description": "Game location this memory relates to.",
                            },
                        },
                    },
                    "description": (
                        "For read: [{id}]. For add: [{path, title, content}] — "
                        "title AND content required. For update: [{id, ...fields}]. "
                        "For delete: [{id}]."
                    ),
                },
                "reasoning": {
                    "type_": "STRING",
                    "description": (
                        "Required. Brief justification for this memory operation "
                        "(what you are trying to learn or change and why)."
                    ),
                },
            },
            "required": ["action", "entries", "reasoning"],
        },
    },
    # -- process_skill: all except simplest --
    {
        "name": "process_skill",
        "scaffolds": _STANDARD_SCAFFOLDS,
        "description": (
            "Manage the skill library. The <skill_library> section of your "
            "prompt shows all skill IDs. Use 'read' to get full details, "
            "'add' to create, 'update' to modify, 'delete' to remove."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "action": {
                    "type_": "STRING",
                    "enum": ["read", "add", "update", "delete"],
                    "description": "The action to perform on skill entries.",
                },
                "entries": {
                    "type_": "ARRAY",
                    "items": {
                        "type_": "OBJECT",
                        "properties": {
                            "id": {
                                "type_": "STRING",
                                "description": (
                                    "Entry ID (required for read/update/delete). "
                                    "For add, optionally specify a descriptive ID "
                                    "(e.g. 'move_to_coords'). You can look up "
                                    "entries by ID or name."
                                ),
                            },
                            "name": {
                                "type_": "STRING",
                                "description": "Skill name (required for add).",
                            },
                            "description": {
                                "type_": "STRING",
                                "description": (
                                    "What the skill does and when to use it "
                                    "(required for add)."
                                ),
                            },
                            "path": {
                                "type_": "STRING",
                                "description": (
                                    "Category path, e.g. 'battle/type_matchups'. "
                                    "Defaults to 'general'."
                                ),
                            },
                            "effectiveness": {
                                "type_": "STRING",
                                "description": "'low', 'medium', or 'high'.",
                            },
                            "code": {
                                "type_": "STRING",
                                "description": (
                                    "Optional executable Python code for the skill. "
                                    "The code receives a `tools` dict with callable "
                                    "functions (e.g. tools['press_buttons'](buttons="
                                    "[...], reasoning=...), tools['get_game_state']"
                                    "()). Return value is sent back to you."
                                ),
                            },
                        },
                    },
                    "description": (
                        "For read: [{id}]. For add: [{name, description}] — both "
                        "required; optionally include 'code' for executable skills. "
                        "For update: [{id, ...fields}]. For delete: [{id}]."
                    ),
                },
                "reasoning": {
                    "type_": "STRING",
                    "description": (
                        "Required. Brief justification for this skill operation "
                        "(what strategy you are recording or updating and why)."
                    ),
                },
            },
            "required": ["action", "entries", "reasoning"],
        },
    },
    # -- run_skill: all except simplest --
    {
        "name": "run_skill",
        "scaffolds": _STANDARD_SCAFFOLDS,
        "description": (
            "Execute a skill's code. The skill must have a 'code' field "
            "containing Python. The code runs in a sandbox with access to "
            "game tools via a `tools` dict (e.g. tools['press_buttons']"
            "(buttons=['A'], reasoning='...'), tools['get_game_state']()). "
            "Use this for executable skills like custom pathfinding or "
            "battle routines."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "skill_id": {
                    "type_": "STRING",
                    "description": "ID of the skill to execute (e.g. 'skill_0001')",
                },
                "reasoning": {
                    "type_": "STRING",
                    "description": "Why you are running this skill now",
                },
                "args": {
                    "type_": "OBJECT",
                    "description": (
                        "REQUIRED for executable skills. Key-value arguments "
                        "passed to the skill code. Example for move_to_coords: "
                        '{"x": 5, "y": 12}. Check the skill\'s description for '
                        "what args it expects."
                    ),
                    "properties": {
                        "x": {
                            "type_": "INTEGER",
                            "description": "Target X coordinate (for navigation skills)",
                        },
                        "y": {
                            "type_": "INTEGER",
                            "description": "Target Y coordinate (for navigation skills)",
                        },
                    },
                },
            },
            "required": ["skill_id", "reasoning", "args"],
        },
    },
    # -- run_code: all except simplest --
    {
        "name": "run_code",
        "scaffolds": _STANDARD_SCAFFOLDS,
        "description": (
            "Execute arbitrary Python code in the game sandbox. Use this to "
            "prototype, debug, and test code BEFORE saving it as a skill. "
            "The code has access to the same `tools` dict as run_skill "
            "(press_buttons, get_game_state, etc.) plus `args`. Set `result` "
            "to return data. Use this to: inspect get_game_state() output, "
            "test map parsing, prototype pathfinding, debug skill code. "
            "Stdout from print() is captured."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "code": {
                    "type_": "STRING",
                    "description": (
                        "Python code to execute. Has access to: "
                        "tools['press_buttons'](), tools['get_game_state'](), "
                        "args, random, collections, heapq, numpy/np, json, re, "
                        "math. Set 'result' variable to return data."
                    ),
                },
                "reasoning": {
                    "type_": "STRING",
                    "description": "What you are testing or prototyping",
                },
                "args": {
                    "type_": "OBJECT",
                    "description": "Optional arguments passed as the `args` dict",
                    "properties": {},
                },
            },
            "required": ["code", "reasoning"],
        },
    },
    # -- process_subagent: all except simplest --
    {
        "name": "process_subagent",
        "scaffolds": _STANDARD_SCAFFOLDS,
        "description": (
            "Manage the subagent registry. The <subagent_registry> section of "
            "your prompt shows all subagent IDs. Use 'read' to get full "
            "config, 'add' to create, 'update' to modify, 'delete' to "
            "remove (built-ins cannot be deleted)."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "action": {
                    "type_": "STRING",
                    "enum": ["read", "add", "update", "delete"],
                    "description": "The action to perform on subagent entries.",
                },
                "entries": {
                    "type_": "ARRAY",
                    "items": {
                        "type_": "OBJECT",
                        "properties": {
                            "id": {
                                "type_": "STRING",
                                "description": (
                                    "Entry ID or name (required for read/update/"
                                    "delete). For add, optionally specify a "
                                    "descriptive ID (e.g. 'battle_handler'). You "
                                    "can look up entries by ID or name."
                                ),
                            },
                            "name": {
                                "type_": "STRING",
                                "description": "Subagent name (required for add).",
                            },
                            "description": {
                                "type_": "STRING",
                                "description": "What the subagent does (required for add).",
                            },
                            "system_instructions": {
                                "type_": "STRING",
                                "description": "System prompt for the subagent (max 12000 chars).",
                            },
                            "directive": {
                                "type_": "STRING",
                                "description": "Per-invocation directive (max 12000 chars).",
                            },
                            "handler_type": {
                                "type_": "STRING",
                                "description": "'one_step' or 'looping' (default 'looping').",
                            },
                            "max_turns": {
                                "type_": "INTEGER",
                                "description": "Max turns for looping subagents (default 25).",
                            },
                            "return_condition": {
                                "type_": "STRING",
                                "description": "When the subagent should return control.",
                            },
                        },
                    },
                    "description": (
                        "For read: [{id}]. For add: [{name, description, "
                        "system_instructions?, directive?}] — name AND description "
                        "required. For update: [{id, ...fields}]. For delete: [{id}]."
                    ),
                },
                "reasoning": {
                    "type_": "STRING",
                    "description": "Required. Brief justification for this subagent operation.",
                },
            },
            "required": ["action", "entries", "reasoning"],
        },
    },
    # -- get_progress_summary: expert scaffolds only --
    {
        "name": "get_progress_summary",
        "scaffolds": EXPERT_SCAFFOLDS,
        "description": (
            "Get progress: milestones, current location/coords, "
            "direct-objective status, completed objectives history, memory "
            "tree overview, and run directory. Call with no arguments."
        ),
        "parameters": {"type_": "OBJECT", "properties": {}, "required": []},
    },
    # -- navigate_to: expert scaffolds only --
    {
        "name": "navigate_to",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "Walk to a tile on the CURRENT map using A* pathfinding (to go to "
            "another map use go_to_exit). The result says whether you reached "
            "the tile. IMPORTANT: Always specify the variance parameter. "
            "If you get blocked repeatedly at the same position, increase "
            "variance to 'medium', 'high', or 'extreme' to explore "
            "alternative paths."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "x": {"type_": "INTEGER", "description": "Target X coordinate"},
                "y": {"type_": "INTEGER", "description": "Target Y coordinate"},
                "variance": {
                    "type_": "STRING",
                    "description": (
                        "REQUIRED. Path variance level: 'none' (optimal path, "
                        "use first), 'low' (1-step variation), 'medium' (3-step "
                        "variation, use if blocked), 'high' (5-step variation, "
                        "use if very stuck), 'extreme' (8-step variation, use as "
                        "last resort). Default: 'none'"
                    ),
                    "enum": ["none", "low", "medium", "high", "extreme"],
                },
                "reason": {
                    "type_": "STRING",
                    "description": (
                        "REQUIRED FORMAT: Must include 'ANALYZE: [current "
                        "location, objective, destination details]' and "
                        "'PLAN: [why navigating here, what to do when arrive]'. "
                        "Example: 'ANALYZE: Currently at Littleroot (8,10). "
                        "Objective: Meet Prof Birch. Destination: Route 101 "
                        "entrance at (15,5). PLAN: Navigate to encounter Birch "
                        "being attacked. Will save him to progress story.'"
                    ),
                },
                "consider_npcs": {
                    "type_": "BOOLEAN",
                    "description": (
                        "Whether to treat NPCs as obstacles during pathfinding. "
                        "Set to true to avoid NPCs (recommended for most "
                        "navigation). Set to false only if NPCs are "
                        "wandering/moving and you want to ignore them."
                    ),
                },
            },
            "required": ["x", "y", "variance", "reason", "consider_npcs"],
        },
    },
    # -- go_to_exit: wherever navigate_to is offered --
    # Local addition. Every harness that gets small models across maps lets the model
    # pick a destination and leaves the walking to a pathfinder; picking an EXITS entry
    # is the destination choice a model makes reliably.
    {
        "name": "go_to_exit",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "Leave the current map through one of its EXITS: walks there with "
            "pathfinding and steps off the map edge or through the door/stairs. "
            "Use it to go to another town, route or building. The result says "
            "where you arrived, or why it failed."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "exit": {
                    "type_": "STRING",
                    "description": (
                        "An entry of the EXITS list: a direction ('north', 'south', "
                        "'east', 'west') or the destination name (e.g. 'OLDALE TOWN', "
                        "'POKEMON CENTER')."
                    ),
                },
                "reasoning": {
                    "type_": "STRING",
                    "description": "One short sentence: why go there.",
                },
            },
            "required": ["exit", "reasoning"],
        },
    },
    # -- use_move: wherever go_to_exit is offered --
    # Local addition. Seen live: battles open with the move cursor on slot 1 and the
    # model mashed A, so a Torchic used SCRATCH for 12+ minutes with EMBER in slot 4.
    # Naming the move is a choice the model makes reliably; steering the cursor is not.
    {
        "name": "use_move",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "In a battle, use one of your Pokemon's moves: opens FIGHT if needed, "
            "moves the cursor to the move and confirms it, then reads through the turn's "
            "text. Answers at your next turn, when the battle ends, or at a question, "
            "quoting what happened and both sides' HP. The battle section names the best "
            "move. Fails without pressing anything if the move has no PP. In a double "
            "battle it hits the foe named by target, else the one the battle section's "
            "Target line names (the bigger threat)."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "move": {
                    "type_": "STRING",
                    "description": "The move's name as listed in the battle section (e.g. 'EMBER'), or its slot number 1-4.",
                },
                "target": {
                    "type_": "STRING",
                    "description": "Double battles only, optional: the foe to hit, e.g. 'XATU'.",
                },
                "reasoning": {
                    "type_": "STRING",
                    "description": "One short sentence: why this move.",
                },
            },
            "required": ["move", "reasoning"],
        },
    },
    # -- macros: wherever use_move is offered --
    # Local addition. Each one does a whole menu chore in one call, checking game
    # memory before every press and stopping with the reason when the screen isn't
    # what it expects: fewer ~30 s decisions, and none of them inside a menu.
    {
        "name": "go_to",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "Walk to a named place in one call: a town, route, building or cave floor, map "
            "after map (e.g. 'Route 117', 'Mauville City Pokemon Center'). Runs from wild "
            "battles on the way; stops for a trainer battle or a question. Flies to a "
            "visited town first when that is much shorter (a Pokemon knows FLY, Feather Badge)."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "place": {"type_": "STRING", "description": "Where to go: a map name as the objectives write it."},
                "reasoning": {"type_": "STRING", "description": "One short sentence: why go there."},
            },
            "required": ["place", "reasoning"],
        },
    },
    # David (2026-11-04): "use TROPIUS + Fly instead of long walks".
    {
        "name": "fly_to",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "Outdoors: fly to a town you have visited, in one call (a party Pokemon must know "
            "FLY, and you need the Feather Badge). go_to already flies when that is shorter."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "town": {"type_": "STRING", "description": "A visited town, e.g. 'LILYCOVE CITY'."},
                "reasoning": {"type_": "STRING", "description": "One short sentence: why fly there."},
            },
            "required": ["town", "reasoning"],
        },
    },
    {
        "name": "talk_to",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "Talk to the person standing at (x, y): walks to a tile next to them, turns to "
            "face them, presses A and reads what they say. If it starts a battle, use use_move."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "x": {"type_": "INTEGER", "description": "The x of the tile the person stands on."},
                "y": {"type_": "INTEGER", "description": "The y of the tile the person stands on."},
                "reasoning": {"type_": "STRING", "description": "One short sentence: who and why."},
            },
            "required": ["x", "y", "reasoning"],
        },
    },
    {
        "name": "heal_at_pokecenter",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "Heals the whole party from anywhere: walks to the nearest Pokemon Center "
            "(running from wild battles on the way), talks to the nurse, says YES, and "
            "checks every party Pokemon is fully healed."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "reasoning": {"type_": "STRING", "description": "One short sentence: why heal now."},
            },
            "required": ["reasoning"],
        },
    },
    {
        "name": "run_from_battle",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "In a WILD battle, choose RUN. The result says whether you got away or the "
            "battle continues. Refused in trainer battles."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "reasoning": {"type_": "STRING", "description": "One short sentence: why run."},
            },
            "required": ["reasoning"],
        },
    },
    {
        "name": "throw_ball",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "In a WILD battle, throw a Poke Ball from the bag. The result says whether the "
            "Pokemon was caught (it is not nicknamed) or broke free. Weaken it first: low "
            "HP makes catching likelier."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "ball": {
                    "type_": "STRING",
                    "description": "Which ball, as listed in your bag (default 'Poke Ball'), e.g. 'Great Ball'.",
                },
                "reasoning": {"type_": "STRING", "description": "One short sentence: why catch it."},
            },
            "required": ["reasoning"],
        },
    },
    {
        "name": "switch_pokemon",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "In a battle, send out another party Pokemon (uses your turn). Refused for the "
            "Pokemon already in battle or a fainted one."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "pokemon": {
                    "type_": "STRING",
                    "description": "The party Pokemon's name (e.g. 'SHROOMISH') or its party number 1-6.",
                },
                "reasoning": {"type_": "STRING", "description": "One short sentence: why switch."},
            },
            "required": ["pokemon", "reasoning"],
        },
    },
    {
        "name": "buy_item",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "Inside a Poke Mart: walks to the clerk and buys `count` of an item, then "
            "closes the shop. The result gives money and bag count before and after, or "
            "lists what the shop sells if it doesn't sell that item."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "item": {"type_": "STRING", "description": "The item's name, e.g. 'POTION' or 'POKE BALL'."},
                "count": {"type_": "INTEGER", "description": "How many to buy, 1-99."},
                "reasoning": {"type_": "STRING", "description": "One short sentence: why buy it."},
            },
            "required": ["item", "count", "reasoning"],
        },
    },
    # Seen live: told to teach Rock Smash, the model pressed B over and over in the HM
    # man's house; the menus (START, BAG, TMs & HMs, party, forget-a-move) are one call.
    {
        "name": "teach_move",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "Outside battle: teach a TM or HM from your bag to a party Pokemon in one call "
            "(it opens the bag, uses the TM/HM, picks the Pokemon and, if it already knows "
            "four moves, forgets `forget`, or the weakest non-HM move if you leave it out). "
            "Refuses a Pokemon that can't learn it."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "item": {"type_": "STRING", "description": "The TM/HM or its move, e.g. 'HM06' or 'ROCK SMASH'."},
                "pokemon": {"type_": "STRING", "description": "Party Pokemon name or slot 1-6, e.g. 'BRELOOM'."},
                "forget": {"type_": "STRING", "description": "Optional: the move to forget if it knows four."},
                "reasoning": {"type_": "STRING", "description": "One short sentence: why teach it."},
            },
            "required": ["item", "pokemon", "reasoning"],
        },
    },
    # David (2026-11-04): BRELOOM, always first in the party, took every battle's EXP.
    {
        "name": "set_lead",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "Outside battle: make a party Pokemon the lead (slot 1), in one call. The lead "
            "starts every battle, and only Pokemon that fight get EXP: lead with the one "
            "that needs levels."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "pokemon": {"type_": "STRING", "description": "Party Pokemon name or slot 1-6, e.g. 'NUMEL'."},
                "reasoning": {"type_": "STRING", "description": "One short sentence: why it should lead."},
            },
            "required": ["pokemon", "reasoning"],
        },
    },
    # David (2026-11-04): the weakest Pokémon holds the Exp. Share once we have it (the
    # harness keeps that up itself; this is for any other held item).
    {
        "name": "give_item",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "Outside battle: have a party Pokemon hold an item, in one call. The item comes "
            "from the bag or from the teammate holding it; what the Pokemon held goes back "
            "to the bag, so nothing is lost."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "item": {"type_": "STRING", "description": "The item's name, e.g. 'EXP SHARE'."},
                "pokemon": {"type_": "STRING", "description": "Party Pokemon name or slot 1-6, e.g. 'NUMEL'."},
                "reasoning": {"type_": "STRING", "description": "One short sentence: why it should hold it."},
            },
            "required": ["item", "pokemon", "reasoning"],
        },
    },
    # David (2026-11-04), "Potions + Jagged Pass": heal from the bag where a Pokémon
    # Center is a long walk away. Seen live in MAGMA HIDEOUT: use_item("ESCAPE ROPE")
    # needed a Pokémon, so field items that take none are allowed without one.
    {
        "name": "use_item",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "Use an item from the bag, in one call. Medicine (Potion, Super Potion, "
            "Antidote, Revive...) goes on a party Pokemon: leave item empty to use the "
            "cheapest one that heals it. ETHER / MAX ETHER restore one move's PP (`move`; "
            "left out, the move missing the most PP), ELIXIR / MAX ELIXIR every move's; "
            "PP UP / PP MAX raise `move`'s max PP. In a battle only medicine and PP items, as "
            "the turn of the Pokemon choosing now (the battle section's ITEMS line suggests "
            "one), e.g. REVIVE on a fainted one. Outside battle, ESCAPE ROPE (caves and "
            "dungeons only) and REPEL / SUPER REPEL / MAX REPEL take no Pokemon."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "item": {"type_": "STRING", "description": "The item's name, e.g. 'SUPER POTION' or 'ESCAPE ROPE'. Optional for medicine."},
                "pokemon": {"type_": "STRING", "description": "For medicine and PP items: party Pokemon name or slot 1-6, e.g. 'AZUMARILL'."},
                "move": {"type_": "STRING", "description": "For ETHER, MAX ETHER, PP UP, PP MAX: the move, e.g. 'THUNDER'."},
                "reasoning": {"type_": "STRING", "description": "One short sentence: why use it."},
            },
            "required": ["reasoning"],
        },
    },
    # David (2026-11-04): swap party members at a Pokémon Center PC (live: deposit
    # CAMERUPT, withdraw the TROPIUS a full party sent to the PC).
    {
        "name": "deposit_pokemon",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "Inside a Pokemon Center: put a party Pokemon in a PC box, in one call (it walks "
            "to the PC and closes it again). Not your last Pokemon that can fight."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "pokemon": {"type_": "STRING", "description": "Party Pokemon name or slot 1-6, e.g. 'CAMERUPT'."},
                "reasoning": {"type_": "STRING", "description": "One short sentence: why store it."},
            },
            "required": ["pokemon", "reasoning"],
        },
    },
    {
        "name": "withdraw_pokemon",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "Inside a Pokemon Center: take a Pokemon out of the PC boxes into your party, in "
            "one call (it finds it in any box). Needs a free party slot: deposit_pokemon one first."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "pokemon": {"type_": "STRING", "description": "Species or nickname of the boxed Pokemon, e.g. 'TROPIUS'."},
                "reasoning": {"type_": "STRING", "description": "One short sentence: why take it."},
            },
            "required": ["pokemon", "reasoning"],
        },
    },
    # Seen live: switch-training cost ~3 model decisions per wild battle (walk into the
    # grass, switch_pokemon, use_move); train runs the whole loop server-side.
    {
        "name": "train",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "Level up one party Pokemon on wild battles, in one call: walks onto the nearest "
            "tiles where this map's wild Pokemon appear, steps until one does, switches the "
            "trainee out on turn 1 for a teammate that wins (teammates take turns), wins with "
            "its best move, and repeats. The trainee shares the EXP without fighting. A weak "
            "team is healed at a Pokemon Center, not with the bag, unless heal_with_items. "
            "Stops at until_level, when the team is weak, at a question (new move, "
            "evolution), a trainer, or after max_minutes, and reports every battle and the "
            "wild levels here. With catch, it also stops when that wild species appears (at "
            "catch_min_level or higher) so you can catch it. With method='fish', stand facing "
            "water instead: it casts your best rod and battles what bites."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "trainee": {
                    "type_": "STRING",
                    "description": "The party Pokemon to level up, e.g. 'SHROOMISH', or 'LOWEST': always your lowest-level one.",
                },
                "until_level": {
                    "type_": "INTEGER",
                    "description": "Stop once the trainee reaches this level (with 'LOWEST': once all of them have).",
                },
                "partner": {
                    "type_": "STRING",
                    "description": "Who fights for it (default: your highest-level Pokemon above half HP).",
                },
                "max_minutes": {"type_": "NUMBER", "description": "Time limit, default 4, at most 10."},
                "catch": {"type_": "STRING", "description": "A wild species to stop for and catch, e.g. 'SABLEYE'."},
                "catch_min_level": {"type_": "INTEGER", "description": "Only stop for one at this level or higher."},
                "heal_with_items": {
                    "type_": "BOOLEAN",
                    "description": (
                        "Heal a weak team with the bag's Potions when they are enough, instead of at a Pokemon "
                        "Center (default false). Never Revive, Max Revive, Full Restore or Max Potion."
                    ),
                },
                "method": {
                    "type_": "STRING",
                    "description": (
                        "'grass' (default): walk in the tall grass. 'fish': cast the best rod (SUPER > GOOD > "
                        "OLD) while you stand facing water."
                    ),
                },
                "reasoning": {"type_": "STRING", "description": "One short sentence: why train now."},
            },
            "required": ["trainee", "reasoning"],
        },
    },
    # Asked for (2026-11-05): trainer rematches through Match Call as training. Trainer
    # battles pay 1.5x EXP; rematch_loop runs the whole loop server-side.
    {
        "name": "rematch_loop",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "Train on trainer rematches on one map, in one call: goes there, battles every "
            "registered trainer who wants a rematch (REMATCH READY in the state) the way train "
            "does (your lowest-level Pokemon leads and a teammate that wins fights for it), "
            "then walks the rematch step counter up and leaves and re-enters the map so more "
            "become ready, and repeats. Trainer battles pay 1.5x EXP. Heals a weak team at a "
            "Pokemon Center. Stops at a question (new move, evolution), when the trainee "
            "faints, on a loss, or after max_minutes, and reports each fight with the EXP gained."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "map": {"type_": "STRING", "description": "The map whose trainers to battle again, e.g. 'ROUTE 120'."},
                "max_minutes": {"type_": "NUMBER", "description": "Time limit, default 6, at most 15."},
                "trainee": {
                    "type_": "STRING",
                    "description": "The party Pokemon to level up (default 'LOWEST': always your lowest-level one).",
                },
                "heal_with_items": {
                    "type_": "BOOLEAN",
                    "description": (
                        "Heal a weak team with the bag's Potions when they are enough, instead of at a Pokemon "
                        "Center (default false). Never Revive, Max Revive, Full Restore or Max Potion."
                    ),
                },
                "reasoning": {"type_": "STRING", "description": "One short sentence: why rematch now."},
            },
            "required": ["map", "reasoning"],
        },
    },
    # -- get_walkthrough: expert scaffolds only --
    {
        "name": "get_walkthrough",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "Get official Emerald walkthrough (Parts 1-21). Part 1: "
            "Littleroot, Part 6: Roxanne, Part 21: Elite Four."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "part": {
                    "type_": "INTEGER",
                    "description": "Walkthrough part 1-21",
                },
            },
            "required": ["part"],
        },
    },
    # -- lookup_pokemon_info: expert scaffolds only --
    {
        "name": "lookup_pokemon_info",
        "scaffolds": GUIDE_SCAFFOLDS,
        "description": (
            "Look up Pokemon information from Bulbapedia (stats, moves, "
            "evolution, locations)."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "topic": {
                    "type_": "STRING",
                    "description": "Pokemon name or topic to look up",
                },
                "source": {
                    "type_": "STRING",
                    "description": "Wiki source (default: bulbapedia)",
                },
            },
            "required": ["topic"],
        },
    },
    # -- replan_objectives: simple + continualharness only (not expert, not simplest) --
    {
        **REPLAN_OBJECTIVES_TOOL_DECLARATION,
        "scaffolds": NO_BUILTINS_SCAFFOLDS,
    },
    # -- evolve_harness: continualharness only --
    {
        "name": "evolve_harness",
        "scaffolds": frozenset({"continualharness"}),
        "description": (
            "Trigger an evolution pass NOW to improve skills, subagents, and "
            "memory based on recent performance. Use this when you notice a "
            "skill or subagent is underperforming and needs improvement, "
            "rather than waiting for the automatic evolution cycle."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "reasoning": {
                    "type_": "STRING",
                    "description": (
                        "What needs improvement and why (e.g., 'navigate_to "
                        "skill gets stuck 80% of the time, needs obstacle "
                        "avoidance')"
                    ),
                },
                "num_steps": {
                    "type_": "INTEGER",
                    "description": (
                        "Number of recent trajectory steps to analyze (default 50)"
                    ),
                },
            },
            "required": ["reasoning"],
        },
    },
]


def _tool_available(entry: Dict[str, Any], scaffold: str) -> bool:
    """Return True if the tool entry is available for *scaffold*."""
    s = entry.get("scaffolds", ALL_SCAFFOLDS)
    if s == ALL_SCAFFOLDS:
        return True
    return scaffold in s


# Speed plan, Phase 3 (David, 2026-11-04): a small model does no worse with a brief
# rationale before its call ("Brief Is Better", arXiv 2604.02155; Chain of Draft,
# 2502.18600), and output costs ~25 tok/s. Opt-in with BRIEF_REASONING=1 so its effect on
# play can be measured on its own: the ANALYZE/PLAN format stays the default.
BRIEF_REASONING_ENV = "BRIEF_REASONING"
BRIEF_REASONING_DESCRIPTION = (
    "One short line, at most 25 words: what you see, then what you do. "
    "Example: 'Birch's dialogue box is open -> press A twice to read it.'"
)
BRIEF_REASONING_SYSTEM_NOTE = (
    "\n\n## Reasoning\nKeep every tool call's reasoning to one short line (at most 25 words): "
    "what you see, then what you do. Think it through, then write only that line."
)
# Parameters that carry the model's rationale.
REASONING_PARAMETER_NAMES = ("reasoning", "reason")


def brief_reasoning_enabled() -> bool:
    return os.environ.get(BRIEF_REASONING_ENV, "").strip() == "1"


def _with_brief_reasoning(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of a tool's parameters with every rationale field asking for one line."""
    props = (parameters or {}).get("properties") or {}
    if not any(name in props for name in REASONING_PARAMETER_NAMES):
        return parameters
    new_props = {
        name: ({**spec, "description": BRIEF_REASONING_DESCRIPTION} if name in REASONING_PARAMETER_NAMES else spec)
        for name, spec in props.items()
    }
    return {**parameters, "properties": new_props}


def build_tools_for_scaffold(scaffold: str) -> List[Dict[str, Any]]:
    """Return the filtered list of Gemini tool declarations for *scaffold*.

    Combines the static ``TOOL_REGISTRY`` with the dynamically-assembled
    local-subagent declarations from
    ``agents.subagents.utils.registry.build_local_subagent_tool_declarations``.
    """
    # 1) Static registry tools
    brief = brief_reasoning_enabled()
    tools: List[Dict[str, Any]] = []
    for entry in TOOL_REGISTRY:
        if not _tool_available(entry, scaffold):
            continue
        tools.append(
            {
                "name": entry["name"],
                "description": entry["description"],
                "parameters": _with_brief_reasoning(entry["parameters"]) if brief else entry["parameters"],
            }
        )

    # 2) Local subagent tools (builtins vs generic primitives)
    #    simplest gets none of these at all.
    if scaffold != "simplest":
        include_builtins = scaffold in EXPERT_SCAFFOLDS
        for spec in LOCAL_SUBAGENT_SPECS:
            if not include_builtins and spec.tool_name in BUILTIN_SUBAGENT_TOOL_NAMES:
                continue
            tools.append(
                {
                    "name": spec.tool_name,
                    "description": spec.description,
                    "parameters": _with_brief_reasoning(spec.parameters) if brief else spec.parameters,
                }
            )

    return tools
