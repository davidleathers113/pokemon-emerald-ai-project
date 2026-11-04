# Local changes to continual-harness

Upstream is sethkarten/continual-harness at `bbab97ad` (2026-05-13, still upstream HEAD
on 2026-11-02). Nothing is committed. Local-only files are listed in `.git/info/exclude`.
Each entry gives what changed, why, and how it was verified.

## Emulator and image
**`Dockerfile.local`** (local-only)
- linux/arm64 image (`mgba==0.10.2` has no macOS-arm64 wheel). Base image and uv pinned by digest.
- libmgba **0.10.5 built from source, pinned to commit 26b7884**. It includes upstream's fix
  for a dangling `currentBank` pointer after the first RTC save write (ce4024a, in 0.10.3+)
  and savestate GPIO/RTC restore fixes (0.10.4/0.10.5). 0.10.2→0.10.5 changes no public
  struct or savestate layout.
- The build **fails** if a needed library symbol is missing or if the bindings load any
  commit other than the pinned one. Before, it only printed a warning.
- Runs `local_patches/fix_mgba_callbacks.py`.
- *Verified:* `ldd` and the bindings' `gitCommit` report 0.10.5 / 26b7884 in the running container.

**`local_patches/fix_mgba_callbacks.py`** (local-only): **the in-game SAVE crash fix**
- mGBA's Python bindings leave three `mCoreCallbacks` function pointers uninitialized. The
  core calls one of them (`savedataUpdated`) whenever a save finishes.
- See `incidents/2026-11-02-save-crash.md`.
- *Verified:*
  - Under ASan: unpatched bindings crash 5/5; patched pass 10/10 with 0 ASan errors.
  - Production image: 120/120 save cycles pass.
  - 30-minute ASan random-play soak passes.

## Saves and checkpoints (`pokemon_env/emulator.py`)
- `save_state` now embeds the in-game flash save in the checkpoint, as mGBA's own
  savestate extdata (`mCoreSaveStateNamed`, `SAVESTATE_SAVEDATA`). The file is written
  atomically. Before, the in-game save was lost on every restart.
- `load_state` restores the embedded save before the state.
  - It rejects truncated or damaged files with `SaveStateError`, rolling back the flash.
  - Old checkpoints load with a warning.
  - Before, a failed load was reported as success.
- *Verified:* `test_phase23_save_persistence.py` (25 checks). A save survives checkpoint →
  new emulator → reset → CONTINUE, and a control run without the flash proves the check
  is meaningful.

## Server (`server/app.py`)
- One re-entrant lock now guards **every** emulator access: game loop, milestone thread,
  save/load/checkpoint endpoints, state/frame/map reads. Before, the endpoints raced the
  game loop on the uvicorn thread.
- Save and load run in the threadpool, not on the event loop.
- `/health` returns 503 when the emulator is missing, starting, or hasn't produced a frame
  for 10 s.
- A failed `--load-state` at startup now **exits** instead of silently starting a new game,
  which would then overwrite the real checkpoint.
- Local endpoints `GET /play`, `GET/POST /agent_pause` (pause flag `.pokeagent_cache/AGENT_PAUSED`).
- The milestone thread reads the current map through `_read_current_map` (see the map fix below).
- *Verified:* 330 s concurrent soak with 9,993 checkpoints, 1,167 checkpoint loads,
  5,037 save/load pairs and 2,282 actions: 0 failures, 1,007/1,007 kept checkpoints load.

## Process supervision (`run.py`)
- A watchdog thread ends the whole run (container exits non-zero) if the game server dies.
  Before, the container stayed "Up" with a dead server.
- *Verified:* `kill -9` of the server makes the container exit in 0.35 s with status 137;
  Ctrl-C still exits 0.

## Map location (`pokemon_env/memory_reader.py`, `server/app.py`, `utils/mapping/map_stitcher.py`)
- The reader took the current map from 0x020322E4, which is pokeemerald's
  `sWarpDestination` (the last warp's target). After CONTINUE from a save it reads 0,0, so
  the AI was told it was in **Petalburg City** with a wrong movement preview while standing
  in Littleroot Town.
- It now reads `gSaveBlock1Ptr->location`, falling back to the old address only before the
  save block exists.
- *Verified:* `test_map_location.py`. Both test states read warp destination (0,0); the
  reader now reports LITTLEROOT TOWN and ROUTE 106, matching the screen.

## Exits summary (`utils/mapping/porymap_state.py`)
- The map section now ends with a plain-language **EXITS** list built from the same porymap
  data, e.g. "north edge of the map (walk off at x=10-11, y=0) -> ROUTE 101" and
  "door/warp at (7, 16) -> LITTLEROOT TOWN PROFESSOR BIRCHS LAB". Edge openings are read
  from the rendered map, so this works for every map.
- Why: upstream removed the explicit warp/connection lists to save tokens, leaving raw JSON
  that the 8B model misread. It took Birch's lab door for "the warp to Route 101" and walked
  in and out of the lab for 20+ moves.
- *Verified:* live on 2026-11-03, the AI left the loop and walked straight up x=10 toward
  the Route 101 exit.

## The harness does the walking and the reading (2026-11-03 night)
Background: a survey of Claude, Gemini and GPT Plays Pokémon and the PokéAgent paper showed
that every working harness has a pathfinder do the walking, detects when the player didn't
move, and doesn't spend model calls on dialogue. Our 8B model pressed one button per
decision (53/53 calls) and walked into trees for 9 decisions despite "UP: BLOCKED" in its
prompt. That came to 1.1 tiles walked per minute.
- **No-movement detection** (`server/field_control.py`, `server/app.py`)
  - Every direction press is checked against the player's tile and map. A blocked press ends
    the rest of the sequence, and the model is told why, e.g. "BLOCKED moving UP at (13,2) on
    ROUTE 101: (13,1) is not walkable … use go_to_exit("north")".
  - Turning in place, ledges and doors are handled.
  - An identical blocked repeat is refused.
- **`go_to_exit(exit)` tool**
  - The model names an exit ("north", "OLDALE TOWN", "PROFESSOR BIRCHS LAB") and the harness
    pathfinds there and steps through.
  - `navigate_to` now refuses unwalkable or off-map targets with a reason, instead of silently
    doing something else.
- **Dialogue auto-advance** (`POKEAGENT_AUTO_DIALOGUE`, default on)
  - Before each model call the harness presses A through plain text boxes and hands the text
    to the model.
  - It never answers YES/NO prompts or menus.
- **Compact history**: each step's reasoning is stored once instead of three times.
  History dropped from 4,818 to 2,135 tokens for the same 16 steps.
- **Bugs found along the way and fixed in the new code:**
  - Facing direction and dialogue state were read from wrong addresses.
  - A press started during a ledge jump was lost.
  - `navigate_to` reported partial or redirected paths as success.
  - `/queue_status` stuck after a trailing WAIT.
- **Game speed** (`server/app.py`)
  - Frames were paced by a fixed sleep on top of each frame's work, so 80 fps ran at about 54.
  - Now the deadline is paced: `GAME_FPS` (default 60, GBA native) while the AI thinks, and
    `TURBO_FPS` (default 480) while the harness is pressing buttons.
  - The viewer frame is encoded at most 40 times a second.
- *Verified:*
  - Tests, all failing on the old code: `test_movement_feedback.py` 6/6,
    `test_movement_exits.py` 9/9, `test_dialogue_auto_advance.py` 6/6,
    `test_dialogue_agent.py` 7/7.
  - All earlier suites still pass.
  - A 40-press walk took 9.44 s before and 1.65 s now.
  - Live: from the Route 101 spot where it had been stuck, the AI reached Oldale Town in
    3.7 minutes at 20 s per decision. The old baseline was 40 s per decision, stuck.

## Press results say what's on screen (2026-11-03, overnight)
- **Symptom (caught by the watchdog):** after winning the first rival battle on Route 103,
  the AI spent 5 minutes under May's last text box.
  - It pressed B and called navigate_to.
  - Every result read "Pressed 1 button; now at (9,3)", so it believed it was still in a battle.
  - Dialogue auto-advance didn't fire for that box until later. The likely cause is a leftover
    window after the battle making it look like a menu; it couldn't be reproduced afterwards.
- **Fix (`server/app.py`):**
  - Every press, walk or exit result now ends with what is still open, e.g.
    `A text box is still open "…Would you like to rest your POKéMON?": press A to continue.`,
    or a choice/menu, or an event in progress.
  - When auto-advance declines while something is on screen, it logs its reasons
    (message box, windows, lock, overworld/battle flags) so the next case can be diagnosed.
- *Verified:* on the Pokémon Center fixture each A advanced one page on screen, and every
  result reported the open box with its text. All movement/dialogue suites pass.

## Map-edge exits must line up with the next map (2026-11-03, overnight)
- **Symptom (caught by the watchdog):** in Oldale Town, at (15,19), the AI called
  `go_to_exit("south")` about 20 times. Each call pressed DOWN into Route 101's trees.
  - EXITS said "south edge (walk off at x=4-15, y=19) -> ROUTE 101". Oldale's bottom row
    really is open ground all the way across.
  - But Route 101 continues only under x=8-11. So the exit list and go_to_exit's planner
    (which picks the nearest open edge tile) both sent it to a dead corner.
- **Fix:**
  - New helper `connection_open_span()` in `utils/mapping/porymap_state.py`. It reads the
    neighbouring map's collision grid from the decomp and keeps only the edge positions where
    the tile across the edge is walkable. It uses pokeemerald's connection offset
    (neighbour = this − offset).
  - Both the EXITS line and `plan_exit_route` (`server/game_tools.py`) use it.
- *Verified:*
  - Oldale now lists "south … x=8-11" and "north … x=8-11".
  - All 134 walkable connections in the game agree with their reverse connection, which
    confirms the offset sign.
  - 12 connections now list no opening. They are sea crossings (Surf) or mountain/cliff
    borders, which can't be walked through anyway.
  - New test `test_go_to_exit_south_from_a_corner_of_oldale_uses_the_real_opening`: goes
    from (15,19) and arrives on ROUTE 101. All exit, movement and dialogue suites pass.

## A warp that cuts a walk short is reported as a warp (2026-11-03, overnight)
- **Symptom (seen right after the exit fix):** the AI went in and out of the Oldale Mart
  for 5+ minutes.
  - Inside, `navigate_to(4,8)` stepped on the exit mat. The warp cut the walk short.
  - The result was sent before the warp finished. It said "STOPPED at (3,7) on OLDALE TOWN
    MART: an event is running … press WAIT", although the player was already outside.
  - The model waited, then (outside, with a battling objective about the Mart) walked back in.
- **Fix (`server/app.py`, `_settle_after_sequence`):** also wait for the map change when the
  sequence was *interrupted*, not only when it finished. If the map changed, the result is
  "Went through a door/warp: now at (14,7) on OLDALE TOWN".
- *Verified:* new test `test_navigate_to_across_an_exit_mat_reports_the_new_map`.
  - It fails on the old code: the call returns while the player is still in the Mart.
  - It passes with the fix. Exit (11/11), movement, dialogue and dialogue-agent suites pass.

## navigate_to names the go_to_exit call for a neighbouring map (2026-11-03, overnight)
- **Symptom:** after the two fixes above, the AI's reason said "Destination: Route 101 (south
  edge)". It then walked one tile per `navigate_to` for minutes, wobbling against Oldale's
  walls (17,13)↔(17,12), and never called `go_to_exit`.
- **Fix:** `exit_hint_for_text()` (`server/game_tools.py`). When a `navigate_to` reason names a
  map that connects to this one by an edge, the result ends with e.g.
  `ROUTE 101 is off this map: go_to_exit("south") walks to the opening and through it in one call.`
  - It's added to both successful and failed walks, not when the walk crossed a door.
  - Names are matched without spaces or case. "ROUTE 11" does not match "ROUTE 110".
  - A town or city also matches without its suffix: the model wrote "toward the exit to
    Littleroot" on Route 101, and the first version missed it.
  - The model often copies the objective's whole route ("Oldale → Route 101 → Littleroot"),
    naming both neighbours. Every named neighbour is then listed with its call; no guessing.
  - Seen working live: the next Oldale decision after the hint first appeared was
    `go_to_exit("south")`, which arrived on Route 101.
- *Verified:* new test `test_navigate_to_toward_a_neighbouring_map_names_the_exit_call`; exit
  suite 12/12.

## Auto-advance overrides the older screen note (2026-11-03, overnight)
- **Symptom:** back in Birch's lab (objective early_015 done), the AI pressed A at May 11
  times in a row.
  - The RESULTS block showed its own press first: `A text box is still open "MAY: …": press A
    to continue`. Below it, auto-advance said `… The text box closed.`
  - The model went by the first line. Each A started May's line again.
- **Fix (`agents/PokeAgent.py`, `_auto_advance_dialogue`):** when auto-advance pressed anything
  after a tool result, its entry ends with "This is the screen now: any 'text box is still
  open' or 'menu is open' note in the result above is from before these presses." The
  short-term memory keeps the plain summary.
- *Verified:* new test `test_auto_dialogue_marks_the_earlier_screen_note_as_out_of_date`;
  dialogue-agent 8/8, phase-4 agent 14/14.

## Objectives the game has already recorded say so (2026-11-03, overnight)
- **Symptom (watchdog: 25 decisions on one tile):** objective early_016 "Receive Pokédex from
  Birch". The AI faced May and pressed A for minutes, believing Birch was in front of it.
  - `local_tools/story_progress.py` on a fresh checkpoint showed FLAG_SYS_POKEDEX_GET and
    FLAG_ADVENTURE_STARTED already set.
  - Birch had handed over the Pokédex inside the cutscene that dialogue auto-advance read
    through. The model never saw it happen.
- **Fix (`server/app.py`, `_note_conditions_met_in_memory`):**
  - For completion conditions the game stores as an event flag, the objective's
    "Completion Condition" line reads e.g. `pokedex_received: ALREADY TRUE in game memory
    (you have the POKéDEX). Call complete_direct_objective(objective_id="early_016") now.`
  - Flags covered: Pokédex, Running Shoes and the eight badges (pret/pokeemerald flags.h,
    SYSTEM_FLAGS 0x860).
  - The agent still makes the call itself, so nothing completes behind its back.
- *Verified:* new `local_tests/test_objective_memory.py`, 2/2. Its fixture
  `test_objective_pokedex.state` is that checkpoint: Pokédex flag set, no badge, so the
  Pokédex objective is flagged and the badge objective is not.
- *Confirmed live:* early_016 completed one decision after the restart.

## Story battles already won say so (2026-11-03, overnight)
- **Symptom:** the Petalburg Woods Grunt was beaten at 14:36, and the researcher handed over
  the Great Ball, which only happens after the win. Story objective petalburg_026 "Battle
  Team Aqua Grunt to save researcher" stayed open anyway.
  - With nothing left to do, the model called get_walkthrough part 1.
  - It then wandered between the woods and Route 104 "to reach Route 101 and save
    Professor Birch".
- **Fix (`server/app.py`, `_CONDITION_TRAINERS`):** the same memory check as for badges,
  using trainer-defeat flags. The game sets TRAINER_FLAGS_START (0x500) + trainer id when a
  trainer is beaten (pret/pokeemerald opponents.h, flags.h).
  - Covers every story battle against named trainers: the three rival battles (any of the
    six rival/starter variants), the Woods and Rusturf grunts, Wally ×2, Maxie, Shelly,
    Matt, Maxie + Tabitha (both needed), Archie, the Elite Four and Wallace.
  - Battles against groups of trainers ("gym trainers", "museum grunts") aren't covered.
- *Verified:* new fixture `local_tests/test_after_woods_grunt.state` (live checkpoint).
  - In it, the Woods Grunt flag is set, exactly one Route 103 rival variant is set (the
    one for DAVID's Torchic), and Rusturf, Roxanne and Maxie/Tabitha are not.
  - New test `test_story_battle_already_won_says_so`; objective-memory 5/5.

## Training: walk into grass, switch-train (2026-11-03)
- **Grass while training.** David: "make sure it doesn't avoid tall grass now. I want it to
  battle pokemon".
  - The pathfinder's grass penalty is now a per-instance `grass_cost`.
  - `server/app.py` `_set_grass_cost_for` sets it to 1.0 while the story objective is an
    unmet level goal (`level_*` / `lead_level_*`) and 8× otherwise.
  - Test `test_grass_is_walked_while_training`; objective-memory 13/13, movement-exits 16/16.
- **Switch-training.** David: "switch immediately to my stronger pokemon every time so we can
  quickly defeat the other pokemon and it will share exp".
  - Gen 3 splits EXP among every Pokémon sent out against the fainted one.
  - The training objective (`custom_train_shroomish_15`) now says: Shroomish stays first,
    `switch_pokemon("COMBUSKEN")` on turn 1 of every battle, win with Combusken, and never let
    Shroomish fight alone.
  - This was an objective edit only, applied live without a restart.

## Team building before Roxanne (2026-11-03)
- **Symptom:** David took over for the Roxanne battle and whited out: "Torchic … is extremely
  weak against rock moves. This is why we need more pokemon."
  - Torchic is pure Fire; Rock hits it 2×, and all three of Roxanne's Pokémon (Geodude Lv12
    ×2, Nosepass Lv15) use Rock Tomb.
  - The built-in objective list goes straight from the city to the gym with no team
    building. My first plan (train Torchic to 16) addressed offence only.
- **Fix:**
  - `agents/objectives/direct_objectives.py` `insert_objective`, plus the operator endpoint
    `POST /insert_objective` (not an agent tool). It inserts a step at or after the current
    objective without overwriting later ones; the full list is saved, so restarts keep it.
  - `server/app.py` `_party_condition`: new completion checks read from the party in memory:
    `party_has_<species>`, `level_<species>_<n>`, and `lead_level_<n>`.
  - Live plan inserted before Roxanne, each step checked against memory:
    1. `custom_catch_shroomish`: catch Shroomish in Petalburg Woods. The ROM gives it catch
       rate 255; the hint says don't attack it first (Ember would knock it out), and buy balls
       if short.
    2. `custom_train_torchic_16`: train to Combusken Lv16 (Fire/Fighting, no longer weak to
       Rock; learns Double Kick), with which moves to forget.
    3. `custom_train_shroomish_12`: train Shroomish to Lv12.
    4. Roxanne's objective now carries the battle plan: Shroomish's Absorb (4×) on each
       Geodude, Combusken's Double Kick (2×) on Nosepass, and Super Potion below half HP.
  - Update after David trained Torchic to Combusken himself: the catch hint now says not to
    attack Shroomish at all (Peck and Ember are 2× on it). When the wild Pokémon in battle is
    the species a `party_has_` objective wants, `_catch_target_line` puts "CATCH THIS ONE …
    throw_ball now; ignore the 'Best move' line" at the top of the state. Test
    `test_battle_against_the_wanted_pokemon_says_catch_it` (wild Shroomish fixture; silent for
    other species and trainer battles); objective-memory 11/11.
  - Data sources: encounter tables and levels from Serebii (Bulbapedia blocks the fetch tool);
    catch rates and types from the ROM's `gSpeciesInfo`. Route 115's Lv23–26 grass was ruled
    out by the walking graph: 0 of its 20 grass tiles can be reached on foot from Rustboro.
- *Verified:*
  - Objective-memory 10/10, including `test_party_conditions_for_team_building` on
    `test_wild_two_party.state` and `test_insert_objective_goes_before_the_current_one_and_persists`.
  - Live: the AI's story objective is now "Catch a SHROOMISH", with the walking route from
    Route 116 back to Petalburg Woods.
  - Also fixed: the route to "Route 116" went into ROUTE 116 TUNNELERS REST HOUSE; an exact
    map name now wins (movement-exits 16/16).

## Trainers not yet beaten are listed with their tiles (2026-11-03)
- **Symptom:** in Rustboro Gym with "Defeat Gym trainers", the AI crept up one tile per
  decision. It didn't know where anyone stood.
- **Fix (`utils/mapping/porymap_state.py`, `_map_trainers` / `_format_trainers`):** the map
  section gets a line under EXITS, e.g. `TRAINERS not yet beaten on this map: - ROXANNE at
  (5, 2); JOSH at (5, 13); TOMMY at (3, 9); MARC at (1, 6)`, plus how a battle starts.
  - A trainer is any map object whose script label starts a trainer battle
    (`trainerbattle_single TRAINER_JOSH, …` in the map's scripts.inc). Gym leaders count too:
    they are TRAINER_TYPE_NONE but battle from their script.
  - The trainer's number comes from `pokemon_env/porymap/include/constants/opponents.h`, a
    copy of pret/pokeemerald's file added for this. Beaten = TRAINER_FLAGS_START (0x500) + id,
    so the list shrinks as trainers fall.
  - Says "every trainer on this map is already beaten" when none are left, and nothing at
    all when the flags can't be read.
  - Also fixed here: inside a building, the town's name in the model's reason
    ("In Rustboro City Gym") no longer triggers `go_to_exit("RUSTBORO CITY")`. That hint had
    been pulling the AI out of the gym.
- *Verified:* new fixture `local_tests/test_rustboro_gym.state` (live, mid-battle with Josh)
  and test `test_unbeaten_trainers_are_listed_with_their_tiles`.
  - Petalburg Woods after the Grunt lists only JAMES (Lyle beaten).
  - The gym lists all four, including Roxanne.
  - The inside-the-gym hint cases were added to `test_way_to_an_objective_target`.
  - Suites: objective-memory 7/7, movement-exits 16/16, battle-view 11/11.

## The story objective says how to get there (2026-11-03)
- **Symptom:** story objective rustboro_032 "Defeat Gym trainers for experience" had target
  "Rustboro Gym". The AI went in, came out, and walked Rustboro City one tile per decision
  for 20+ minutes "to find trainers". The door `(27, 19) -> RUSTBORO CITY GYM` was in its
  exit list.
- **Fix (`server/game_tools.py` `way_to_place`, `server/app.py` `_note_way_to_story_target`):**
  - When the story objective's target is another map, its navigation hint now starts with
    the way there from the walking graph. Example:
    `RUSTBORO CITY GYM is through an exit of this map: go_to_exit("RUSTBORO CITY GYM").`
    For a target further away, the hint gives the full route and the first call.
  - Targets name maps loosely, so a map matches when every word of the target appears in its
    name in order: "Rustboro Gym" ~ RUSTBORO CITY GYM, "Birch's Lab" ~ … PROFESSOR BIRCHS LAB.
  - Nothing is added when the player is already there, when the objective is already marked
    done, or when no map matches ("Mr. Briney's Cottage").
  - The route finder now takes a name-matching function, so the navigate_to hint and this
    hint share it. A one-hop door destination is worded "is through an exit of this map".
- *Verified:* new test `test_way_to_an_objective_target`; movement-exits 16/16, objective-memory 6/6.

## Items already in the bag say so (2026-11-03)
- **Symptom:** after the Woods Grunt, story objective petalburg_027 "Receive Great Ball from
  Devon Researcher" was next. The Great Ball had already gone into the bag inside
  auto-advanced dialogue.
- **Fix (`server/app.py`, `_CONDITION_ITEMS`):** objectives about being given an item now
  check the bag (`read_full_inventory`, read once per call).
  - Covers the Poké Balls, the Great Ball, Devon Goods, Itemfinder, either bike, the
    Meteorite, Go-Goggles, Devon Scope and Magma Emblem.
  - Also covers every story TM (39, 08, 34, 50, 42, 40, 04, 03) and HM (05, 06, 03, 02, 08, 07).
  - The PokéNav uses its system flag (FLAG_SYS_POKENAV_GET, SYSTEM_FLAGS + 0x2).
- *Verified:* new test `test_item_already_in_the_bag_says_so` on `test_after_woods_grunt.state`.
  The Great Ball and Poké Balls are reported; TM39 and the bike are not. Objective-memory 6/6.

## Routes to maps further away, on foot (2026-11-03, overnight)
- **Symptom (found reviewing the 2-hour heartbeat):** in Littleroot the model wrote "travel
  west through Route 102 to Petalburg City". It then made about 25 one-tile navigate_to
  calls around Littleroot (12:00–12:27).
  - Route 102 is three maps away, north via Route 101 and Oldale.
  - The go_to_exit hint only covered maps directly next door, so it said nothing.
- **Fix (`server/game_tools.py`):** when navigate_to's reason names a map that isn't a
  neighbour, the hint gives the way on foot and the first call. Example:
  `PETALBURG CITY is not next to this map; the way on foot is ROUTE 101 → OLDALE TOWN →
  ROUTE 102 → PETALBURG CITY. go_to_exit("north") takes the first step.`
  - The search runs over a walking graph. Each map is split into the areas you can walk
    between (from the decomp's collision grid), and map edges and doors join the areas.
  - So a route never crosses the sea or the trees between two halves of a route:
    - Oldale → Lilycove (needs Surf) gives no route at all.
    - Petalburg → Rustboro goes through Petalburg Woods.
    - Indoors, the first step is the door.
  - The search starts from the player's own area.
  - Ledges count as two-way, which is wrong only on the rare route that climbs back up one.
  - The first call on a cold cache costs about 0.5 s; after that it is cached.
- *Verified:* new test `test_navigate_to_toward_a_map_further_away_names_the_first_exit`.
  - It covers the Littleroot case, two destinations in different directions from Oldale,
    the lab door, the woods, the no-Surf case and the current map's own name.
  - The old code returned "" for every one of these. Movement-exits 14/14.

## An exit edge that can't be walked says so (2026-11-03, overnight)
- **Symptom:** the exit list showed `south edge of the map -> ROUTE 105` on Route 104 the same
  way as a real exit. That edge is all sea.
- **Fix (`utils/mapping/porymap_state.py`, `_format_exits`):** an edge with no open tile on
  either side now reads `(no way across on foot: water or cliff all along it)`.
- *Verified:* new test `test_exit_list_says_when_an_edge_cannot_be_walked`.

## "You are already there" for objectives about reaching a place (2026-11-03, overnight)
- **Symptom:** the AI stood on Route 104 with story objective petalburg_023, "Travel west to
  Route 104 South". It read "South" as a direction and walked back into Petalburg looking
  for the way to Route 105.
- **Fix (`server/app.py` `_note_conditions_met_in_memory`, `server/game_tools.py`
  `arrived_at`):**
  - Covers navigate objectives whose condition is about being somewhere (`*_reached`,
    `*_entered`, `returned_*`, `in_*`).
  - When the current map is the target, the condition line says `ALREADY TRUE (you are on
    ROUTE 104, which is Route 104 South)` and gives the call.
  - The target matches the end of the map name, so "Birch's Lab" matches LITTLEROOT TOWN
    PROFESSOR BIRCHS LAB.
  - A trailing North/South/East/West holds only when the player's walkable area lies
    clearly (more than 10% of the map) on that side.
    - Route 104's halves are separate areas joined only through Petalburg Woods.
    - The woods' south door at y=38 of 80 still counts as South.
    - One open map (Route 111) claims neither half.
  - Conditions that only name a place (`gym_puzzle_solved`) are left alone.
- *Verified:* new tests `test_navigate_objective_for_the_current_map_says_so` and
  `test_arrived_at_tells_the_halves_of_a_route_apart`; objective-memory 4/4.
  - Full local suite: 52/52.
- *Confirmed live (14:05):* after the restart, the AI walked to Route 104 and completed
  petalburg_023 at once. Its reasoning quoted the new line ("Route 104, which is Route 104
  South").
  - It then walked to the woods door and completed petalburg_024 at 14:08.

## Walking tools refuse during a battle (2026-11-03, overnight)
- **Symptom (watchdog: 25 decisions on one tile):** at first this looked like a battle stall,
  but an offline replay of the checkpoint showed the battle advancing normally (Ember,
  super effective, fainted). Bug Catcher Lyle's Wurmple came one after another.
  - Between two of them, the model decided the battle was over. The prompt said BATTLE MODE.
  - It called navigate_to and go_to_exit. Their direction presses only moved the battle
    menu cursor ("Pressed 14 buttons; now at (9,32)").
  - The result read "did not leave the map", which says nothing about why.
- **Fix (`server/app.py`, `_refuse_during_battle`):** when gMain.inBattle is set, both tools
  return at once without pressing anything. The message says a battle is still going on and
  to finish it with press_buttons. press_buttons is unaffected.
- *Verified:* new fixture `local_tests/test_battle_wurmple.state` (the live checkpoint) and
  test `test_walking_tools_refuse_during_a_battle`.
  - Both tools refuse, the game is still in the battle afterwards, and press_buttons A
    still plays it.
  - Movement-exits 15/15; full local suite 53/53.

## A press that did nothing says what the screen is (2026-11-03, overnight)
- **Symptom (watchdog: 25 decisions on one tile):** a wild battle in Petalburg Woods ended
  around 14:40. The model went on writing "Currently in battle with Poochyena" and pressing
  A on empty grass until it was restarted at 14:57.
  - The prompt said `Game State: overworld`.
  - Every result said "Nothing happened: there is no one and nothing to use".
  - Each reasoning line was copied from the one before it in short-term memory.
- **Fix (`server/app.py`, `_nothing_happened_note`):**
  - The note now adds: "Game memory says you are walking on the map: no battle is going on
    and no text box or menu is open."
  - It counts no-op A presses in a row on the same tile and facing. From the second one it
    adds "A has now done nothing here N times in a row: stop pressing A and walk
    (navigate_to or go_to_exit) toward your objective."
  - The count resets on any move, turn, text box or battle.
- *Verified:* `test_a_press_that_does_nothing_says_so` was extended to check the
  game-memory sentence on the first press and the count on the second. Movement-feedback
  6/6, movement-exits 15/15, dialogue-auto-advance 6/6.

## Objective completion by ID (2026-11-03, overnight)
- **Symptom (caught by the overnight watchdog):** 25+ decisions on one tile in Oldale's
  Pokémon Center.
  - The AI healed successfully, then called `complete_direct_objective(category="battling")`
    instead of "story".
  - So the story objective "Heal at Oldale" never advanced and it healed again.
  - Meanwhile it marked three battling objectives done that it hadn't done.
  - The harness's own prompt text told it `Use complete_direct_objective(category="battling")`
    under the battling list.
- **Fix:**
  - `complete_direct_objective` takes `objective_id` (the ID printed in OBJECTIVES, e.g.
    `early_012`). The server derives the category, and an ID that isn't current is refused
    with the list of current IDs.
  - Prompt text (`PokeAgent.py`, `SIMPLE.md`) and the tool schema were updated.
  - Category-only calls still work for other scaffolds.
- **Second bug found while testing it:**
  - `/mcp/complete_direct_objective` lazily loaded objectives *without* restoring the run's
    saved `objectives.json`. A completion that arrived before the first state request reset a
    resumed run to objective 0 and saved that.
  - Both lazy-load paths now share `_load_objectives_sequence()`, which restores first.
- **Repair of the live run:**
  - The three wrongly completed battling objectives were un-done.
  - Story was set to `early_013`; "Heal at Oldale" really was done.
  - The bad file is kept as `objectives.json.reset-by-bug` in that run folder.
- *Verified:* a test server resumed with saved story index 12. First call
  `objective_id=early_012` → completed, advanced to `early_013`, saved index 13. A wrong ID
  is refused with the current IDs. Agent/backend suites still pass.

## Battles: the prompt names the opponent and the best move; `use_move` (2026-11-03)
- **Symptom:** a Lv11 Torchic took 12+ minutes to beat a Lv5 Poochyena.
  - The battle section said only "Opponent detection disabled (feature not reliable)", so
    the model never knew what it was fighting.
  - Every battle opens with the move cursor on slot 1. The model mashed A, so it used
    SCRATCH instead of the super-effective EMBER in slot 4.
- **Fix (`pokemon_env/memory_reader.py`, `read_battle_view`):** reads the live battle
  structs (pret/pokeemerald symbol map): gBattleMons for both sides (species, level, HP,
  types, status, moves, PP), the action/move cursors, and the player's battle controller
  callback, which says which menu is open ("action", "move", "target") or that text or an
  animation is playing ("busy").
  - Move power/type/accuracy come from the ROM's gBattleMoves, matchups from its
    gTypeEffectiveness. Each move gets an effectiveness against the opponent's types, STAB,
    and a score (power × effectiveness × 1.5 for STAB × accuracy); the best one is named.
  - Opponents outside the local species table (Zubat, Marill...) are named from the ROM.
  - `read_comprehensive_battle_info` fills `player_pokemon`/`opponent_pokemon` from the same
    structs. The old opponent probes (an undecrypted gEnemyParty read, a wrong gBattleMons
    address and layout, a RAM scan) are gone; their result was being thrown away anyway.
- **Fix (`utils/state_formatter.py`, `format_battle_view`):** a short battle section:
  opponent, your Pokémon, the open menu and cursor, one line per move ("4 EMBER FIRE 40
  PP25 2x STAB"), and `Best move now: EMBER (slot 4) — call use_move("EMBER").`
- **Fix (`server/app.py`, `/mcp/use_move`):** takes a move name or slot. From the action
  menu it moves to FIGHT and presses A, waits (up to 90 game frames) for the move menu, moves
  the cursor by the shortest 2x2 path (never through an empty slot), checks the cursor
  landed, then presses A. It refuses outside a battle, while text plays ("press A to
  continue"), for an unknown move (listing the four) and for a move with 0 PP. It does not
  wait for the turn; dialogue auto-advance reads the text.
  - Registered like go_to_exit: `agents/tools/registry.py`, the endpoint map in
    `agents/PokeAgent.py`, the battler subagent's allowed tools, and `SIMPLE.md`, whose
    old "the cursor starts on the first move, so `['A','A']`" advice now says to use
    `use_move`. The walking-tool battle refusal points at it too.
- *Verified:* new `local_tests/test_battle_view.py` (11 tests) on `test_battle_wurmple.state`:
  - Trainer battle, opponent WURMPLE Lv3 BUG 15/15, move menu with the cursor on slot 4,
    EMBER 2x + STAB and best; the agent's state text shows the section.
  - use_move("SCRATCH") from the move menu: SCRATCH PP 27→26, EMBER untouched.
  - B, RIGHT (cursor on BAG), then use_move("EMBER"): EMBER PP 25→24.
  - "FLY" is refused listing the moves and nothing is pressed; a call while the attack
    animation plays is refused.
  - Movement-exits 15/15, movement-feedback 6/6, objective-memory 6/6, dialogue-agent 8/8,
    dialogue-auto-advance 6/6, phase4-agent 14/14.

## Wrong gBattleTypeFlags address: trainer battles read as wild (2026-11-03)
- **Symptom:** Bug Catcher Lyle's battle showed as wild.
- **Cause:** `BATTLE_TYPE_FLAGS` was 0x02022AAE, which reads 0 in every battle; the real
  gBattleTypeFlags is the u32 at 0x02022FEC (reads 0xC in Lyle's battle). The bits were also
  wrong: `read_battle_details` tested 0x01 (DOUBLE) as TRAINER, 0x02 (LINK) as DOUBLE.
- **Fix:** the address and the bits from pret `include/constants/battle.h` (TRAINER 1<<3,
  DOUBLE 1<<0, FIRST_BATTLE 1<<4, ...). `can_escape`/`is_capturable` are now true for a wild
  battle other than the first (Birch's) one.
- **Second bug this exposed:** the game does not clear gBattleTypeFlags after a battle:
  `test_after_woods_grunt.state` and `test_movement_route101.state` (both overworld) read
  0x4. `is_in_battle` treated any non-zero flags as a battle, which was harmless only while
  the address was wrong. It now uses gMain.inBattle alone.
- *Verified:* `test_not_in_battle_on_the_overworld_after_a_battle` (both fixtures).

## Fixes from the code review (2026-11-03)
- `/checkpoint` failures now return HTTP 500/503 instead of 200 with `{"status": "error"}`,
  so the /play page, `play-local.sh stop` and the agent all see a failed checkpoint.
- The agent's startup check waits on `/health` 503 ("starting") instead of declaring the
  server ready after 20 instant retries.
- The server clears a leftover `AGENT_PAUSED` flag at startup, so no start path can leave
  the AI silently idle.
- The run.py watchdog ignores a server stopped by Ctrl+C (SIGINT), so there's no false
  "server died" exit.
- A re-prompted tool call that ran and then errored is no longer recorded as a NO-OP.
- Loading a checkpoint into a cartridge whose save type isn't detected yet no longer fails.
  mGBA writes it to the backing `.sav`, as its `GBASavedataLoad` does.
- `local_tools/*` and `test_map_location.py` resolve relative paths before changing directory.
- Removed `# noqa` comments and regex use (`sed`, `grep '^…'`) from the local files.
- *Verified:* every local test rerun on the rebuilt image. Save cycles 40/40; server soak
  of 7,236 checkpoints, 3,679 save/load pairs and 1,237 actions with 0 failures and
  731/731 reloads.

## Macros: heal, run, catch, switch, buy in one call each (2026-11-03)
- **Symptom:** every model decision takes ~30 s, and the mistakes happen in the decisions
  between presses. Healing at a Pokémon Center took a walk, A, three text boxes, a YES/NO
  and more text: 8+ decisions, each able to press A on NO or walk off mid-dialogue. Catching,
  running and switching each need a menu cursor moved to the right row, which is what the
  model gets wrong (see `use_move`). Battle text also waits for A in every single-player
  battle ("Wild SHROOMISH appeared!", "Got away safely!"), and auto-advance only reads field
  text, so each battle message cost a decision.
- **Fix (`server/menu_state.py`, new):** RAM reads that say which screen and cursor the game
  shows, by the *running task or callback* (gTasks[].func, gMain.callback2,
  gBattlerControllerFuncs), sMenu's cursor, the text printers' "waiting for A" state,
  gBattleOutcome, gBattlescriptCurrInstr (nickname prompt, learn-move prompt), gBagPosition,
  gBagPockets, gPartyMenu, sPartyMenuInternal's action list, sMartInfo / sShopData, the
  quantity task's data[1]/data[5], gItems prices, and object events by sprite (nurse 58,
  clerk 83). The approach follows the method pokebot-gen3 documents for Emerald (same pret
  symbols, detect by task/callback, bounded waits); no code was copied from it (GPL).
- **Fix (`server/app.py`, five endpoints):** each presses only after reading that the screen
  and cursor are what it expects, bounds every wait in game frames (plus a wall-clock
  guard), and returns `{success, status, summary}` with the outcome read from memory.
  - `heal_at_pokecenter()`: finds the nurse by sprite, stands two tiles out from her facing
    direction when a counter tile is in between (true in every Pokémon Center), walks there
    with navigate_to, faces her, A, reads her text to the YES/NO task, checks YES is
    highlighted, A, reads to the end, and checks every party Pokémon has full HP and status 0.
    Outside a Pokémon Center it refuses, naming `go_to_exit("POKEMON CENTER")` when the map
    has that door; on 2F it says to go down.
  - `run_from_battle()`: wild only (trainer: "you can't run from a trainer battle"; also the
    Birch battle). Presses through waiting battle text to FIGHT/BAG/POKEMON/RUN (or B from
    the move menu), RUN, then presses through the text: gBattleOutcome 4 and back on the map
    = escaped; back at the action menu = "Couldn't get away: the battle continues".
  - `throw_ball(ball="Poke Ball")`: wild single battles only; refuses a ball you don't have,
    listing the ones you do. BAG, LEFT/RIGHT to the POKé BALLS pocket (gBagPosition.pocket),
    cursor to the ball's row (gBagPockets order), A, checks gSpecialVar_ItemId and that USE
    is highlighted, A. Then: nickname question answered NO with B (B always means NO there),
    the Pokédex page closed with B, and gBattleOutcome 7 = caught (party count says party or
    PC); back at the action menu = broke free. Reports the ball count before/after.
  - `switch_pokemon(pokemon)`: name or slot; refuses an Egg, the Pokémon already in battle
    (gBattleMons[0].personality) or a fainted one. POKEMON, cursor to the slot whose
    personality matches (the menu lists gPlayerParty), A, cursor to SHIFT (by
    sPartyMenuInternal's actions), A; confirms the battler's personality changed. If the game
    keeps the party menu open it quotes the game's message and backs out with B.
  - `buy_item(item, count)`: finds the clerk like the nurse, talks, waits for Task_ShopMenu,
    reads the stock (sMartInfo) and refuses with the stock and prices if it isn't sold, or
    if money can't cover it ("you can afford 9"). BUY, list cursor to the item (sShopData
    row + scroll), A, checks the quantity box is for that item, sets the count with
    RIGHT (+10) / UP (+1) and checks data[1], A, checks YES, A, presses through "Here you
    go!", then backs out with B and closes "Please come again!". Reports money and bag count
    before/after and anything received free (a Premier Ball with 10+ Poké Balls).
  - Any other battle question ("Use next POKéMON?", learning a move) or the game opening the
    party menu itself ends a battle macro with that question quoted: those are the model's.
- Registered like `use_move`: `agents/tools/registry.py` (five tools), the endpoint map, the
  150 s timeout and post-call wait in `agents/PokeAgent.py` (`MACRO_TOOLS`), the battler
  subagent's allowed tools (run, throw, switch), and `SIMPLE.md` (two lines).
- **Second bug found while testing:** under load (nine test containers at once) the nurse's
  greeting was once called a failure: the auto-advance stopped as the question finished
  printing, a few frames before the YES/NO task started. Talking now waits for the menu's
  task, reading more text if a box is still up (`_talk_until`).
- **Test fix (`test_phase4_backend.py`):** two checks asserted exactly 14 tools; they had been
  failing since `use_move` (15). They now assert that every simple-scaffold tool converts.
- New fixtures (`local_tests/`): `test_wild_shroomish_intro.state` and
  `test_wild_shroomish.state` (wild Shroomish in Petalburg Woods, intro text / action menu),
  `test_wild_two_party.state` (wild Silcoon; Torchic + a Shroomish caught with a RAM-assisted
  catch), `test_rustboro_mart.state` (Rustboro Mart entrance).
- *Verified:* new `local_tests/test_macros.py`, 20/20, twice in parallel under load:
  - Heal: Torchic set to 10/35 and poisoned in RAM, walked three tiles away; after the call
    35/35, status 0, standing at (7,4) facing the nurse and free to walk.
  - Run: escaped (gBattleOutcome 4, back on the map), also starting from the intro text; with
    the speeds set in RAM so escape is impossible: "Couldn't get away", still at the action
    menu. Trainer battle refused with nothing pressed.
  - Throw: Poké Balls 5→4 and either caught (party 2, battle over) or broke free (battle on,
    action menu); with the wild HP set to 1 and asleep, a Great Ball caught it, the nickname
    question was answered NO (nickname stays SHROOMISH), gBattleOutcome 7. "Ultra Ball" and a
    trainer battle refused.
  - Switch: by name and by slot, the battler's personality is Shroomish's afterwards; the
    active Pokémon and a fainted one (HP 0 in RAM) refused.
  - Buy: 3 Potions $2784→$1884, Potions 0→3; 12 Poké Balls (RIGHT, UP, UP) 5→17 plus a
    Premier Ball; Ultra Ball refused with the stock; 20 Potions refused ("you can afford 9");
    Route 101 refused. The shop is closed and the player can walk after each.
  - Battle-view 11/11, movement-exits 16/16, movement-feedback 6/6, objective-memory 7/7,
    dialogue-agent 8/8, dialogue-auto-advance 6/6, phase4-agent 14/14, phase4-backend 14/14.
  - Each address was checked by printing it next to a screenshot on these fixtures.

## Battles play out to the next decision (2026-11-03)
- **Symptom (David: "taking 30 seconds between each click"):** seen live in Rustboro Gym,
  after each `use_move` the model spent a separate ~30 s decision pressing A through
  "TORCHIC used EMBER!", "It's not very effective…", "Foe GEODUDE fainted!", "DAVID got
  $160 for winning!". It also alternated use_move / press A on a finished battle's text
  for minutes. use_move returned as soon as the move was chosen, and the pre-decision
  auto-advance only read field text (WEAKNESSES L9, M20).
- **Fix (`server/app.py`):** `use_move` now plays the turn out with the macros' loop
  (`_play_out_battle`: A only while a text printer waits for A and no YES/NO is up) and
  answers at the next decision:
  - FIGHT/BAG/POKEMON/RUN open → `status: YOUR_TURN`, the turn's messages quoted in order
    (`What happened: "TORCHIC used GROWL! · Foe WURMPLE's ATTACK fell! · …"`) and
    `Now: TORCHIC HP 24/33; foe WURMPLE HP 15/15.`
  - battle over (gMain.inBattle clear, back on the map) → `status: BATTLE_OVER`, won / lost /
    ran / caught from gBattleOutcome, the position and party HP, plus any field text box
    still open (a gym leader's words).
  - a question → `status: STOPPED` with it quoted: "Use next POKéMON?", a trainer's "Will
    DAVID change POKéMON?", deleting a move to learn one, the nickname question (auto-advance
    only), the party menu the game opens after a faint, or an evolution.
  - Called while text still plays, it presses through to the menu first and then attacks
    (it used to refuse with "press A to continue").
  - `run_from_battle`, `throw_ball` and `switch_pokemon` end with the same report.
  - Bounded by `MACRO_BATTLE_FRAMES` (4800 frames), a wall-clock guard and 40 presses.
- **Fix (auto-advance, `/mcp/auto_advance_dialogue`):** in a battle whose screen shows text
  or an animation (no battle menu open) and no YES/NO, it plays the text out the same way
  (90 s cap; the agent's request times out at 120 s) and never answers a question, not even
  the nickname one. "Battle text read (…)" goes to RESULTS FROM PREVIOUS STEP. When the
  battle ends into field text (the trainer's last words), that is read too.
- **Memory reads (`server/menu_state.py`), each checked on a fixture:**
  - Battle messages are read from gDisplayedStringBattle (0x02022E2C, decoded with the
    field-message decoder) only while window 0's text printer (B_WIN_MSG,
    sTextPrinters[0].active) is printing. The same buffer also holds the "What will TORCHIC
    do?" prompt, the move menu's PP/TYPE lines and healthbox names ("TORCHIC♂"); the first
    version quoted those.
  - Battle-script YES/NO: the opcode at gBattlescriptCurrInstr is yesnoboxlearnmove 0x5A,
    yesnoboxstoplearningmove 0x5B or yesnobox 0x67 (indexes found by matching the .sym's
    Cmd_* addresses in the ROM's gBattleScriptingCommandsTable at 0x0831BD10).
    **Found while testing:** "Use next POKéMON?" is this script yesnobox, not the
    PlayerHandleYesNoInput controller the macros checked (yes_no_prompt read False there),
    so the old loop did not recognise it as a question (inferred: it would have waited out
    its frame budget; not run on the old code).
  - Evolution: gMain.callback2 = CB2_EvolutionSceneUpdate (0x0813E3A4); its text is
    gStringVar4.
- `agents/tools/registry.py` (use_move description) and `SIMPLE.md` (use_move line; "Battle
  text read") say so.
- **Test change (`test_battle_view.py`):** `test_use_move_refuses_while_text_plays` asserted
  the old mid-turn return; it is now `test_use_move_leaves_no_text_playing`.
- *Verified:* new `local_tests/test_battle_flow.py`, checked from memory each time:
  - EMBER on Lyle's WURMPLE: one call returns `BATTLE_OVER`, the messages in order from
    "TORCHIC used EMBER!" to "DAVID got $48 for winning!", gBattleOutcome 1, Lyle's trainer
    flag set, player free to walk (no A left).
  - GROWL: returns `YOUR_TURN` with the action menu open in memory, and HP matching
    gBattleMons; no "What will…" in the quote.
  - GROWL pressed by hand, then use_move("EMBER") mid-text: plays to the menu, uses EMBER
    (PP GROWL 39, EMBER 24), battle won.
  - "Use next POKéMON?": test_wild_two_party.state with the battling Torchic set to 1 HP and
    poisoned in RAM (no fixture reaches a battle question naturally), GROWL → `STOPPED`,
    "TORCHIC fainted!" then the question quoted; the yesnobox is still up.
  - Auto-advance: from "Wild SHROOMISH appeared!" one A, texts exactly ["Wild SHROOMISH
    appeared!", "Go! TORCHIC!"], action menu open; at "Use next POKéMON?" 0 presses and the
    question still up; at the action menu 0 presses.
  - Suites, all ten at once in parallel on the final code: battle-flow 14/14 (both fixes),
    macros 20/20, battle-view 11/11, movement-exits 16/16, movement-feedback 6/6,
    objective-memory 7/7, dialogue-agent 8/8, dialogue-auto-advance 6/6, phase4-agent 14/14,
    phase4-backend 14/14.
  - Not verified: a live run with the model; the trainer switch question, learn-a-move and
    evolution stops (no fixture reaches them, WEAKNESSES L10).

## Heal before trainers when the team is weak (2026-11-03)
- **Symptom (David: "It should go to the pokemon center between trainers if the HP or PP
  are low"):** seen live in Rustboro Gym, the model went from trainer to trainer with a
  worn-down Torchic (WEAKNESSES M21, W3).
- **Fix (`pokemon_env/memory_reader.py`):** `assess_team_condition` / `read_team_condition`:
  weak when any party Pokémon has fainted, any other is under 50% of max HP
  (`TEAM_LOW_HP_SHARE`), or every damaging move (power > 0) of the lead is at ≤ 25% of its
  base PP (`TEAM_LOW_PP_SHARE`; base PP and power from gBattleMoves in ROM, 0x0831C898).
  Eggs are ignored.
- **Fix (`server/game_tools.py`):** `way_to_pokemon_center`: the nearest "… POKEMON CENTER
  1F" on foot by the same walking graph as the story route hints (the BFS is now
  `_walk_paths`, shared with `_far_map_hint`); "" when already inside one.
- **Fix (`server/app.py`, `_note_team_condition`):** outside a battle, while weak, the story
  objective's navigation hint starts with e.g. `TEAM IS WEAK (TORCHIC 15/35 HP): heal before
  any trainer — heal_at_pokecenter() inside RUSTBORO CITY POKEMON CENTER 1F; the way:
  go_to_exit("RUSTBORO CITY POKEMON CENTER 1F"), then heal_at_pokecenter().` (inside the
  Center: "call heal_at_pokecenter() now"). On a battle objective the description starts
  with "HEAL FIRST (team is weak), then:". It runs after the route hint so it comes first;
  legacy objectives get the same; with no objective loaded it goes at the top of the state.
  Nothing in a battle.
- **Fix (`utils/mapping/porymap_state.py`):** the TRAINERS line ends with "(heal first: team
  is weak)" while weak.
- **Fix (`utils/state_formatter.py`):** in a wild battle, a lead under 25% HP gets
  "run_from_battle() or switch_pokemon() is safer than attacking".
- `SIMPLE.md`: one line, "If the OBJECTIVES or the state say TEAM IS WEAK, heal …".
- Not done: navigate_to / go_to_exit don't steer around trainers' sight lines while weak.
- *Verified (`test_battle_flow.py`):* the rules on hand-made parties (17/35 HP weak, 18/35
  not; SCRATCH 8/35 + EMBER 6/25 PP weak, SCRATCH 9/35 not; a fainted lead hands "lead" to
  the next); from memory, the Pokémon Center fixture healthy, then weak at 15/35 (RAM);
  outside in Rustboro City the line names `go_to_exit("RUSTBORO CITY POKEMON CENTER 1F")`;
  inside the Center "call heal_at_pokecenter() now"; healthy adds nothing; from the gym
  fixture's tile the way is RUSTBORO CITY → RUSTBORO CITY POKEMON CENTER 1F with
  `go_to_exit("RUSTBORO CITY")`; mid-battle (gym fixture, party HP 1 in RAM) no objective
  line, but TRAINERS says "(heal first: team is weak)".

## Grinding in one call: `train` (2026-11-03)
- *Review fix (before going live):* on a full-suite run, the last step before the deadline
  started an encounter whose intro wasn't yet "in battle", and `train` returned "time's up"
  mid-intro. The time check now runs only after any transition has settled, so a battle
  `train` itself started is finished first. The time-limit test also accepts the designed
  `no_partner` stop, since encounters are random. test_train_macro 15/15 on two runs (3 and
  5 battles a minute).
- **Symptom (David: "Can we speed it up between battles? It sits for a while after the
  battle is over."):** measured live, switch-training a Lv7 SHROOMISH with a Lv17
  COMBUSKEN cost about three model decisions per wild battle (walk into the grass,
  `switch_pokemon("COMBUSKEN")`, `use_move` until it ends), ≈ 90 s of model time per
  battle; the pause after each battle was the model deciding where to walk next
  (WEAKNESSES M27).
- **Fix (`server/app.py`, `/mcp/train`):** `train(trainee, partner=None, until_level=None,
  max_minutes=4)` runs the loop server-side:
  - Outside a battle it walks (navigate_to) to the nearest pair of side-by-side encounter
    tiles and steps between them with the per-press movement check until gMain.inBattle.
    A press that doesn't reach the other tile (someone on it) is retried, then the next
    pair is tried.
  - In a wild battle it plays the opening text out first (gBattleMons still holds the
    *last* battle's Pokémon during "Wild X appeared!"; seen in the tests, the first version
    picked TORCHIC's EMBER for SHROOMISH), switches the trainee out on its first turn
    (`mcp_switch_pokemon`; default partner = highest-level teammate above half HP, as
    `_switch_train_line` picks), then uses the partner's best move (`best_move` of the
    battle view, highest expected damage) with `mcp_use_move` each turn.
  - Stops and returns on: until_level reached; team weak (`read_team_condition` with the
    partner's PP counted: new `lead_index` argument); a question or menu (learn move,
    evolution, "Use next POKéMON?", party menu); a trainer battle (not fought); no healthy
    partner while the trainee is out; max_minutes (a battle in progress is finished, up to
    90 s more); anything unexpected (couldn't reach the grass, another map, a script took
    control, 12 turns in one battle, 300 steps without a battle). Every wait is bounded
    by frames and wall clock.
  - Refuses without pressing anything: no grass and not a cave (with the way to the
    nearest map that has some, e.g. "RUSTBORO CITY → ROUTE 116. go_to_exit("RUSTBORO
    CITY") takes the first step"), a weak team (with the way to heal), no partner, bad
    arguments.
  - Returns `{success, status, summary, battles: [...], stopped_because, steps,
    elapsed_s}`; each battle records opponents, outcome, the trainee's level/EXP/HP before
    and after and the partner's HP. Summary e.g. "5 wild battles in 64 s (SILCOON Lv5,
    WURMPLE Lv5, …): SHROOMISH Lv6→7 (+105 EXP shared, HP 26/26), TORCHIC 33→28/35.
    Stopped: the 1-minute limit is up".
  - **Advantage rule** (David: "You don't need to switch train when the other pokemon is
    weak against shroomish"): the trainee fights itself, without switching, when its best
    move with PP left is ≥2x against the foe, none of the foe's types is ≥2x against the
    trainee's (ROM type chart, `type_effectiveness`), and it is above half HP
    (`_trainee_advantage`). Checked every turn, so it is switched out once under half.
    The in-battle hint then reads `SHROOMISH has the advantage over GEODUDE (ABSORB 4x):
    fight with use_move("ABSORB") — no switch needed.` instead of SWITCH-TRAINING.
- **Fix (`server/game_tools.py`):** encounter tiles come from porymap metatile behaviors
  (pokeemerald TILE_FLAG_HAS_ENCOUNTERS): tall/long/ash grass anywhere, the cave floor
  (`MB_CAVE`) only on `MAP_TYPE_UNDERGROUND` maps. Sand and the Abandoned Ship's indoor
  tiles are left out: beaches and the ship have no land Pokémon. `way_to_wild_encounters`
  reuses the walking graph of `way_to_pokemon_center`.
- **Level-up stall (`server/menu_state.py`, `_play_out_battle`):** seen live at 18:23
  ("switch_pokemon did nothing: The battle did not reach a menu in time; the screen last
  said … "SHROOMISH grew to LV. 10!"", then the auto-advance ran out its limit too) and
  reproduced in testing: the stat panel after "X grew to LV. N!" waits for a key
  without a text printer, so use_move, the battle macros and the auto-advance waited out
  their whole 4800-frame budget (80 s) after **every level-up**. Read on
  test_wild_two_party.state: gBattlescriptCurrInstr 0x082DABC9 holds opcode 0x6C
  (drawlvlupbox) until two A presses (opcode found from memory at the stall, not from
  the .sym). The shared play-out loop (`BattleFlow.level_up_box`) now presses A while
  that opcode is current; the KO turn with a level-up takes 4 s.
- **Turbo while macros wait (`_turbo()`):** the game loop ran at GAME_FPS (60) whenever
  nothing was queued, so battle animations and text between a macro's presses played at
  normal speed. `_play_out_battle`, `_await_game`, `_await_battle_menu` and the whole
  `train` call now hold a counter that the loop counts as busy (TURBO_FPS). Their waits
  were already frame budgets, so turbo doesn't shorten them in game time. Measured on
  test_wild_two_party.state, one wild battle through use_move (EMBER KO plus Torchic's
  level-up): **18.4–19.0 s before, 5.2–5.3 s after** (two runs each).
- **Battle quotes at turbo (`_record_battle_message`):** with turbo, test_battle_flow's
  auto-advance test lost "Go! TORCHIC!": the play-out sampled the message window every
  30 ms and a message that doesn't wait for A printed and was replaced in between. The
  game loop now records each battle message as it starts printing (per frame, only a
  byte read unless the battle message window is printing), and the play-out quotes
  from that log.
- **Objective text:** `_switch_train_line` outside a battle puts `TRAINING: call
  train("SHROOMISH", until_level=12) now …` at the top of the state (nothing while the
  team is weak, or with no teammate); an unmet `level_<species>_<n>` condition ends with
  `Outside a battle, call train("SHROOMISH", until_level=12).` The in-battle SWITCH-TRAINING
  line is unchanged. `use_move` / `switch_pokemon` results now carry `end` (how the
  play-out stopped).
- **Agent:** `train` registered in `agents/tools/registry.py` (all guide scaffolds),
  `MACRO_TOOLS`, and SIMPLE.md (one line). `PokeAgent.tool_timeout_s` gives train
  max_minutes × 60 + 240 s (the server caps max_minutes at 10, same numbers). Not added to
  the battle subagent: after its battle train walks on into more battles, and that
  subagent has no walking tools on purpose.
- *Verified (`local_tests/test_train_macro.py`, new fixture `test_train_woods.state`):*
  - One-minute run in Petalburg Woods: 5 wild battles in 64 s (three runs: 64, 70, 60 s;
    the last on the final code, with a TAILLOW among them), each with SHROOMISH switched
    out on turn 1, still at full HP, +19–25 EXP each; SHROOMISH's EXP in memory rose by
    the sum (105), Lv6→7; TORCHIC (also leveled 11→12) 33→28/35; stopped on the time
    limit, free to walk. ≈ 4.3–5.0 battles/min with zero model decisions, against ~3
    decisions (≈ 90 s of model time) per battle before.
  - Level-up panel: SCRATCH KO on test_wild_two_party.state, "TORCHIC grew to LV. 12!"
    quoted, BATTLE_OVER in 4.1 s (was 80+ s).
  - Advantage rule (foe types written in gBattleMons): Rock/Ground → "SHROOMISH has the
    advantage … (ABSORB 4x)" and train lets SHROOMISH win itself (EXP gained, no switch);
    Flying (Taillow) and Normal (Whismur) → SWITCH-TRAINING / train switches to TORCHIC;
    Rock/Ground but SHROOMISH at 11/23 → switch.
  - Cave (test_rusturf_tunnel.state, with the blocked-floor renderer fix): no pair uses
    the blocked (6,11); 4 wild battles in 45 s (GEODUDE, WHISMUR ×3), +126 EXP shared.
  - test_objective_memory's switch-training test now expects the advantage line for
    TORCHIC vs a Bug-type (it asserted "" before the rule) and checks SWITCH-TRAINING
    with the foe's types set to Normal in RAM.
  - until_level: SHROOMISH's level byte set one below its EXP's level in RAM → one battle,
    Lv5→6, stopped "SHROOMISH reached Lv6".
  - "Use next POKéMON?" (Torchic 1 HP + poison, Silcoon 500 HP in RAM): stops with the
    question quoted and still up.
  - Trainer battle: the battle step stops without a press; train itself refuses there
    first (TORCHIC alone, no partner).
  - Refusals: Pokémon Center (no grass, way to ROUTE 116), weak team (TORCHIC 10/33, way
    to heal), unknown trainee/partner, already at until_level; nothing pressed.
  - Not verified: a live run with the model; learn-a-move and evolution stops through
    train (no fixture is one level away; they reuse use_move's detection); a trainer
    spotting the player mid-step.

## Model calls (`utils/agent_infrastructure/vlm_backends.py`, OpenAI backend only)
- `max_output_tokens` is 1024 for local endpoints (observed p95 is 685; one runaway reply
  reached 5,957 tokens in 6m50s).
- Only transient errors are retried (connection, timeout, 408, 429, 5xx), at most 3 tries
  and ≤30 s of backoff. The SDK's own retries are off, and the request timeout is 240 s.
- The tool schema keeps nested types (objects, arrays of objects, numbers). Before,
  everything was flattened to strings.
- Malformed tool-call JSON is reported to the model instead of silently becoming `{}`.
- Optional `OPENAI_TEMPERATURE` / `OPENAI_TOP_P`. The launcher sets 0.7 / 0.8 from the
  Qwen3-VL model card; otherwise Ollama defaults to 1.0.
- `prompt_cache_key` goes in `extra_body` (openai 1.90 rejects the kwarg).
- *Verified:* `test_phase4_backend.py` passes. 30 live calls to Ollama all returned valid
  tool calls (p50 ≈ 5 s).

## Agent loop (`agents/PokeAgent.py`)
- The call guard never stacks a second request on Ollama's single slot.
- Text-only replies are re-prompted once, then recorded as a visible NO-OP.
- History tool results and replies are capped at 1,500 characters, and the prompt is
  trimmed to a 28K-token budget.
- The pause flag (`AGENT_PAUSED`) is honoured at the top of the loop.
- *Verified:* `test_phase4_agent.py` (14 checks).

## Prompt (`agents/prompts/pokeagent-directives/SIMPLE.md`) and tools (`agents/tools/registry.py`)
- The `simple` scaffold also gets `navigate_to`, `get_walkthrough` and `lookup_pokemon_info`.
  Deliberate: an 8B model needs them. It makes runs not comparable to the paper's `simple`
  baseline, and two upstream prompt-selection tests expect the old tool count.
- SIMPLE.md now describes exactly the tools it has (13 then; 14 with use_move), allows the walkthrough/wiki, limits
  memory writes, and says to press B if SAVE is highlighted (upstream's "never save in game").
- *Verified:* in a live test the model pressed A on SAVE 9/10 with the old wording and pressed
  B 10/10 with the new wording.

## Launcher and controller (local-only)
**`play-local.sh`**
- Works with macOS bash 3.2 and is shellcheck-clean.
- Checks the Ollama settings, preloads the model and keeps it loaded.
- Never sends the game to OpenAI unless `BACKEND` says so.
- Resumes the whole run through `--backup-state`: memory, objectives, history, game and save.
- Enables the objective roadmap.
- `stop` checkpoints first, and the live save is copied out before any container is removed.
- Fails loudly with logs if startup fails.
- *Verified:* live launch, stop and resume cycles on 2026-11-03.

**`server/play.html`**
- Uses the server's pause state.
- Checkpoints every minute while a human plays, plus at each hand-over and when the tab closes.
- Shows an offline banner and visible errors.
- A held arrow keeps walking (the `normal` speed preset); taps no longer just turn the player.

**`hints/emerald_hints.json`**: 7 hints seeded into the AI's memory, deduplicated by title.

## Story scenes the game has recorded say so (2026-11-03)
- **Symptom:** story objective "Exit Gym - Devon Researcher reports theft" stayed open with
  FLAG_DEVON_GOODS_STOLEN (0x8E) already set, and the agent wandered. Same stall as the
  Pokédex, Great Ball and Woods Grunt. Story events pass inside auto-advanced dialogue or
  while David plays, and most story steps had no memory check at all.
- **Fix (`server/app.py`, `pokemon_env/memory_reader.py`):** every story objective from
  index 39 was checked against the existing checks. For each one that wasn't covered, the
  map script that plays the event was read (pret/pokeemerald `data/maps/<Map>/scripts.inc`,
  plus `data/scripts` and `src` to make sure nothing clears the flag or moves the var back).
  Only conditions the script clearly ties to one flag or var were mapped.
  - `read_game_var(var_id)`: reads SaveBlock1 `vars[]` at +0x139C (pret `include/global.h`:
    `/*0x139C*/ u16 vars[VARS_COUNT]`, right after the 300 flag bytes at +0x1270), VARS_START
    0x4000. Special vars (0x8000+) aren't in the save and return None.
  - `_CONDITION_FLAGS`, 26 more: Steven found / letter delivered, sailed to Dewford /
    Slateport (the towns' arrival flags; only the boat gets you there at that point),
    shipyard checked, the museum scene (entered, Archie, goods delivered: all
    FLAG_DELIVERED_DEVON_GOODS, set at the end of that one scene), Meteor Falls, Archie
    leaving Mt. Chimney, the Wynaut egg, Castform, the Route 120 Steven/Kecleon/Devon Scope
    scene, the Fortree Gym Kecleon, the submarine escape, Kyogre waking and Archie's
    regret, Steven leading to the Cave of Origin, Wallace's Sky Pillar order, Archie and
    Maxie leaving, Wallace's thanks (same talk as HM07), the League guards, the Hall of
    Fame (FLAG_SYS_GAME_CLEAR). Battling: the Aqua Hideout Master Ball item flag and
    Steven at Meteor Falls.
  - New `_CONDITION_VARS` {condition: (var, minimum, text)}, 9: Mr. Stone met
    (VAR_DEVON_CORP_3F_STATE 1), Blue Orb stolen (VAR_MT_PYRE_STATE 1), four Space Center
    steps (VAR_MOSSDEEP_SPACE_CENTER_STATE 2 after the three grunts upstairs, 3 after Maxie
    and Tabitha give up), Kyogre vs Groudon seen and Rayquaza woken
    (VAR_SOOTOPOLIS_CITY_STATE 2 / 5), Rayquaza calming Sootopolis (VAR_SKY_PILLAR_STATE 2).
    Only vars that never move backward were used.
  - **Bug found on the way:** `maxie_tabitha_defeated` checked trainer flags 734 and 514,
    which are never set. That fight is a `DoSpecialTrainerBattle` with Steven; it ends in
    `HandleSpecialTrainerBattleEnd` (src/battle_tower.c), which doesn't call
    `SetBattledTrainersFlags`. It now uses VAR_MOSSDEEP_SPACE_CENTER_STATE ≥ 3.
  - `_CONDITION_TRAINERS`: museum grunts (both, fought back to back by the script) and
    "find Archie in the Seafloor Cavern" (his trainer flag).
  - New `_CONDITION_MOVES`: Rock Smash, Surf, Fly, Dive and Waterfall taught = a party
    Pokémon knows the move.
  - "Heal at <town> Pokémon Center" steps (`healed_*`, `*_heal_done`,
    `elite_four_final_heal`, 23 of them) use the `party_healed` check. The game doesn't
    record where you healed, and a full-HP team is the point of each one.
- **Left unmapped (56 story steps from index 39), and why:**
  - *Map-name mismatch, not a memory problem (about 40):* navigate steps whose target
    doesn't end a ROM map name, so the location rule can't fire. For example "Dewford
    Gym" vs DEWFORD TOWN GYM, "Granite Cave" vs GRANITE CAVE 1F, "Devon Corporation",
    "Meteor Falls", "Mt. Pyre", "Weather Institute", "Victory Road", the other `*_gym_entered`
    steps, `*_traversed`, `*_exited`, the underwater and Sky Pillar floor steps. Fix with
    target aliases in `arrived_at`, not flags.
  - *Puzzles* (`gym_puzzle_solved`, Lavaridge/Fortree/Mossdeep/Sootopolis/Seafloor
    puzzles): no flag or var records them.
  - *Groups of optional trainers* (Dewford and Petalburg gym trainers, Mt. Chimney summit,
    Weather Institute, Mt. Pyre summit, Seafloor grunts): the game doesn't say which ones
    are required. "All" would rarely be true and "any" too early. The boss's own check
    follows each one.
  - `briney_cottage_reached`: FLAG_LANDMARK_MR_BRINEY_HOUSE means "ever entered", and the
    house is on Route 104, passed long before. `cable_car_used`: VAR_CABLE_CAR_STATION_STATE
    goes back to 0 after each ride; the ride count is an encrypted game stat.
    `champion_chamber_entered`: VAR_ELITE_4_STATE is reset to 0 in `data/event_scripts.s`.
    `sky_pillar_entered`: the door-open var (4) comes before entering, and the next value
    (5) is the summit.
  - `badge_check_passed`: the walkthrough puts the badge gate before Victory Road, but the
    guards are in the League building after it (that event is `elite_four_entered`).
    `bike_registered`: stored as SaveBlock1's registered item, not a flag or var.
    `gym_blocked_discovered`, `aqua_hideout_blocked`, `battle_frontier_unlocked`,
    `battle_frontier_explored`: nothing in the scripts records them.
  - Battling: catching Rayquaza, Kyogre, Groudon or the Regis sets the same
    FLAG_DEFEATED_* as knocking them out, so the flag can't say "caught". The roaming
    Lati@s catch sets no flag (FLAG_CAUGHT_LATIAS_OR_LATIOS is the Southern Island one).
    Species catches, shopping, team levels and "organize team" aren't game events.
- *Verified:*
  - Var offset on fixtures: VAR_STARTER_MON = 1 (Torchic) on `test_after_woods_grunt.state`.
    VAR_BIRCH_LAB_STATE = 3 on `test_movement_route101.state` (starter, no Pokédex) and 5
    wherever FLAG_SYS_POKEDEX_GET is set. VAR_PETALBURG_WOODS_STATE is 0 in
    `test_battle_wurmple.state` (before the Grunt) and 1 after it. VAR_PETALBURG_GYM_STATE
    is 2 after the Wally tutorial. VAR_RUSTURF_TUNNEL_STATE and VAR_ROUTE116_STATE are 0 on
    every fixture, all before the theft (no fixture has FLAG_DEVON_GOODS_STOLEN), matching
    the story. `local_tools/story_progress.py` uses the same offset.
  - New tests in `test_objective_memory.py`:
    - `test_game_vars_read_from_saveblock1`.
    - `test_story_scenes_recorded_in_flags_and_vars`: before the theft nothing is true.
      With FLAG_DEVON_GOODS_STOLEN set in RAM the theft step says ALREADY TRUE, and so
      does Mr. Stone with VAR_DEVON_CORP_3F_STATE = 1. Maxie + Tabitha stays open at
      Space Center state 2.
    - `test_heal_steps_and_hm_steps_read_the_party`.
    - `test_every_mapped_condition_is_an_objective`: every key of every table is a real
      completion_condition, and no condition is in two tables.
  - Suites run one at a time: objective-memory 18/18, movement-exits 17/17, macros 20/20,
    train-macro 15/15.

## Places reached earlier, and POKéNAV calls (2026-11-03, WEAKNESSES M38, M39)
- `server/game_tools.py`: `first_visit_flag(target)` gives the flag the game sets on the
  first arrival at a place (FLAG_VISITED_* for towns, FLAG_LANDMARK_* for buildings and
  caves), matched by name from flags.h. `_note_conditions_met_in_memory` uses it for
  `_reached`/`_entered` navigate steps, so a step counts as done after the player moves on.
  Checked against all 45 such story steps: 13 have a flag, and every match names the right
  place. `returned_` steps are left to the map name.
- `server/field_control.py`, `server/menu_state.py`: the snapshot also reads whether a text
  printer is active and whether the POKéNAV call task (`ExecuteMatchCall`, 0x081960E0) runs.
  `dialogue_status` returns PLAIN for both, ahead of the "extra window means a choice" rule.
- Fixtures: `test_after_briney.state` (at sea, cottage visited, Dewford not yet) and
  `test_boat_peeko_box.state` (Dad's call, first box).
- Operator: with David in Granite Cave, story steps 043–046 were completed, the Flash step
  was removed at his request, and the Steven steps now target GRANITE CAVE STEVENS ROOM
  instead of B2F.
- Suites run one at a time: objective-memory 19/19, dialogue-auto-advance 7/7,
  dialogue-agent 8/8, battle-flow 14/14, macros 20/20, movement-feedback 6/6,
  movement-exits 21/21, train-macro 15/15.

## Catching at a level (2026-11-03, David: "catch a level 10+ Sableye")
- Objective `level_<species>_<n>` with no such species in the party now means "catch one
  at Lv<n>+". The condition note names the call:
  `train("<lead>", catch="<SPECIES>", catch_min_level=<n>)`.
- `train(catch=, catch_min_level=)`: walks the encounter tiles as usual and stops
  (`catch_target`) at the first decision of a battle against the wanted species at that
  level, before any move. A lower one is fought as usual.
- `_catch_target_line` (via `_catch_wanted`): in battle with the wanted species it gives a
  `_catch_plan`: a sleep or paralysis move first if the active Pokémon has one with PP and
  the foe has no status, then the best ball in the bag (Ultra, then Great, then Poké).
  When the one in front is under the level, it says "NOT THIS ONE … run_from_battle".
- Why the plan: a full-HP Sableye (catch rate 45) breaks out of about 94% of Poké Balls.
  Paralyzed and in a Great Ball it is about 13% a throw. The bag had 5 Poké and 2 Great
  Balls, and Dewford has no Poké Mart.
- Encounter data from pret `src/data/wild_encounters.json`: Granite Cave B2F has Sableye
  at 20%, Lv10–12; B1F has it at 10%, Lv9–11.
- Operator: story step `dewford_049b` "Catch a Sableye at Lv10 or higher on Granite Cave
  B2F" (`level_sableye_10`), after the letter and before returning to Dewford.
- Tests: `test_train_stops_for_the_pokemon_to_catch`,
  `test_catch_at_a_level_names_the_one_and_turns_down_the_rest`. Two assertions were
  updated because they encoded the old behaviour: "not in the party, so no hint" became
  the catch hint, and "Poke Ball" became the Great Ball the fixture's bag holds.
  Objective-memory 20/20, train-macro 16/16.

## One-way ledges in routes; naming the ladder (2026-11-03, WEAKNESSES M40)
- `server/game_tools.py`: `_route_areas` (ledge tiles ← → ↑ ↓ are walls) and
  `_ledge_jumps` (from the tile behind each ledge to the tile beyond it, one way).
  `_walk_exits` and `_walk_paths` use them; in the BFS a jump costs no map step and adds
  nothing to the path. `arrived_at` still uses the two-way `_walk_areas`.
- `_walk_exits`: when warps to one map arrive in different parts of it, the go_to_exit
  argument is "NAME (x, y)". `server/field_control.py` `match_exits` accepts that form.
- Operator: David delivered the letter himself; steps 048/049 were completed. Story step
  `dewford_049c` "Train Sableye to Lv15 in Granite Cave" (`level_sableye_15`) comes before
  the return to Dewford.
- Suites run one at a time: movement-exits 22/22, objective-memory 20/20, macros 20/20,
  train-macro 16/16, movement-feedback 6/6.

## Team plan for the Route 110 rival and Wattson (2026-11-04, WEAKNESSES M41–M43)
- Data from pret: rival (Torchic player) LOMBRE 18 / SLUGMA 18 / MARSHTOMP 20; Wattson
  VOLTORB 20 / ELECTRIKE 20 / MAGNETON 22 / MANECTRIC 24 (all Shock Wave); Route 112 is
  75% NUMEL (Lv14-16, catch rate 255, MAGNITUDE at 19). SHROOMISH becomes BRELOOM at 23
  and learns MACH PUNCH.
- Story steps added: `slateport_064b` train Shroomish to 23 (Route 110); rival battle
  hint with the switch plan; `mauville_071b` Sableye to 22; `mauville_071c` catch Numel
  (Route 112); `mauville_071d` Numel to 19.
- Code: `species_family` (evolutions from ROM); train sends a benched trainee in for
  the EXP; the trainee fights alone unless the foe is a threat.
- Suites: objective-memory 21/21, train-macro 17/17, macros 20/20, battle-flow 14/14.

## Faster play: go_to and the stall watcher (2026-11-04, David: "do 1 and 2")
- Measured first (two hours of the AI's log, duplicates removed): ~35 s per decision, and
  about half of the walk/train calls left the player on the same tile.
- `go_to(place)`: walks to a named place map after map (`game_tools.place_route`: exact
  map name first, then the nearest map holding the target's words). The loop is shared
  with heal_at_pokecenter (`_walk_along`): it runs from wild battles, reads through plain
  scene text (Birch's on Route 110 used to stop the heal walk), and stops for a trainer
  battle or a question. The story route hint now ends with `Or go_to("…") walks the whole
  way in one call.` Registered in the tool list and SIMPLE.md; timeout 600 s.
- `local_tools/stall_watch.py`: polls the newest decisions every 15 s; on SAME_TILE (3
  walk/train calls ending on one tile, a call that ran over 60 s counting as progress),
  REPEAT (same call and arguments 4 times) or DOWN (server silent 3 polls) it pauses the
  agent and prints one line for a watcher.
- Tests: `test_go_to_walks_map_after_map` (Route 101 → Oldale Pokémon Center; Day Care →
  Route 117), `local_tests/test_stall_watch.py` (4). Macros 22/22.

## Route 101 walking tests no longer meet wild Pokémon (2026-11-04, WEAKNESSES L12)
- Cause of the 1-in-3 flake: the test game loop runs on wall-clock time and pokeemerald
  advances its RNG every frame, so which grass step meets a wild Pokémon varied run to run.
- `InProcessServer.load(state, wild_encounters=False)` sets VAR_REPEL_STEP_COUNT (0x4021)
  to 1000 in RAM after the load (pret `IsWildLevelAllowedByRepel`: no wild Pokémon below
  the lead's level; Route 101's are Lv2-3, the fixture's Torchic Lv6). The movement suites
  use it for every Route 101 load; the retry in `test_go_to_exit_by_destination_name` is
  gone. Macro tests about running from grass keep encounters on.
- *Verified:* pacing 60 grass steps at y=4: 5 and 6 wild battles without, 0 and 0 with.
  `test_harness_can_turn_wild_encounters_off` checks the counter and that no battle starts.
  Movement suites 7/7 and 26/26 in two back-to-back rounds.

## "Poke Mart" and "Poke Center" name their maps (2026-11-04, WEAKNESSES R17)
- Seen live: go_to("Mauville City Poke Mart") → "can't be reached on foot from ROUTE 117";
  the map is MAUVILLE CITY MART.
- `game_tools._PLACE_ALIASES`: Poke Mart / Pokemon Mart / Pokemart → MART; Poke Center /
  Pokecenter / Pokemoncenter → POKEMON CENTER. `_place_words` now applies every alias in a
  name (it stopped at the first). `_squash_name` drops accents (NFKD): "PokéMart" kept its
  É before, since `'É'.isalnum()` is True, so "Oldale Pokémon Center" never matched either.
- *Verified:* A/B with the old table and `_squash_name`: all five probe names and the
  ROUTE 117 → MAUVILLE CITY MART route fail before, pass after. Tests in
  `test_loose_target_names_match_their_map` and `test_go_to_walks_map_after_map` (the live
  call, walked from the Day Care).

## "Will DAVID change POKéMON?" is answered, not handed back (2026-11-04, WEAKNESSES R16)
- Seen live on Route 117 vs Breeder Isaac: use_move returned "The battle asks a YES/NO
  question … answer it yourself"; the model opened the START menu. M46's counter-switch
  never ran: the question is the battle script's yesnobox (pret
  `BattleScript_HandleFaintedMon`), read as `script_question`, and the branch checked only
  `yes_no_prompt`. It now takes either, and reads "X is about to use Y" from the
  question's own text as well (they share one message).
- `_battle_report` adds the play-out's notes: "Answered for you: switched to COMBUSKEN to
  face ARON (matchup 4 vs 0.5)". Before, the quote showed "SABLEYE, good! Come back!" and
  nothing said why.
- Fixture `local_tests/test_switch_question.state`: Isaac's battle (6 Pokémon) at the first
  FIGHT menu, made from test_route117_blocked.state (walked to (33,10) with
  wild_encounters=False, so a Repel is running in it).
- *Verified:* `test_will_player_change_pokemon_is_answered_not_handed_back` fails on the
  old condition with the live message and passes now.

## Stall watcher: judge a call by where it left the player; skip old rows (2026-11-04)
- Seen live 06:14: "SAME_TILE … (12, 4) on VERDANTURF TOWN (navigate_to, navigate_to,
  go_to_exit)" paused the agent though the go_to_exit had reached ROUTE 117. A row's
  location is where its decision was made, so a call's end tile is the next row's.
  `stall_reason` now counts a move call only when the next row is on the same tile; the
  newest call waits for the next decision.
- Seen live 01:17: after a restart the new run's trajectory file began with the previous
  run's rows (timestamps kept), and the stall already handled paused the agent again.
  The watcher now judges only rows after its start or the last unpause (`rows_after`).
- Seen live 01:35 (my first version of that cutoff): rows are stamped in the container's
  UTC, the watcher ran on the Mac's local clock five hours behind, so the cutoff let
  every row through and a REPEAT already handled paused the agent again 11 s after the
  unpause. The cutoff is now the newest row's own timestamp at start/unpause
  (`newest_time`), so no clock or time zone is involved. Test
  `test_the_cutoff_is_taken_from_the_rows_own_clock`; on the live trajectory the cutoff
  passes 0 old rows.
- *Verified:* `test_stall_watch.py` 6/6 (two new; the old three-walks test assumed a row
  is where its call ended and now adds the next row). Replaying every run file from
  2026-11-04: the Verdanturf report is gone; the real stalls (navigate_to ×3 at (0,6) on
  ROUTE 117, 06:17) still fire.

## Party Pokémon from outside the Hoenn dex are named (2026-11-04, WEAKNESSES R18)
- Seen live: a caught MARILL Lv13 read as "Species_183" in the party, so the story step
  "Train Marill to Lv18" found no Marill and train("MARILL") had nothing to train.
- `read_party_pokemon` replaces a "Species_N" name with the ROM's gSpeciesNames entry
  (`_battle_species_name`, which the battle view already used): "Marill".
- *Verified:* `test_a_pokemon_from_outside_the_hoenn_dex_is_named` on
  test_marill_in_party.state (saved from the live game): party names, `level_marill_18`
  → "MARILL is Lv13 of 18 needed", no catch wanted, `_train_member` finds slot 4, and the
  prompt says Marill. Without the fix the party reads 'Species_183'. Objective-memory
  22/22.

## Edge exits avoid dead-end pockets; train only picks grass it can reach (2026-11-04, WEAKNESSES R19, R20)
- Seen live: go_to_exit("east") from VERDANTURF TOWN (19,6) took the nearest edge tile
  and landed on ROUTE 117 (0,6), walled in on three sides. `_edge_dead_ends(map, next
  map, direction, offset)` lists the edge positions whose arrival tile next door is in an
  area (`_route_areas`) with no exit (`_walk_exits`) except back; `list_map_exits` adds
  them to each edge exit and `plan_exit_route` tries those tiles last.
- Seen live from that notch: train stopped in 0.2 s on every call. `wild_encounter_spots`
  path-checks up to 40 candidate pairs (`find_path`, NPCs considered, no partial paths)
  and keeps 6 reachable ones; `"unreachable"` marks grass that exists but can't be walked
  to. train then refuses with the way round: `way_to_wild_encounters` counts other parts
  of the player's own map (`_walk_paths(other_parts_of_here=True)`) and prefers that map
  on a tie, so from the notch it says VERDANTURF TOWN → ROUTE 117, go_to_exit("west").
- Fixtures from test_marill_in_party.state: `test_verdanturf_east.state` (VERDANTURF TOWN
  (19,6)) and `test_route117_notch.state` (ROUTE 117 (0,6)).
- *Verified:* `test_an_edge_exit_never_lands_in_a_dead_end_pocket` lands at (0,8); with
  the ordering off it lands at (0,6). `test_train_only_picks_grass_it_can_walk_to`: with
  the path check off the notch gets the six unreachable pairs from the live log.

## No attacks left: name the switch, refuse useless status moves, heal walks past trainers (2026-11-04, WEAKNESSES M48, R21)
- Seen live 06:32-06:35: TEAM IS WEAK (MACH PUNCH 0/30, HEADBUTT 0/15, MEGA DRAIN 0/10
  PP) → heal_at_pokecenter → "Did not heal: something has the player's attention on the
  way" (a trainer spotted the player) → in that battle BRELOOM used LEECH SEED five times
  on a WINGULL already seeded.
- `_walk_along` (go_to, heal_at_pokecenter) waits out a scripted scene up to
  `WALK_SCENE_WAITS` times (`MACRO_TEXT_FRAMES` each) for a battle, plain text or walking
  control; a trainer's spotting then reports "a trainer battle started on the way".
- Battle view: with no damaging move with PP, `switch_hint` (memory_reader
  `_switch_hint`) names the teammate to send in, by `counter_score` (moved from app.py,
  same score as the counter-switch), among those able to fight with damaging PP; the line
  reads `call switch_pokemon("COMBUSKEN")`.
- use_move refuses a status move that can't work (`status_move_failure`): LEECH SEED on
  a seeded foe (gStatuses3, STATUS3_LEECHSEED) or a Grass type; a sleep/poison/paralysis/
  burn move on a foe with a major status; poison on Poison/Steel; burn on Fire; THUNDER
  WAVE on Ground. Nothing is pressed; the refusal names the switch when there is one.
- *Verified:* `test_a_trainer_who_spots_the_player_on_the_way_is_reported_as_a_battle`
  (the old code returned the live message from the same fixture),
  `test_no_damaging_pp_names_the_switch`, `test_use_move_refuses_a_status_move_that_cannot_work`
  (five cases, RAM writes on test_switch_question.state).

## The game's replacement menu, and loops that don't walk (2026-11-04, WEAKNESSES M49)
- Seen live 07:06-07:36: MARILL fainted in a wild battle; the game opened its party
  menu to replace it (cursor on MARILL, action SEND OUT). The model alternated
  press_buttons(A) and switch_pokemon for 30 minutes; switch_pokemon refused ("pick one
  with press_buttons"); the stall watcher saw neither SAME_TILE nor REPEAT.
- `_shift_from_party_menu` picks SHIFT or SEND OUT (menu_state.MENU_SEND_OUT, pret
  party_menu.c). switch_pokemon, with that menu open (`_forced_party_menu`), skips the
  POKEMON step. `_play_out_battle` sends in the replacement itself: `_replacement()`,
  the battle view's `_switch_hint` (best counter_score among teammates able to fight
  with damaging PP), at most twice per play-out; the note reads "sent in X for the
  fainted Pokémon". If nobody qualifies, the report names switch_pokemon instead of
  press_buttons.
- stall_watch: CYCLE (the last 6 calls repeat a 2- or 3-call pattern) and NO_PROGRESS
  (8 decisions), both only while nothing changes: the tile, in battle or not, and the
  prompt's party lines and battle lines (`_fingerprint`), so an alternation in a battle
  that is moving on is not a stall.
- *Verified:* fixture route: test_wild_two_party.state, Torchic fainted (RAM), YES to
  "Use next POKéMON?" → the replacement menu on the fainted TORCHIC.
  `test_switch_pokemon_works_in_the_replacement_menu` (old code: the live refusal),
  `test_the_play_out_sends_in_a_replacement_itself` (old code: ends at "menu").
  stall-watch 9/9; replaying today's run file the new rules fire on the known stalls
  (the Route 110 gate door, the bike shop, Isaac's switch question at 05:56, the
  replacement menu at 06:49), and the 05:56 rows show the fingerprint holds battle HP.

## Moves scored by estimated HP; the trainee fights alone only if it wins the race (2026-11-04, WEAKNESSES M50, M51)
- `memory_reader.estimate_damage(attacker, defender, move_id)`: average HP one hit
  takes, from pret CalculateBaseDamage with the battle structs' stats, stat stages and
  abilities (`_read_battle_mon` now reads them), badge boosts (ShouldGetStatBadgeBoost),
  then STAB, the ROM type chart, ability immunities (Levitate, Volt/Water Absorb, Flash
  Fire, Wonder Guard) and the 92.5% average roll. Fixed damage: NIGHT SHADE / SEISMIC
  TOSS and PSYWAVE = level, SONIC BOOM 20, DRAGON RAGE 40, SUPER FANG half the HP left.
- Battle view: each move's `damage`; `expected` = min(damage, foe HP) × accuracy (a sure
  knockout beats a bigger hit that may miss); ties go to the bigger hit. The moves list
  shows "~N HP".
- `_trainee_advantage`: the damage race (see WEAKNESSES M51), `TRAINEE_SOLO_HITS` = 2.
- *Verified:* against the game on test_switch_question.state (foe HP raised in RAM):
  NIGHT SHADE took exactly 24 (= level) every time; SCRATCH took 37-43 against an
  estimate range of 37.4-44.0 (86 and 77 were critical hits). The first version left
  out badge boosts and SCRATCH's 41-42 topped its range: that is how the Stone Badge
  boost was found. Tests `test_damage_estimates_match_the_game`,
  `test_a_fixed_damage_move_is_best_when_it_does_more`; the Lyle test's `expected == 120`
  (power units) is now `15` (HP, capped at WURMPLE's 15) with a comment.

## Abilities (2026-11-04, WEAKNESSES R22; David: catch a Huge Power Marill)
- pret: Emerald picks a Pokémon's ability with the box mon's abilityNum bit
  (GetMonAbility → GetAbilityBySpecies: gSpeciesInfo abilities[abilityNum], +0x16/+0x17),
  not personality & 1. CreateBoxMon sets the bit only when the species has a second
  ability.
- `parse_pokemon` fills `PokemonData.ability_num`; `read_party_pokemon` names it
  (`species_ability`, `ability_name` from gAbilityNames 0x0831B6DB in pret's symbols).
  The state's party entries carry `ability`; PARTY STATUS lines read
  "Marill (Lv.13, HUGE POWER)". Battle-view lines add the game's gBattleMons ability.
- *Verified:* the live Marill (test_marill_in_party.state) has abilityNum 1 → HUGE POWER;
  the other three read EFFECT SPORE, BLAZE, KEEN EYE. On Isaac's ARON (two abilities)
  the reading from gEnemyParty equals the game's gBattleMons ability (STURDY); the first
  check read the wrong enemy slot (gBattlerPartyIndexes names the one out).

## COUNTER is not an attack (2026-11-04, WEAKNESSES M52)
- Seen live 08:17 (the stall watcher's REPEAT): BRELOOM used COUNTER ×4 on an ILLUMISE
  that only used status moves; the view had made it "Best move now" (~2 HP from its ROM
  power 1), and it also counted as a damaging move with PP, so neither the switch hint
  nor TEAM IS WEAK fired.
- `memory_reader.is_attack(gBattleMoves entry)`: power > 0 and not COUNTER, MIRROR COAT,
  BIDE or SPIT UP (`CONDITIONAL_DAMAGE_EFFECTS`, pret battle_move_effects.h). Used by
  `estimate_damage`, `counter_score` (via `attack_type`), `_switch_hint` and
  `assess_team_condition`. Battle-view moves carry `conditional`; both labels say "only
  returns a hit taken". `status_move_failure` refuses COUNTER / MIRROR COAT when none of
  the foe's moves (gBattleMons) is a physical / special attack.
- *Verified:* `test_counter_is_not_an_attack`: no best move, `switch_pokemon("COMBUSKEN")`
  named, the refusal; with POUND in the foe's moves COUNTER is allowed; a BRELOOM with
  COUNTER 6 PP and HEADBUTT 0 PP is TEAM IS WEAK.

## Learn-move advice; party types from the ROM (2026-11-04, WEAKNESSES M53)
- Seen live: BRELOOM learned COUNTER at Lv28 and the model forgot MACH PUNCH.
- `_learn_move_advice(texts)` reads "X is trying to learn Y." from the battle's texts,
  finds X in the party (species or nickname), and compares Y with its four moves by
  `memory_reader.move_worth` (power × accuracy × 1.5 for its own type; status and
  conditional moves 0; power-1 attacks count as 60). Y worth no more than the weakest →
  "don't learn Y: press B (NO), then A (YES) to stop learning it"; else "learn Y and
  forget <weakest> (move N)". `_battle_end_text("learn_move")` appends it.
- `memory_reader.party_mon_types`: parse_pokemon leaves type1/type2 unset (the party
  struct has no types), so `counter_score` had never given party moves STAB; types now
  come from gSpeciesInfo by species.
- *Verified:* `test_learn_move_advice` on the live party (test_marill_in_party.state):
  COUNTER → don't learn; SKY UPPERCUT → learn, forget LEECH SEED (move 4); BRELOOM's types
  GRASS/FIGHTING; MACH PUNCH worth 60, COUNTER 0.

## Catch plan: weaken a healthy foe first, without risking a knockout (2026-11-04, WEAKNESSES M54)
- Seen live 08:41-08:43: seven GREAT BALLs at a full-HP ELECTRIKE.
- `_catch_weaken_move(reader, view, foe)`: only while the foe is above half HP; FALSE
  SWIPE (EFFECT_FALSE_SWIPE 101) if the Pokémon out has it, else its weakest attack
  whose highest roll (`damage` / 0.925, critical hits aside) is below the foe's HP.
  `_catch_plan` then says "first use_move(X) once to weaken it … Then call throw_ball".
  Kept to one move and one line (David: "Just catch the sableye. It's not that hard.").
- *Verified:* `test_catch_plan_weakens_a_healthy_foe_without_risking_a_knockout`
  (test_wild_shroomish.state: SCRATCH at most ~8 of 23 HP; RAM 10/23 → balls at once;
  6/6 → balls at once). Objective-memory 25/25, train-macro 18/18 (its catch stop now
  reads "…use_move("SCRATCH") once…" against SILCOON), battle-view 17/17.

## Walks go round floor switches and scene triggers (2026-11-04, WEAKNESSES R23)
- Seen live in MAUVILLE CITY GYM: talk_to's walk to Wattson from the right switch
  (0,15) crossed the switches at (4,12) and (3,9) and shut the barriers again.
- `game_tools.active_triggers(env, map_dir)`: the map's coord_events of type "trigger"
  whose var equals var_value now (`read_game_var`; var names from
  `pokemon_env/porymap/include/constants/vars.h`, copied from pret, parsed by
  `_var_ids` including "(TEMP_VARS_START + 0x0)" forms). Unreadable vars count as
  firing; spent ones (Birch's ROUTE 110 scene once its var moved on) stay walkable.
  `load_pathfinding_state` puts them in the porymap state as `triggers`;
  `_extract_map_data` passes them on.
- `Pathfinder.find_path`: A* first with the triggers (other than start and goal)
  blocked; if that finds no full path, again without them. Variance candidates avoid
  them too when the first search succeeded.
- `navigate_to` accepts `avoid: [[x, y], ...]` (operator use; not in the model's tools).
- *Verified:* `test_walks_go_round_floor_switches` on test_mauville_gym_switch1.state:
  new route (0,15)→(2,10)→(4,10)→(5,3) touches no switch; with the trigger list emptied
  the same search crosses (3,9); navigate_to reaches (5,3), also with an avoid tile.
  (Avoiding (4,10) itself leaves no switch-free way, so the walk rightly crosses one.)

## "Better matchup": name a clearly better teammate mid-battle (2026-11-04, WEAKNESSES M55)
- Seen live 10:05-10:18, lost to Wattson: 0.5x moves 14 times against MAGNETON, no
  switch by choice.
- `memory_reader._party_battle_mon` builds a battle-mon dict for a party Pokémon from
  its struct Pokemon stats (+0x54..+0x62); `_best_hit` is an attacker's hardest hit with
  PP; `_better_switch` picks the teammate that hits at least 2x harder than the best
  move and still wins the race after one free hit coming in (fewest hits to win, then
  most HP left). Not offered while the Pokémon out knocks the foe out this turn.
  `view["better_switch"]`; the formatter adds the "Better matchup" line after "Best move".
- `_train_partner`: in a battle, the able teammate (above half HP) with the fewest hits
  to win (`reader.turns_to_win`), level breaking ties; outside battle the highest level,
  as before.
- *Verified:* `test_a_clearly_better_teammate_is_named`: on test_switch_question.state
  the plain WHISMUR gets no advice (SABLEYE knocks it out this turn); made Electric/
  Steel with 100 HP in RAM → "Better matchup: call switch_pokemon("COMBUSKEN") instead
  (EMBER 2x, ~125 HP a hit vs ~24 for NIGHT SHADE …)".

## teach_move: TMs and HMs in one call (2026-11-04, WEAKNESSES M56)
- Seen live: the model stalled on "Teach Rock Smash to a Pokémon", pressing B in the HM
  man's house.
- `mcp_teach_move(item, pokemon, forget=None)`: START (cursor from sStartMenuCursorPos
  onto BAG in sCurrentStartMenuActions) → the bag (CB2_BagMenuRun, Task_BagMenu_
  HandleInput) → TMs & HMs pocket → the item (bag cursor) → USE (gSpecialVar_ItemId
  checked) → A through "Booted up an HM. … Teach X to a POKéMON?" (YES) → the teach
  party screen (cursor moved from gPartyMenu.slotId, the Pokémon checked by
  personality) → A; with four moves, A through "…Should a move be deleted?" (YES) to the
  forget screen (CB2 0x081BFAB4, Task_HandleReplaceMoveInput; sMonSummaryScreen +0x40C6
  firstMoveIndex, +0x40C4 newMove), cursor onto `forget`, checked, A → texts until the
  bag → B out of the menus. Success is read from the party's moves.
- Refusals before any press: not in the TMs & HMs pocket (lists what is), not in the
  party, NOT ABLE (`menu_state.can_learn_tmhm`: ROM gTMHMLearnsets), already known, the
  named move unknown, an HM move as `forget`. Without `forget`: the weakest non-HM move
  by `move_worth`.
- Registered: agents/tools/registry.py, PokeAgent MACRO_TOOLS (endpoint and timeout) and
  its short-memory line, SIMPLE.md.
- *Verified:* the flow was watched in memory first (scratch probe). Test
  `test_teach_move_teaches_an_hm_and_forgets_the_named_move`: HM06 written into the bag,
  gPartyMenu.slotId left on 1 → BRELOOM learns ROCK SMASH in place of LEECH SEED,
  COMBUSKEN untouched, the player can walk; refusals for "already knows", HM05 for the
  two Pokémon the ROM says are NOT ABLE (COMBUSKEN, MARILL), and a TM not in the bag.

## Walks clear rocks and trees the party can (2026-11-04, WEAKNESSES R24)
- Seen live on ROUTE 111: rocks at (19,100) and (18,101) on the only way north; HM06
  taught and the Dynamo Badge, but every walk refused ("no walkable path"). The player
  got through by hand: face the rock, A, YES.
- `game_tools.FIELD_MOVE_OBSTACLES` (CUTTABLE_TREE: CUT + badge 1; BREAKABLE_ROCK: ROCK
  SMASH + badge 3), `field_moves_reader` (installed by app.py: moves known by party
  Pokémon whose badge flag is set; cached 1 s), `clearable_obstacles(map_dir)`.
  `_obstacle_tiles` leaves them out, so `_route_areas`, `_walk_exits`, `place_route` and
  the hints route through them; `load_pathfinding_state` passes them as porymap
  `clearable`; `find_path` unblocks them and `_astar` charges `CLEARABLE_TILE_COST`.
- The walk's per-press check now records `report["blocked_tile"]`. `_clear_obstacle`
  (app.py): if that tile is clearable, A, presses through the text and YES, waits for
  the object to vanish and the player to walk. navigate_to clears and walks on (up to 3
  times, "Used ROCK SMASH at (x, y) on the way."); go_to_exit clears and replans (up to
  5 tries in all), so go_to and heal walks get it too.
- Boulders (STRENGTH) stay walls: pushing needs room behind, which a route can't promise.
- Fixture `test_route111_rocks.state`: test_mauville_gym_live.state with the Dynamo Badge
  flag (RAM), HM06 in the bag (RAM) taught by teach_move (it forgot COUNTER, the weakest),
  go_to Route 111, navigate_to (19,103). The rocks' flags are FLAG_TEMP_*: they come back
  each time the map loads.
- *Verified:* `test_walks_smash_a_rock_the_party_can_clear`: navigate_to (19,99) uses
  ROCK SMASH on (18,101) and arrives; the route areas join (19,103) and (19,99); with the
  badge flag cleared (RAM) nothing is clearable, the areas are split and the walk stops
  short.

## train heals a weak team itself (2026-11-04, WEAKNESSES M57)
- Seen live: three train refusals in a row ("the team is weak … heal first"), the model
  alternating train and heal_at_pokecenter, the watcher pausing it.
- `_train_heal_trip(origin)`: mcp_heal_at_pokecenter, then mcp_go_to(origin). mcp_train
  uses it once per call (`TRAIN_MAX_HEALS`): at the start when the team is weak outside
  a battle (then re-reads party, grass and partner), and after a battle that stops on
  team_weak (then finds the grass again). The summary begins "On the way: healed at X and
  came back to Y." A trip that fails hands back with the reason ("tried to heal first:
  Did not heal: a trainer battle started on the way …"). In a battle, the refusal stands.
- `_note_train_walks_itself`: any objective whose description or hint names train( gets
  "train walks to reachable grass by itself: don't navigate_to the grass first."
- *Verified:* from ROUTE 117 (0,8) (test_verdanturf_east.state, then east) with COMBUSKEN at
  10 HP (RAM): "healed at VERDANTURF TOWN POKEMON CENTER 1F and came back to ROUTE 117",
  COMBUSKEN healed, training went on (2 runs). From PETALBURG WOODS a ROUTE 104 trainer
  stops the trip and the refusal says so. On the way: from test_marill_in_party.state
  (18,14) the trip could cross an unbeaten trainer at (8,12) depending on his facing, so
  that start was not used.

## go_to no longer bounces between two maps (2026-11-04, WEAKNESSES R25)
- Seen live 12:48-12:54 UTC: go_to("Route 113") from ROUTE 111 bounced ROUTE 111 ↔ ROUTE
  112 for fifteen maps, from the south part and then from the north part too.
- `_walk_exits`: where one side of a map has several neighbours (ROUTE 111's west: ROUTE
  113 and ROUTE 112), the go_to_exit argument is the map's name, not "west".
- `turn_back_gates`: triggers on VAR_TEMP_* vars whose script moves the player are walls
  for route areas unless the bag holds what the script checks for (`items_reader`,
  installed by the server). Route 111's desert splits the route in two without
  Go-Goggles.
- `_walk_along`: the same exit off the same map twice in one walk stops it ("the way
  loops: …").
- stall_watch WANDER: 5 minutes among 2-3 maps with at least 2 returns, no objective
  completed and the party unchanged. Replayed over the last runs' rows (about 32 hours) it
  fires 30 times, all back-and-forth loops (Oldale Mart, Verdanturf ↔ ROUTE 117, Mauville
  bike shop, ROUTE 111 ↔ 112 from 12:28 UTC).
- *Verified:* from test_route111_south_to_113.state (live save) go_to now walks Fiery Path
  → ROUTE 112 → ROUTE 111 north and is stopped by trainer WILTON's battle; the old code
  reproduces the live fifteen-map bounce. A forced looping plan stops after one round.

## Meteor Falls: rooms, scenes, bridges and copied exit lines (2026-11-04, WEAKNESSES R26-R29)
- Seen live 13:40-14:01 UTC: the AI walked in and out of METEOR FALLS 1F 1R without
  completing "reach Meteor Falls"; never found the Magma/Aqua scene at (14,18); after it
  was boxed in at (13,18); and in ROUTE 112 CABLE CAR STATION go_to_exit with the EXITS
  line copied whole failed again and again.
- R26 `_is_floor_or_variant` reads the room suffix (1R, 2R); CABLE CAR names the station.
- R27 `story_scenes` / `_note_story_scenes`: scenes waiting on this map (a trigger on a
  saved var that equals its value now) lead the story objective's hint with their tile.
- R28 `BRIDGE_ELEVATION`: elevation-15 tiles join the floor they are entered from to the
  same floor beyond; A* carries the walker's elevation across them.
- R29 `match_exits` takes an EXITS line verbatim; go_to_exit stops retrying once off the
  starting map and looks twice for a late map change.
- *Verified:* on the live save test_meteor_falls_before_scene.state the objective reads
  ALREADY TRUE; the hint's navigate_to(14, 18) plays the scene (VAR_METEOR_FALLS_STATE
  0 → 1, ~90 s of text) and go_to_exit("ROUTE 114") walks back over the bridge; with the
  old pathfinder the walk stopped at (22,18). Each new test fails on the old code.

## Tools and tests (local-only)
- `local_tools/checkpoint_from_sav.py`: turn any `.sav` into a verified checkpoint.
- `local_tools/story_progress.py`: read story flags and variables from a checkpoint.
- `local_tests/`: see SETUP.md.
