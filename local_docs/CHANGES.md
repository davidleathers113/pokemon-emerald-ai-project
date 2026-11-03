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

## Tools and tests (local-only)
- `local_tools/checkpoint_from_sav.py`: turn any `.sav` into a verified checkpoint.
- `local_tools/story_progress.py`: read story flags and variables from a checkpoint.
- `local_tests/`: see SETUP.md.
