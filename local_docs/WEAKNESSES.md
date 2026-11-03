# Weaknesses, pitfalls and errors: running list

One line per problem the setup has hit (or is expected to hit), what caused it, and what was
done. Details and test evidence for each fix are in `CHANGES.md` under the section named in
the last column. Newest entries go at the bottom of each table. Status: **fixed** (with a test
that fails on the old code, unless noted), **mitigated** (made rare or cheap, not gone),
**open**, **watch** (reported by others, not yet seen here).

## Emulator, saves and infrastructure
| # | Problem | Cause | Status | Fix / where |
|---|---|---|---|---|
| I1 | In-game SAVE crashed the emulator | mGBA Python bindings leave `mCoreCallbacks` pointers uninitialized; one is called after every save | fixed | `local_patches/fix_mgba_callbacks.py`; CHANGES "Emulator and image", incidents/2026-11-02-save-crash.md |
| I2 | First RTC save write left a dangling flash pointer | libmgba < 0.10.3 bug | fixed | libmgba 0.10.5 pinned by commit; CHANGES "Emulator and image" |
| I3 | In-game save lost on every restart | checkpoints held no flash data; .sav lived in container /tmp | fixed | flash embedded in checkpoints; CHANGES "Saves and checkpoints" |
| I4 | A failed checkpoint load was reported as success / started a new game over the real one | return value ignored | fixed | CHANGES "Saves and checkpoints", "Server" |
| I5 | Endpoints raced the game loop | no lock around emulator access | fixed | one lock for every emulator access; CHANGES "Server" |
| I6 | Dead server left the container "Up" | run.py didn't watch the server | fixed | CHANGES "Process supervision" |
| I7 | Runaway 6,000-token replies, 7-minute steps, stacked retries | no output cap; every error retried 10× | fixed | CHANGES "Model calls" |
| I8 | Game ran slower than native (80 fps target gave ~54) | fixed sleep on top of frame work | fixed | deadline pacing, turbo while pressing; CHANGES "The harness does the walking" |

## Reading the game (wrong or missing facts in the prompt)
| # | Problem | Cause | Status | Fix / where |
|---|---|---|---|---|
| R1 | Told it was in Petalburg while standing in Littleroot | map read from `sWarpDestination` (0,0 after CONTINUE) | fixed | read `gSaveBlock1Ptr->location`; CHANGES "Map location" |
| R2 | Facing direction and dialogue state wrong | wrong addresses | fixed | CHANGES "The harness does the walking" |
| R3 | Took Birch's lab door for the way to Route 101 | exits only in raw JSON the 8B model misread | fixed | plain EXITS list; CHANGES "Exits summary" |
| R4 | Exit list said an edge was open where the next map has trees | openings read from this map only | fixed | CHANGES "Map-edge exits must line up with the next map" |
| R5 | Route 104 listed "south edge -> ROUTE 105" (all sea) like a real exit | no check that an edge can be walked | fixed | CHANGES "An exit edge that can't be walked says so" |
| R6 | Battle labelled wild during a trainer battle | `gBattleTypeFlags` read from the wrong address (0x02022AAE, real 0x02022FEC) with wrong bit meanings | fixed | CHANGES battle-type section (2026-11-03). Pitfall found on the way: the game never clears these flags after a battle, so `is_in_battle` now uses gMain.inBattle only |
| R7 | Opponent never shown ("Opponent detection disabled") | upstream turned off its unreliable reader | fixed | battle view from `gBattleMons` + ROM move/type tables; CHANGES battle view / use_move section |
| R8 | Battle "phase" read as nonsense ("phase_49") | `gBattleCommunication` read from 0x02024A60 (real 0x02024332) | fixed (no test; the value is shown only in the old fallback view) | corrected constant in memory_reader.py |

## Model behaviour (small local model)
| # | Problem | Cause | Status | Fix / where |
|---|---|---|---|---|
| M1 | One button per decision; 1.1 tiles walked a minute | model doesn't batch; no pathfinding used | fixed | navigate_to / go_to_exit, batching; CHANGES "The harness does the walking" |
| M2 | Walked into trees 9 times despite "BLOCKED" in the prompt | ignores static prompt text | fixed | each blocked press is detected, the rest skipped, the repeat refused; same section |
| M3 | Spent model calls reading dialogue | one call per text box | fixed | dialogue auto-advance; same section |
| M4 | Pressed A for minutes at a "text box still open" that had already closed | stale screen note after auto-advance | fixed | CHANGES "Auto-advance overrides the older screen note" |
| M5 | Called complete_direct_objective without an id; nothing completed | tool accepted a category only | fixed | CHANGES "Objective completion by ID" |
| M6 | In/out door loop at the Oldale Mart | a walk ended on a warp mat and was reported as "stopped short" | fixed | CHANGES "A warp that cuts a walk short is reported as a warp" |
| M7 | One-tile walks toward a neighbouring map | didn't connect "Route 101" in its plan to go_to_exit | fixed | CHANGES "navigate_to names the go_to_exit call for a neighbouring map" |
| M8 | 25 min of one-tile walks toward a map three maps away | no route beyond the next map | fixed | walking graph over areas, edges and doors; CHANGES "Routes to maps further away, on foot" |
| M9 | Pressed A at May for 15 min to "receive" a Pokédex it already had | the hand-over happened inside auto-advanced dialogue | fixed | event flags (items, badges); CHANGES "Objectives the game has already recorded say so" |
| M10 | Read "Route 104 South" as a direction and headed for the sea | objective names a place it was already on | fixed | location check incl. route halves; CHANGES "You are already there" |
| M11 | Called walking tools mid-battle; presses moved the battle cursor | believed the battle had ended | fixed | walking tools refuse in battle; CHANGES "Walking tools refuse during a battle" |
| M12 | Pressed A on empty grass for 17 min "in battle with Poochyena" | copied its own earlier reasoning from short-term memory | fixed | no-op A presses state the screen from memory and count repeats; CHANGES "A press that did nothing says what the screen is" |
| M13 | After a won story battle, drifted to the walkthrough's opening chapter | "Battle the Grunt" objective stayed open | fixed | trainer-defeat flags; CHANGES "Story battles already won say so" |
| M14 | Battles take many turns: mashes A, so the first move (Scratch) is used instead of Ember | cursor starts on slot 1 each battle; no move info in the prompt | fixed (live run pending) | battle view names the best move; `use_move("EMBER")` drives the menus; CHANGES battle view / use_move section |
| M16 | Left the Rustboro Gym and combed the city tile by tile for "the gym trainers" for 20+ min | the objective's target ("Rustboro Gym") wasn't tied to a way there; route hints only fired when the model's own reason named a map | fixed | story objective's hint now leads with the route and first call (`go_to_exit("RUSTBORO CITY GYM")`); CHANGES "The story objective says how to get there". Also: 5-minute live checks (David, 2026-11-03) to catch stalls like this sooner |
| M17 | My own route hint pulled the AI out of Rustboro Gym: "In Rustboro City Gym" in its reason matched "Rustboro City" | name matching didn't know the player was inside that town's building | fixed | town name ignored while inside its buildings; CHANGES "Trainers not yet beaten are listed" |
| M18 | Inched through the gym one tile per decision, not knowing where the trainers stood | trainer positions and beaten state not in the prompt | fixed | TRAINERS line from map scripts + trainer flags; same CHANGES section |
| M15 | "Receive Great Ball" objective open with the Great Ball already in the bag | condition not tied to memory | fixed | bag and PokéNav-flag checks; CHANGES "Items already in the bag say so" |
| M19 | Menu chores (heal, run, catch, switch, buy) cost 5-10 decisions each, each a chance to press A on NO or lose the cursor row; every battle message waits for A | one ~30 s decision per press; battle text isn't field text, so auto-advance skips it | fixed (live run pending) | five macros that check memory before each press; CHANGES "Macros: heal, run, catch, switch, buy" |

## Known limits of the fixes (open)
| # | Limit | Why it's acceptable for now |
|---|---|---|
| L1 | The neighbour-exit hint ignores whether the player's part of the map reaches that edge (Route 103 west → Route 110 needs Surf) | the walk then reports that it can't get there |
| L2 | Ledges count as two-way in the walking graph | only matters for routes that climb back up a ledge |
| L3 | Group-battle objectives ("defeat the gym trainers", "museum grunts") aren't checked against memory | no single flag; the agent judges these itself |
| L4 | A trailing North/South on a map that is one open area claims neither half | stays silent rather than guess |
| L5 | The watchdog's "same tile for 25 decisions" also fires during long, healthy battles | a false alarm costs one look, a missed stall costs hours |
| L6 | `use_move` is tested on a trainer single battle only: not yet on doubles (target menu), a wild battle, a 0-PP move or the move-reorder screen | each case refuses or reports rather than pressing blindly; watch the first double battle (Tate & Liza, Maxie + Tabitha) |
| L7 | Six upstream pytest checks fail; we think they failed before our changes (they expect strings the old formatter never printed, or the old tool count). This isn't confirmed: no clean baseline was run | our local suites are the gate; recheck before any upstream PR |
| L8 | Macro paths not yet seen on a real save: "Use next POKéMON?" after a faint, a learn-move prompt, a catch with a full party (sent to the PC), a trapped switch, a double battle, a full bag at the Mart, the Gold Card nurse, the many-clerk Lilycove store. A failed run is tested only with speeds forced in RAM | each stops with the game's own text quoted, or refuses before pressing; watch the first of each live |
| L9 | `use_move` still leaves the turn's battle messages (each waits for A) to the model | it returns once the move is chosen; the macros' battle-text loop (`_play_out_battle`) could finish the turn the same way: next candidate |

## Reported by others: watch list (researched 2026-11-03)
What other people running LLM agents on Pokémon have published. "Here" says how it maps to
our setup; the source quotes went through a summarising fetch tool, so the wording is close,
not exact. Ranked by how soon they're likely to bite on the road to Rustboro, Dewford and
Slateport.

| # | Reported problem | Source | Here |
|---|---|---|---|
| W1 | The agent's picture of the screen is stale or contradicts itself ("in the overworld" with a dialog box open), on **this harness** with a local Qwen | [continual-harness #53](https://github.com/sethkarten/continual-harness/issues/53) (open, no answer) | **seen and fixed** as M4, M11, M12 |
| W2 | Wrong beliefs about PP and items; long-context memory of a move "being depleted"; a 100-hour loop on "Full Heal restores HP" | [Gemini Plays Pokémon, making-of](https://blog.jcz.dev/the-making-of-gemini-plays-pokemon) | PP and moves now come from RAM in every battle prompt (M14); item effects not covered: watch |
| W3 | Low-HP "panic": reasoning degrades, tools forgotten, panic switching, compounding mistakes | [dbreunig on Gemini](https://www.dbreunig.com/2025/06/17/an-agentic-case-study-playing-pokémon-with-gemini.html), [PokéAgent paper](https://arxiv.org/html/2603.15563), [PokéLLMon](https://arxiv.org/html/2402.01118v2) | watch: heal, flee and switch are now one call each (M19); still no rule for *when* to use them (heal/flee rule from RAM HP) |
| W4 | After a whiteout, deleted its own notes and believed it had already cleared the area | [LessWrong on Claude](https://www.greaterwrong.com/posts/HyD3khBjnBhvsp8Gb/so-how-well-is-claude-playing-pokemon) | watch; our objectives come from flags/bag/map, not notes, which limits this |
| W5 | Skipped post-battle NPC dialogue that held the hint; formed a hypothesis and never tested it | [Gemini Crystal](https://blog.jcz.dev/gemini-3-pro-vs-25-pro-in-pokemon-crystal), [arXiv 2605.09998](https://arxiv.org/html/2605.09998v1) | watch: auto-advance hands the text over, but the model may ignore it (Mr. Briney, Devon Goods) |
| W6 | "The tool is broken" loops when the pathfinder finds no path; missing or wrong tool arguments stalled 1,003 turns | same two sources | partly covered (navigate_to says why); watch |
| W7 | Routes through NPCs; an off-screen NPC breaks the route | [Gemini Crystal](https://blog.jcz.dev/gemini-3-pro-vs-25-pro-in-pokemon-crystal) | watch: navigate_to has consider_npcs |
| W8 | Mixing in other versions' mechanics (looked for TEA from FRLG) | [Gemini making-of](https://blog.jcz.dev/the-making-of-gemini-plays-pokemon) | watch: Ruby/Sapphire/ORAS details |
| W9 | Mazes treated as new every visit; claims stuck while tiles are unexplored. Fix used: persistent fog-of-war map | same | watch: Granite Cave, Meteor Falls |
| W10 | Vision can't tell cuttable trees; confuses NPCs, items and buildings; trusts its eyes over a RAM location tag | [LessWrong draft](https://www.greaterwrong.com/posts/8aPyKyRrMAQatFSnG/untitled-draft-x7cc), PokéAgent paper | partly: map grid and location come from memory; Cut trees: watch |
| W11 | One goal at a time: can't level up while advancing; a gym loss started a two-day grinding arc | LessWrong on Claude, Gemini Crystal | watch: Roxanne is next |
| W12 | Full bag broke the Safari Zone | [LessWrong, Gemini vs Claude](https://www.greaterwrong.com/posts/7mqp8uRnnPdbBzJZE/is-gemini-now-better-than-claude-at-pokemon) | watch |
| W13 | Re-talking to trainers it already beat | LessWrong on Claude | watch |
| W14 | Boulder/spinner puzzles: pushed the same boulder one tile again and again | Gemini posts | watch: Mauville/Lavaridge/Mossdeep gyms, Seafloor Cavern |
| W15 | Naming screens and new UIs (wrong letters; Pokégear treated as a normal menu) | Gemini Crystal | watch: PokéNav, nicknames |
| W16 | Battle bookkeeping lost (move-use counts) | Gemini Crystal | battle view reads it from RAM each turn (M14) |
| W17 | Small models hit a floor: Flash-Lite stalls under 20% of Emerald; no open model has finished a full game | [arXiv 2605.09998](https://arxiv.org/html/2605.09998v1), PokéAgent paper | expectation-setting: every stall we remove matters |
| W18 | The game world keeps moving while the model thinks; slow API calls | PokéAgent paper, [continual-harness #10](https://github.com/sethkarten/continual-harness/issues/10) | known: ~30 s per decision; the game idles at native speed while the model thinks |
| W19 | No sense of elapsed time; stops building tools after ~500 stuck turns | LessWrong draft, arXiv 2605.09998 | covered by the watchdog |
| W20 | Long context makes it repeat its own history | Gemini making-of, LessWrong draft | **seen** (M12); mitigations: compact history, memory-checked objectives |
