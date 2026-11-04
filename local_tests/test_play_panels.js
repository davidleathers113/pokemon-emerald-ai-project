// Smoke test for the controller page's live panels (server/play_panels.js):
// renders the fixtures in local_tests/play_fixtures/ and checks the content
// a person watching the page relies on. Run: node local_tests/test_play_panels.js
"use strict";
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.join(__dirname, "..");
const P = require(path.join(ROOT, "server", "play_panels.js"));
const fixture = (name) =>
  JSON.parse(fs.readFileSync(path.join(__dirname, "play_fixtures", name), "utf8"));
const FEED = fixture("ai_feed.json");
const DASH = fixture("dashboard.json");
const OBJS = fixture("objectives.json");
const NOW = P.parseUtc(FEED.now);

// The section of html from the element carrying `marker` to the next `until`.
function slice(html, marker, until) {
  const start = html.indexOf(marker);
  assert.ok(start !== -1, `missing ${marker}`);
  const end = html.indexOf(until, start + marker.length);
  return html.slice(start, end === -1 ? html.length : end);
}

const tests = {
  test_escape_html() {
    assert.strictEqual(P.escapeHtml(`<a href="x">'&'</a>`),
      "&lt;a href=&quot;x&quot;&gt;&#39;&amp;&#39;&lt;/a&gt;");
    assert.strictEqual(P.escapeHtml(null), "");
    assert.strictEqual(P.escapeHtml(0), "0");
  },

  test_parse_utc_treats_zoneless_times_as_utc() {
    assert.strictEqual(P.parseUtc("2026-11-04T16:10:05"), Date.UTC(2026, 10, 4, 16, 10, 5));
    assert.strictEqual(P.parseUtc("2026-11-04T16:09:20.5"), Date.UTC(2026, 10, 4, 16, 9, 20, 500));
    assert.strictEqual(P.parseUtc("2026-11-04T16:04:10.123456"), Date.UTC(2026, 10, 4, 16, 4, 10, 123));
    assert.strictEqual(P.parseUtc("2026-11-04T16:10:05Z"), Date.UTC(2026, 10, 4, 16, 10, 5));
    assert.strictEqual(P.parseUtc("2026-11-04T18:10:05+02:00"), Date.UTC(2026, 10, 4, 16, 10, 5));
    assert.ok(Number.isNaN(P.parseUtc(null)));
  },

  test_feed_in_flight_row_counts_from_since() {
    const html = P.renderFeed(FEED, NOW);
    assert.ok(html.includes("thinking… 12 s"), html);
    assert.ok(html.includes("#61"));
    assert.ok(html.indexOf("inflight") < html.indexOf('data-step="60"'), "in-flight row is on top");
    assert.ok(P.renderFeed(FEED, NOW + 60000).includes("thinking… 1 m 12 s"));
    assert.ok(!P.renderFeed({ ...FEED, in_flight: null }, NOW).includes("thinking…"));
  },

  test_feed_step_rows() {
    const html = P.renderFeed(FEED, NOW);
    assert.ok(html.includes("go_to_exit(&quot;ROUTE 112&quot;)"), "one arg shows as a positional call");
    assert.ok(html.includes("press_buttons(buttons=[&quot;A&quot;,&quot;A&quot;], speed=&quot;fast&quot;)"));
    assert.ok(html.includes("talk_to_npc()"));
    assert.ok(html.includes("33.2 s"));
    assert.ok(html.includes(`<span class="t">${P.fmtClock(Date.UTC(2026, 10, 4, 16, 9, 20, 500))}</span>`));
    assert.ok(html.includes("Arrived: now at (7,0) on ROUTE 112"));
    assert.ok(html.includes('<div class="step fail" data-step="58">'));
    const failed = slice(html, 'data-step="58"', 'data-step="57"');
    assert.ok(failed.includes('class="result fail"'), failed);
    assert.ok(!slice(html, 'data-step="60"', 'data-step="59"').includes("fail"));
    assert.ok(slice(html, 'data-step="57"', "</div></div>").includes("result pending"));
  },

  test_feed_events_interleave_by_time() {
    const html = P.renderFeed(FEED, NOW);
    const order = ['data-step="60"', 'data-step="59"', "▶ Resumed by human", 'data-step="58"',
      "⏸ Paused by stall_watch: SAME_TILE: 6 steps on (12,9)", 'data-step="57"'];
    const at = order.map((m) => html.indexOf(m));
    assert.ok(at.every((i) => i !== -1), JSON.stringify(at));
    assert.deepStrictEqual([...at].sort((a, b) => a - b), at, "rows newest first, events between steps");
    assert.ok(!html.includes("too old to show"), "events older than the oldest step are left out");
  },

  test_feed_expanded_rows_stay_open() {
    assert.ok(P.renderFeed(FEED, NOW, { 59: true }).includes('class="step open" data-step="59"'));
    assert.ok(P.renderFeed(FEED, NOW).includes('class="step" data-step="59"'));
  },

  test_feed_escapes_model_text() {
    const html = P.renderFeed(FEED, NOW);
    assert.ok(html.includes("HP &lt; 50% &amp; the trainer said &quot;Let&#39;s battle!&quot;"));
    assert.ok(html.includes("&lt;SUPER POTION&gt;"));
    assert.ok(!html.includes("<SUPER POTION>"));
  },

  test_feed_paused_and_empty() {
    assert.ok(P.renderFeed({ ...FEED, paused: true }, NOW).includes("AI is paused"));
    assert.ok(P.renderFeed({ steps: [], events: [], in_flight: null }, NOW).includes("No steps yet."));
    assert.ok(P.renderFeed(null, NOW).includes("Waiting"));
  },

  test_party_summary() {
    const html = P.renderParty(DASH);
    assert.ok(html.includes("<b>ROUTE 112</b> (7, 47)"));
    assert.ok(html.includes("₽18,921"));
    assert.ok(html.includes("Badges 3/8"));
    assert.ok(html.includes("Poke Ball ×22 · Great Ball ×5"));
    assert.ok(html.includes('class="warn-strip">EXP: BRELOOM is Lv38'));
    assert.ok(html.includes("set_lead(&quot;AZUMARILL&quot;)"));
    assert.ok(!P.renderParty({ ...DASH, exp_note: "" }).includes("warn-strip"));
  },

  test_party_cards() {
    const html = P.renderParty(DASH);
    assert.strictEqual(html.split('class="mon-head"').length - 1, 6, "six cards");
    const lead = slice(html, "BRELOOM</span>", '<div class="mon');
    assert.ok(lead.includes('<span class="tag lead">LEAD</span>'), lead);
    assert.ok(!lead.includes('class="nick"'), "nickname equal to species is not repeated");
    assert.strictEqual(html.split(">LEAD<").length - 1, 1, "only the lead is tagged");
    assert.ok(lead.includes('<span class="mv">MACH PUNCH</span><span class="type">FIGHTING</span><span class="pp">25/30</span>'));
    assert.ok(lead.includes('class="move no-pp"'), "a move with 0 PP is marked");
    assert.ok(lead.includes("EFFECT SPORE"));
    assert.ok(lead.includes('hp-fill hp-high" style="width:66%"'), lead);
    const azu = slice(html, "AZUMARILL</span>", '<div class="mon');
    assert.ok(azu.includes("“Bubbles &amp; &lt;Co&gt;”"), azu);
    assert.ok(azu.includes("hp-low"), azu);
    assert.ok(azu.includes('<span class="status">PSN</span>'));
    const fainted = slice(html, "GROVYLE</span>", '<div class="mon');
    assert.ok(fainted.includes('hp-fill hp-zero" style="width:0%"'), fainted);
    assert.ok(fainted.includes("Fainted"));
    assert.ok(html.includes('<div class="mon fainted"><div class="mon-head"><span class="species">GROVYLE'));
    assert.ok(slice(html, "TAILLOW</span>", '<div class="mon').includes("hp-mid"));
    assert.ok(html.includes('<div class="mon egg"><div class="mon-head"><span class="species">Egg</span>'));
    assert.ok(!html.includes("WYNAUT"), "an egg's species stays hidden");
    assert.ok(!slice(html, "NUMEL</span>", '<div class="mon').includes(">OK<"));
  },

  test_group_chapters() {
    const ch = P.groupChapters(OBJS.story.objectives);
    assert.deepStrictEqual(ch.map((c) => [c.key, c.done, c.total, c.hasCurrent]),
      [["early", 3, 3, false], ["chimney", 2, 5, true], ["lavaridge", 0, 3, false]]);
  },

  test_progress_header_badges_and_marks() {
    const html = P.renderProgress(OBJS, DASH.badges, {});
    assert.ok(html.includes("5 of 11 story objectives done"));
    assert.strictEqual(html.split('class="pip earned"').length - 1, 3);
    assert.strictEqual(html.split('class="pip"').length - 1, 5);
    for (const name of ["Stone", "Knuckle", "Dynamo", "Heat", "Balance", "Feather", "Mind", "Rain"]) {
      assert.ok(html.includes(`>${name}</span>`), name);
    }
    const chimney = slice(html, 'data-chapter="chimney"', "</section>");
    assert.ok(chimney.includes('<span class="count">2/5</span>'));
    assert.ok(chimney.includes('<span class="mark">✓</span>'));
    assert.ok(chimney.includes('<span class="mark">▶</span>'));
    assert.ok(chimney.includes('<span class="mark">○</span>'));
    // completed_at is UTC; it is shown in the machine's local time.
    const d = new Date(Date.UTC(2026, 10, 4, 15, 30, 45));
    const two = (n) => String(n).padStart(2, "0");
    const when = `Nov ${d.getDate()} ${two(d.getHours())}:${two(d.getMinutes())}`;
    assert.ok(chimney.includes(`<span class="when">${when}</span>`), chimney);
    const current = slice(chimney, "obj current", "</li>");
    assert.ok(current.includes("scroll-target"));
    assert.ok(current.includes("Defeat Maxie"));
    assert.ok(current.includes("Maxie stands by the machine at the crater&#39;s edge."));
    assert.ok(current.includes("Target: MT CHIMNEY"));
  },

  test_progress_collapses_finished_chapters() {
    const html = P.renderProgress(OBJS, DASH.badges, {});
    assert.ok(html.includes('<section class="chapter finished"><button type="button" class="chap-head" data-chapter="early" aria-expanded="false">'));
    assert.ok(!html.includes("Leave the moving truck"), "a collapsed chapter's rows aren't rendered");
    assert.ok(html.includes('<section class="chapter open"><button type="button" class="chap-head" data-chapter="lavaridge"'));
    const toggled = P.renderProgress(OBJS, DASH.badges, { early: true, chimney: false });
    assert.ok(toggled.includes("Leave the moving truck"));
    assert.ok(toggled.includes('data-chapter="chimney" aria-expanded="false"'));
    assert.ok(!toggled.includes("Defeat Maxie"));
  },

  test_progress_battling_list() {
    const html = P.renderProgress(OBJS, DASH.badges, {});
    const battle = slice(html, "Battling goals", "</ol>");
    assert.ok(battle.includes('<span class="count">1/3</span>'));
    assert.ok(battle.includes("✓</span><span class=\"desc\">Win the Roxanne rematch"));
    assert.ok(battle.includes("▶</span>"));
    assert.ok(battle.includes("Train in the grass on ROUTE 112."));
    assert.ok(battle.includes("○</span><span class=\"desc\">Teach a Fire-type move"));
    assert.ok(!battle.includes("scroll-target"), "auto-scroll only targets the story objective");
  },

  test_progress_escapes_objective_text() {
    const html = P.renderProgress(OBJS, DASH.badges, {});
    assert.ok(html.includes("Beat Team Magma &lt;Grunt&gt; &amp; &quot;Tabitha&quot;"));
    assert.ok(!html.includes("<Grunt>"));
    assert.ok(P.renderProgress(null, null, {}).includes("Waiting"));
  },

  test_play_html_wiring() {
    const html = fs.readFileSync(path.join(ROOT, "server", "play.html"), "utf8");
    for (const id of ['id="feed"', 'id="party"', 'id="progress"']) assert.ok(html.includes(id), id);
    assert.ok(html.includes('<script src="/play_panels.js"></script>'));
    for (const url of ['"/ai_feed"', '"/dashboard"', '"/objectives"']) assert.ok(html.includes(url), url);
    assert.ok(html.includes("pollFrame, 100"), "frame poll untouched");
  },
};

let failures = 0;
const names = Object.keys(tests);
for (const name of names) {
  const started = Date.now();
  try {
    tests[name]();
    console.log(`PASS ${name} (${((Date.now() - started) / 1000).toFixed(1)}s)`);
  } catch (e) {  // report every failure, not just the first
    failures += 1;
    console.log(`FAIL ${name}: ${e.name}: ${e.message}`);
  }
}
console.log(`${names.length - failures}/${names.length} passed`);
process.exit(failures ? 1 : 0);
