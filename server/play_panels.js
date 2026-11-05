// Renderers for the controller page (top bar facts, party rail, objective strip,
// activity feed, objectives panel).
// Pure functions: JSON in, HTML string out, no DOM access, so
// local_tests/test_play_panels.js can exercise them under plain node.
// play.html does the polling, DOM writes and click handling.
const PlayPanels = (() => {
  const BADGE_NAMES = ["Stone", "Knuckle", "Dynamo", "Heat", "Balance", "Feather", "Mind", "Rain"];
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const MAX_EVENTS = 10;
  // The party's status names for "no status problem": nothing is shown for them
  // (seen on David's page: "55/55 HP · NONE" in red read like an error).
  const HEALTHY_STATUSES = new Set(["", "OK", "NONE", "NORMAL"]);

  function escapeHtml(value) {
    if (value === null || value === undefined) return "";
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function isDigit(ch) { return ch >= "0" && ch <= "9"; }

  // The server's clock is UTC but its timestamps carry no zone, and Python's
  // isoformat gives 0-6 fraction digits where Date.parse only promises 3.
  function parseUtc(s) {
    if (typeof s !== "string" || s === "") return NaN;
    let iso = s.trim().replaceAll(" ", "T");
    const t = iso.indexOf("T");
    const tail = t === -1 ? "" : iso.slice(t + 1);
    const zoned = iso.endsWith("Z") || tail.includes("+") || tail.includes("-");
    const dot = tail.indexOf(".");
    if (dot !== -1) {
      let end = dot + 1;
      while (end < tail.length && isDigit(tail[end])) end += 1;
      const frac = (tail.slice(dot + 1, end) + "000").slice(0, 3);
      iso = iso.slice(0, t + 1) + tail.slice(0, dot + 1) + frac + tail.slice(end);
    }
    if (t === -1) iso += "T00:00:00";
    return Date.parse(zoned ? iso : `${iso}Z`);
  }

  function pad2(n) { return String(n).padStart(2, "0"); }

  function fmtClock(ms) {
    if (!Number.isFinite(ms)) return "--:--:--";
    const d = new Date(ms);
    return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
  }

  function fmtStamp(ms) {
    if (!Number.isFinite(ms)) return "";
    const d = new Date(ms);
    return `${MONTHS[d.getMonth()]} ${d.getDate()} ${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
  }

  function fmtElapsed(sinceMs, nowMs) {
    const secs = Number.isFinite(sinceMs) ? Math.max(0, Math.floor((nowMs - sinceMs) / 1000)) : 0;
    return secs < 60 ? `${secs} s` : `${Math.floor(secs / 60)} m ${pad2(secs % 60)} s`;
  }

  function fmtMoney(n) {
    const digits = String(Math.trunc(Math.abs(Number(n) || 0)));
    let out = "";
    for (let i = 0; i < digits.length; i += 1) {
      if (i > 0 && (digits.length - i) % 3 === 0) out += ",";
      out += digits[i];
    }
    return (Number(n) < 0 ? "-" : "") + out;
  }

  // One argument reads like the model's call, go_to_exit("ROUTE 112");
  // several keep their names so the order of keys doesn't matter.
  function fmtCall(tool, args) {
    let inner = "";
    if (args !== null && typeof args === "object" && !Array.isArray(args)) {
      const keys = Object.keys(args);
      inner = keys.length === 1
        ? JSON.stringify(args[keys[0]])
        : keys.map((k) => `${k}=${JSON.stringify(args[k])}`).join(", ");
    } else if (args !== null && args !== undefined) {
      inner = JSON.stringify(args);
    }
    return `${tool || "?"}(${inner})`;
  }

  // ---------- Top bar ----------

  // Where the run stands, as chips in the top bar: location, badges, money, balls.
  // These used to head the party panel, below the fold on a laptop screen.
  function renderFacts(dash) {
    if (!dash) return "";
    const pos = Array.isArray(dash.position) && dash.position.length >= 2
      ? ` <span class="muted mono">${escapeHtml(dash.position[0])}, ${escapeHtml(dash.position[1])}</span>` : "";
    const badges = Array.isArray(dash.badges) ? dash.badges : [];
    const earned = badges.filter(Boolean).length;
    const dots = BADGE_NAMES.map((name, i) =>
      `<span class="badge${badges[i] ? " earned" : ""}" title="${name} Badge${badges[i] ? "" : " (not yet)"}"></span>`).join("");
    const balls = (Array.isArray(dash.balls) ? dash.balls : [])
      .map((b) => `${escapeHtml(b.name)} ×${escapeHtml(b.count)}`).join(" · ");
    return `<span class="fact loc" title="Location"><b>${escapeHtml(dash.location || "Unknown")}</b>${pos}</span>`
      + `<span class="fact" title="Badges"><span class="badges">${dots}</span><b class="mono">Badges ${earned}/8</b></span>`
      + `<span class="fact" title="Money"><b class="mono">₽${fmtMoney(dash.money)}</b></span>`
      + `<span class="fact balls" title="Balls">${balls || "No balls"}</span>`;
  }

  // ---------- Activity feed ----------

  function stepRow(s, expanded) {
    // Step numbers start again at each server restart: the time keeps the key unique.
    const key = `${s.step}@${s.time}`;
    const open = expanded && expanded[key] ? " open" : "";
    const fail = s.success === false ? " fail" : "";
    const dur = typeof s.duration_s === "number" ? `${s.duration_s.toFixed(1)} s` : "";
    const reasoning = s.reasoning
      ? `<div class="reasoning" title="Click to expand">${escapeHtml(s.reasoning)}</div>` : "";
    const result = s.result === null || s.result === undefined
      ? `<div class="result pending">…</div>`
      : `<div class="result${fail}">${escapeHtml(s.result)}</div>`;
    return `<div class="step${fail}${open}" data-step="${escapeHtml(key)}">`
      + `<div class="step-head"><span class="t">${fmtClock(parseUtc(s.time))}</span>`
      + `<span class="n">#${escapeHtml(s.step)}</span>`
      + `<code class="call">${escapeHtml(fmtCall(s.tool, s.args))}</code>`
      + (s.model ? `<span class="model">${escapeHtml(s.model)}</span>` : "")
      + `<span class="dur">${dur}</span></div>${reasoning}${result}</div>`;
  }

  function eventRow(e) {
    const source = escapeHtml(e.source || "unknown");
    const text = e.text ? `: ${escapeHtml(e.text)}` : "";
    let label = `${escapeHtml(e.kind)} by ${source}${text}`;
    if (e.kind === "pause") label = `⏸ Paused by ${source}${text}`;
    if (e.kind === "resume") label = `▶ Resumed by ${source}${text}`;
    // David's party rule, applied by the harness ("Lead set to NUMEL (lowest level, Lv20)").
    if (e.kind === "party") label = `👥 ${escapeHtml(e.text || "")}`;
    // The gemini+ollama backend changing model ("Switched to qwen3-vl:8b-instruct: …").
    if (e.kind === "backend") label = `🔁 ${escapeHtml(e.text || "")}`;
    return `<div class="event ${e.kind === "pause" ? "ev-pause" : "ev-resume"}">`
      + `<span class="t">${fmtClock(parseUtc(e.time))}</span><span>${label}</span></div>`;
  }

  // One bar per recent step, oldest on the left: height is the time it took, red a
  // failure, a pulsing bar the step still running. The run's rhythm at a glance
  // (stalls, a run of failures) before reading any rows.
  function renderStrip(feed) {
    const steps = (feed && Array.isArray(feed.steps) ? feed.steps : []).slice().reverse();
    if (!steps.length) return "";
    const longest = Math.max(1, ...steps.map((s) => Number(s.duration_s) || 0));
    const bars = steps.map((s) => {
      const d = Number(s.duration_s) || 0;
      const running = s.result === null || s.result === undefined;
      const cls = s.success === false ? " fail" : running ? " pending" : "";
      const height = Math.max(10, Math.round((d / longest) * 100));
      // "Step 7", not "#7": the rows below own "#n", and tests find rows by it.
      const label = `Step ${s.step} ${s.tool || "?"} · ${d ? `${d.toFixed(1)} s` : "running"}`;
      return `<span class="bar${cls}" style="height:${height}%" title="${escapeHtml(label)}"></span>`;
    });
    const f = feed.in_flight;
    if (f) bars.push(`<span class="bar pending" style="height:30%" title="Step ${escapeHtml(f.step)} thinking"></span>`);
    const fails = steps.filter((s) => s.success === false).length;
    const durs = steps.map((s) => Number(s.duration_s)).filter((d) => d > 0);
    const avg = durs.length ? `${(durs.reduce((a, b) => a + b, 0) / durs.length).toFixed(1)} s` : "–";
    return `<div class="strip-wrap"><div class="strip" aria-hidden="true">${bars.join("")}</div>`
      + `<div class="strip-legend"><span>Last ${steps.length} steps</span>`
      + `<span class="mono">avg ${avg} · ${fails} failed</span></div></div>`;
  }

  // nowMs is "now" on the server's clock (play.html corrects for skew via feed.now),
  // so the thinking counter doesn't drift when the browser's clock is off.
  function renderFeed(feed, nowMs, expanded) {
    if (!feed) return `<div class="empty">Waiting for the AI…</div>`;
    const steps = Array.isArray(feed.steps) ? feed.steps : [];
    const items = steps.map((s) => ({ ms: parseUtc(s.time), step: Number(s.step), html: stepRow(s, expanded) }));
    // Only events inside the window the steps cover, or a long history of
    // pauses would push the steps out of the feed.
    const stepTimes = items.map((i) => i.ms).filter(Number.isFinite);
    const oldest = stepTimes.length ? Math.min(...stepTimes) : -Infinity;
    const events = (Array.isArray(feed.events) ? feed.events : [])
      .filter((e) => !(parseUtc(e.time) < oldest))
      .slice(0, MAX_EVENTS);
    for (const e of events) items.push({ ms: parseUtc(e.time), html: eventRow(e) });
    const key = (ms) => (Number.isFinite(ms) ? ms : -Infinity);
    items.sort((a, b) => key(b.ms) - key(a.ms));

    let html = renderStrip(feed);
    if (feed.paused) html += `<div class="feed-paused">AI is paused</div>`;
    const f = feed.in_flight;
    if (f) {
      const since = parseUtc(f.since);
      // The model that chose the last step is the one thinking now, unless the
      // backend switches mid-step (then the 🔁 row says so).
      const model = steps[0] && steps[0].model ? `<span class="model">${escapeHtml(steps[0].model)}</span>` : "";
      html += `<div class="inflight"><span class="n">#${escapeHtml(f.step)}</span>`
        + `<span class="thinking" data-since-ms="${Number.isFinite(since) ? since : ""}">`
        + `thinking… ${fmtElapsed(since, nowMs)}</span>${model}</div>`;
    }
    if (!items.length && !f) return `${html}<div class="empty">No steps yet.</div>`;
    return `${html}<div class="steps">${withRestartDividers(items).map((i) => i.html).join("")}</div>`;
  }

  // Newest first, step numbers fall; where an older step's number is not below the
  // newer one's, the server was restarted in between (its numbering starts again).
  // Seen on David's page after a restart: "#2 thinking, #1, #7, #6, #5…".
  function withRestartDividers(items) {
    const out = [];
    let newer = null;
    for (const item of items) {
      if (Number.isFinite(item.step)) {
        if (newer && item.step >= newer.step) {
          out.push({ html: `<div class="event restart">— restarted ${fmtClock(newer.ms)} —</div>` });
        }
        newer = item;
      }
      out.push(item);
    }
    return out;
  }

  // ---------- Party rail ----------

  function hpClass(hp, max) {
    if (!(hp > 0)) return "hp-zero";
    const pct = max > 0 ? (hp / max) * 100 : 0;
    if (pct > 50) return "hp-high";
    return pct > 20 ? "hp-mid" : "hp-low";
  }

  // A compact card: six have to fit the rail's height without scrolling, so the
  // ability, nickname and move types move to hover titles.
  function monCard(p) {
    const lead = (p.lead === undefined ? p.slot === 1 : p.lead) ? `<span class="tag lead">LEAD</span>` : "";
    if (p.egg) {
      return `<div class="mon egg"><div class="mon-head"><span class="species">Egg</span>${lead}</div></div>`;
    }
    const hp = Number(p.hp) || 0;
    const max = Number(p.max_hp) || 0;
    const pct = max > 0 ? Math.max(0, Math.min(100, Math.round((hp / max) * 100))) : 0;
    const nick = p.nickname && String(p.nickname).toUpperCase() !== String(p.species).toUpperCase()
      ? `“${p.nickname}”` : "";
    let status = "";
    if (hp <= 0) status = `<span class="status">Fainted</span>`;
    else if (p.status && !HEALTHY_STATUSES.has(String(p.status).toUpperCase())) {
      status = `<span class="status">${escapeHtml(p.status)}</span>`;
    }
    const moves = (Array.isArray(p.moves) ? p.moves : []).map((m) => {
      const noPp = m.pp === 0 ? " no-pp" : "";
      return `<li class="move${noPp}" title="${escapeHtml(m.type)}"><span class="mv">${escapeHtml(m.name)}</span>`
        + `<span class="pp">${escapeHtml(m.pp)}/${escapeHtml(m.max_pp)}</span></li>`;
    }).join("");
    const about = [nick, p.ability].filter(Boolean).join(" · ");
    return `<div class="mon${hp <= 0 ? " fainted" : ""}${lead ? " lead" : ""}">`
      + `<div class="mon-head"><span class="species" title="${escapeHtml(about)}">${escapeHtml(p.species)}</span>${lead}`
      + `<span class="lv">Lv${escapeHtml(p.level)}</span></div>`
      + `<div class="hp-row"><div class="hpbar"><div class="hp-fill ${hpClass(hp, max)}" style="width:${pct}%"></div></div>`
      + `<span class="hp-text">${hp}/${max}</span>${status}</div>`
      + `<ul class="moves">${moves}</ul></div>`;
  }

  function renderParty(dash) {
    if (!dash) return `<div class="empty">Waiting for game state…</div>`;
    const party = Array.isArray(dash.party) ? dash.party : [];
    return (dash.exp_note ? `<div class="warn-strip">${escapeHtml(dash.exp_note)}</div>` : "")
      + (party.length ? party.map(monCard).join("") : `<div class="empty">No Pokémon.</div>`);
  }

  // The party's total HP as one number for the rail's header ("93% HP").
  function partyHealth(dash) {
    const party = dash && Array.isArray(dash.party) ? dash.party.filter((p) => !p.egg) : [];
    const max = party.reduce((a, p) => a + (Number(p.max_hp) || 0), 0);
    if (!max) return "";
    const hp = party.reduce((a, p) => a + Math.max(0, Number(p.hp) || 0), 0);
    return `${Math.round((hp / max) * 100)}% HP`;
  }

  // ---------- Objectives ----------

  function trackSummary(track) {
    const list = track && Array.isArray(track.objectives) ? track.objectives : [];
    return {
      list,
      current: list.find((o) => o.status === "current") || null,
      done: track && typeof track.done === "number" ? track.done : list.filter((o) => o.status === "done").length,
      total: track && typeof track.total === "number" ? track.total : list.length,
    };
  }

  function stripCell(label, track, which) {
    const { current, done, total } = trackSummary(track);
    const pct = total ? Math.round((done / total) * 100) : 0;
    let body;
    if (current) {
      const target = current.target_location ? `Target <b>${escapeHtml(current.target_location)}</b> · ` : "";
      body = `<div class="obj-desc" title="${escapeHtml(current.navigation_hint || "")}">${escapeHtml(current.description)}</div>`
        + `<div class="obj-target">${target}<button type="button" class="linkish" data-open-track="${which}">See all</button></div>`;
    } else {
      body = `<div class="obj-desc muted">${total ? "All done." : "Loading… (objectives load with the AI's first step)"}</div>`;
    }
    return `<div class="obj-cell" data-track-cell="${which}"><div class="obj-top"><span class="label">${label}</span>`
      + `<span class="meter"><i style="width:${pct}%"></i></span><span class="num">${done}/${total}</span></div>${body}</div>`;
  }

  // The two current objectives under the game screen, always in view; the full lists
  // are one click away in the Objectives tab.
  function renderObjectiveStrip(objectives) {
    if (!objectives) return `<div class="obj-cell muted">Waiting for objectives…</div>`;
    return stripCell("Story", objectives.story, "story") + stripCell("Battle", objectives.battling, "battle");
  }

  // Chapters are the id prefix before the first "_" (early_001 → early), kept in
  // first-appearance order, which is story order.
  function groupChapters(objectives) {
    const chapters = [];
    const byKey = {};
    for (const o of Array.isArray(objectives) ? objectives : []) {
      const id = String(o.id || "");
      const cut = id.indexOf("_");
      const key = cut === -1 ? id : id.slice(0, cut);
      if (!byKey[key]) {
        byKey[key] = { key, items: [], done: 0, total: 0, hasCurrent: false };
        chapters.push(byKey[key]);
      }
      const c = byKey[key];
      c.items.push(o);
      c.total += 1;
      if (o.status === "done") c.done += 1;
      if (o.status === "current") c.hasCurrent = true;
    }
    return chapters;
  }

  function objRow(o) {
    if (o.status === "done") {
      return `<li class="obj done"><span class="mark">✓</span><span class="desc">${escapeHtml(o.description)}</span>`
        + `<span class="when">${fmtStamp(parseUtc(o.completed_at))}</span></li>`;
    }
    if (o.status === "current") {
      const hint = o.navigation_hint ? `<div class="hint">${escapeHtml(o.navigation_hint)}</div>` : "";
      const target = o.target_location ? `<div class="target">Target: ${escapeHtml(o.target_location)}</div>` : "";
      return `<li class="obj current scroll-target"><span class="mark">▶</span>`
        + `<div class="desc"><div>${escapeHtml(o.description)}</div>${hint}${target}</div></li>`;
    }
    return `<li class="obj upcoming"><span class="mark">○</span><span class="desc">${escapeHtml(o.description)}</span></li>`;
  }

  function capitalize(s) { return s ? s[0].toUpperCase() + s.slice(1) : "Other"; }

  // openChapters holds the user's toggles by chapter key; without one a chapter
  // is open unless every objective in it is done. track is "story" (default) or
  // "battle": one list at a time, so the panel's auto-scroll has one current row.
  function renderProgress(objectives, badges, openChapters, track) {
    if (!objectives) return `<div class="empty">Waiting for objectives…</div>`;
    const toggles = openChapters || {};
    const showBattle = track === "battle";
    const story = trackSummary(objectives.story);
    const battle = trackSummary(objectives.battling);
    const pips = BADGE_NAMES.map((name, i) => {
      const got = Array.isArray(badges) && badges[i];
      return `<span class="pip${got ? " earned" : ""}" title="${name} Badge${got ? "" : " (not yet)"}">${name}</span>`;
    }).join("");
    const seg = `<div class="seg" role="group" aria-label="Objective list">`
      + `<button type="button" data-track="story" aria-pressed="${!showBattle}">Story <span class="mono">${story.done}/${story.total}</span></button>`
      + `<button type="button" data-track="battle" aria-pressed="${showBattle}">Battle <span class="mono">${battle.done}/${battle.total}</span></button></div>`;
    const top = `<div class="prog-top">${seg}<div class="pips">${pips}</div></div>`;

    let body;
    if (showBattle) {
      body = battle.list.length
        ? `<ol class="objs battling">${battle.list.map(objRow).join("")}</ol>`
        : `<div class="empty">No battling goals.</div>`;
    } else if (!story.list.length) {
      // Right after a server restart the objectives are loaded with the agent's first
      // step; until then /objectives has none ("0 of 0 … done" read like a fault).
      body = `<div class="empty">Loading objectives… (they load with the AI's first step)</div>`;
    } else {
      body = groupChapters(story.list).map((c) => {
        const finished = c.done === c.total;
        const open = toggles[c.key] === undefined ? !finished : Boolean(toggles[c.key]);
        const rows = open ? `<ol class="objs">${c.items.map(objRow).join("")}</ol>` : "";
        return `<section class="chapter${open ? " open" : ""}${finished ? " finished" : ""}">`
          + `<button type="button" class="chap-head" data-chapter="${escapeHtml(c.key)}" aria-expanded="${open}">`
          + `<span class="caret">${open ? "▾" : "▸"}</span><span class="chap-name">${escapeHtml(capitalize(c.key))}</span>`
          + `<span class="count">${c.done}/${c.total}</span></button>${rows}</section>`;
      }).join("");
    }
    return `<div class="prog">${top}${body}</div>`;
  }

  // A different /health server_id than the one this page first saw: the game server
  // restarted, maybe with new page code, so the page reloads once. David (2026-11-04)
  // keeps one tab open now that restarts no longer open a new one.
  function serverRestarted(seenId, health) {
    const id = health && health.server_id;
    return Boolean(seenId && id && id !== seenId);
  }

  return {
    escapeHtml, parseUtc, fmtClock, fmtElapsed, fmtCall,
    renderFacts, renderFeed, renderStrip, renderParty, partyHealth,
    renderObjectiveStrip, renderProgress, groupChapters, trackSummary, serverRestarted,
  };
})();

if (typeof module !== "undefined") module.exports = PlayPanels;
else Object.assign(globalThis, PlayPanels);
