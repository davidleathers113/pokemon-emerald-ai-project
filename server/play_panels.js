// Renderers for the controller page's live panels (AI feed, party, progress).
// Pure functions: JSON in, HTML string out, no DOM access, so
// local_tests/test_play_panels.js can exercise them under plain node.
// play.html does the polling, DOM writes and click handling.
const PlayPanels = (() => {
  const BADGE_NAMES = ["Stone", "Knuckle", "Dynamo", "Heat", "Balance", "Feather", "Mind", "Rain"];
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const MAX_EVENTS = 10;

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

  function stepRow(s, expanded) {
    const open = expanded && expanded[s.step] ? " open" : "";
    const fail = s.success === false ? " fail" : "";
    const dur = typeof s.duration_s === "number" ? `${s.duration_s.toFixed(1)} s` : "";
    const reasoning = s.reasoning
      ? `<div class="reasoning" title="Click to expand">${escapeHtml(s.reasoning)}</div>` : "";
    const result = s.result === null || s.result === undefined
      ? `<div class="result pending">…</div>`
      : `<div class="result${fail}">${escapeHtml(s.result)}</div>`;
    return `<div class="step${fail}${open}" data-step="${escapeHtml(s.step)}">`
      + `<div class="step-head"><span class="t">${fmtClock(parseUtc(s.time))}</span>`
      + `<span class="n">#${escapeHtml(s.step)}</span>`
      + `<code class="call">${escapeHtml(fmtCall(s.tool, s.args))}</code>`
      + `<span class="dur">${dur}</span></div>${reasoning}${result}</div>`;
  }

  function eventRow(e) {
    const source = escapeHtml(e.source || "unknown");
    const text = e.text ? `: ${escapeHtml(e.text)}` : "";
    let label = `${escapeHtml(e.kind)} by ${source}${text}`;
    if (e.kind === "pause") label = `⏸ Paused by ${source}${text}`;
    if (e.kind === "resume") label = `▶ Resumed by ${source}${text}`;
    return `<div class="event ${e.kind === "pause" ? "ev-pause" : "ev-resume"}">`
      + `<span class="t">${fmtClock(parseUtc(e.time))}</span><span>${label}</span></div>`;
  }

  // nowMs is "now" on the server's clock (play.html corrects for skew via feed.now),
  // so the thinking counter doesn't drift when the browser's clock is off.
  function renderFeed(feed, nowMs, expanded) {
    if (!feed) return `<div class="empty">Waiting for the AI…</div>`;
    const steps = Array.isArray(feed.steps) ? feed.steps : [];
    const items = steps.map((s) => ({ ms: parseUtc(s.time), html: stepRow(s, expanded) }));
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

    let html = feed.paused ? `<div class="feed-paused">AI is paused</div>` : "";
    const f = feed.in_flight;
    if (f) {
      const since = parseUtc(f.since);
      html += `<div class="step inflight"><div class="step-head">`
        + `<span class="t">${fmtClock(since)}</span><span class="n">#${escapeHtml(f.step)}</span>`
        + `<span class="thinking" data-since-ms="${Number.isFinite(since) ? since : ""}">`
        + `thinking… ${fmtElapsed(since, nowMs)}</span></div></div>`;
    }
    if (!items.length && !f) return `${html}<div class="empty">No steps yet.</div>`;
    return html + items.map((i) => i.html).join("");
  }

  function hpClass(hp, max) {
    if (!(hp > 0)) return "hp-zero";
    const pct = max > 0 ? (hp / max) * 100 : 0;
    if (pct > 50) return "hp-high";
    return pct > 20 ? "hp-mid" : "hp-low";
  }

  function monCard(p) {
    const lead = (p.lead === undefined ? p.slot === 1 : p.lead) ? `<span class="tag lead">LEAD</span>` : "";
    if (p.egg) {
      return `<div class="mon egg"><div class="mon-head"><span class="species">Egg</span>${lead}</div></div>`;
    }
    const hp = Number(p.hp) || 0;
    const max = Number(p.max_hp) || 0;
    const pct = max > 0 ? Math.max(0, Math.min(100, Math.round((hp / max) * 100))) : 0;
    const nick = p.nickname && String(p.nickname).toUpperCase() !== String(p.species).toUpperCase()
      ? ` <span class="nick">“${escapeHtml(p.nickname)}”</span>` : "";
    let status = "";
    if (hp <= 0) status = ` · <span class="status">Fainted</span>`;
    else if (p.status && p.status !== "OK") status = ` · <span class="status">${escapeHtml(p.status)}</span>`;
    const moves = (Array.isArray(p.moves) ? p.moves : []).map((m) => {
      const noPp = m.pp === 0 ? " no-pp" : "";
      return `<li class="move${noPp}"><span class="mv">${escapeHtml(m.name)}</span>`
        + `<span class="type">${escapeHtml(m.type)}</span>`
        + `<span class="pp">${escapeHtml(m.pp)}/${escapeHtml(m.max_pp)}</span></li>`;
    }).join("");
    return `<div class="mon${hp <= 0 ? " fainted" : ""}">`
      + `<div class="mon-head"><span class="species">${escapeHtml(p.species)}</span>${nick}`
      + `<span class="lv">Lv${escapeHtml(p.level)}</span>${lead}</div>`
      + `<div class="hpbar"><div class="hp-fill ${hpClass(hp, max)}" style="width:${pct}%"></div></div>`
      + `<div class="hp-text">${hp}/${max} HP${status}</div>`
      + (p.ability ? `<div class="ability">${escapeHtml(p.ability)}</div>` : "")
      + `<ul class="moves">${moves}</ul></div>`;
  }

  function renderParty(dash) {
    if (!dash) return `<div class="empty">Waiting for game state…</div>`;
    const pos = Array.isArray(dash.position) && dash.position.length >= 2
      ? ` (${escapeHtml(dash.position[0])}, ${escapeHtml(dash.position[1])})` : "";
    const badges = Array.isArray(dash.badges) ? dash.badges.filter(Boolean).length : 0;
    const balls = (Array.isArray(dash.balls) ? dash.balls : [])
      .map((b) => `${escapeHtml(b.name)} ×${escapeHtml(b.count)}`).join(" · ");
    const party = Array.isArray(dash.party) ? dash.party : [];
    return `<div class="party-sum">`
      + `<div><b>${escapeHtml(dash.location || "Unknown")}</b>${pos}</div>`
      + `<div>₽${fmtMoney(dash.money)} · Badges ${badges}/8</div>`
      + `<div class="balls">${balls || "No balls"}</div></div>`
      + (dash.exp_note ? `<div class="warn-strip">${escapeHtml(dash.exp_note)}</div>` : "")
      + `<div class="mons">${party.length ? party.map(monCard).join("") : `<div class="empty">No Pokémon.</div>`}</div>`;
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

  function objRow(o, scrollTarget) {
    if (o.status === "done") {
      return `<li class="obj done"><span class="mark">✓</span><span class="desc">${escapeHtml(o.description)}</span>`
        + `<span class="when">${fmtStamp(parseUtc(o.completed_at))}</span></li>`;
    }
    if (o.status === "current") {
      const hint = o.navigation_hint ? `<div class="hint">${escapeHtml(o.navigation_hint)}</div>` : "";
      const target = o.target_location ? `<div class="target">Target: ${escapeHtml(o.target_location)}</div>` : "";
      return `<li class="obj current${scrollTarget ? " scroll-target" : ""}"><span class="mark">▶</span>`
        + `<div class="desc"><div>${escapeHtml(o.description)}</div>${hint}${target}</div></li>`;
    }
    return `<li class="obj upcoming"><span class="mark">○</span><span class="desc">${escapeHtml(o.description)}</span></li>`;
  }

  function capitalize(s) { return s ? s[0].toUpperCase() + s.slice(1) : "Other"; }

  // openChapters holds the user's toggles by chapter key; without one a chapter
  // is open unless every objective in it is done.
  function renderProgress(objectives, badges, openChapters) {
    if (!objectives) return `<div class="empty">Waiting for objectives…</div>`;
    const toggles = openChapters || {};
    const story = objectives.story || { objectives: [] };
    const list = Array.isArray(story.objectives) ? story.objectives : [];
    const done = typeof story.done === "number" ? story.done : list.filter((o) => o.status === "done").length;
    const total = typeof story.total === "number" ? story.total : list.length;
    const pips = BADGE_NAMES.map((name, i) => {
      const got = Array.isArray(badges) && badges[i];
      return `<span class="pip${got ? " earned" : ""}" title="${name} Badge${got ? "" : " (not yet)"}">${name}</span>`;
    }).join("");

    const chapters = groupChapters(list).map((c) => {
      const finished = c.done === c.total;
      const open = toggles[c.key] === undefined ? !finished : Boolean(toggles[c.key]);
      const rows = open ? `<ol class="objs">${c.items.map((o) => objRow(o, true)).join("")}</ol>` : "";
      return `<section class="chapter${open ? " open" : ""}${finished ? " finished" : ""}">`
        + `<button type="button" class="chap-head" data-chapter="${escapeHtml(c.key)}" aria-expanded="${open}">`
        + `<span class="caret">${open ? "▾" : "▸"}</span><span class="chap-name">${escapeHtml(capitalize(c.key))}</span>`
        + `<span class="count">${c.done}/${c.total}</span></button>${rows}</section>`;
    }).join("");

    const battle = objectives.battling;
    const blist = battle && Array.isArray(battle.objectives) ? battle.objectives : [];
    const bdone = blist.filter((o) => o.status === "done").length;
    const battling = blist.length
      ? `<h3 class="battle-head">Battling goals <span class="count">${bdone}/${blist.length}</span></h3>`
        + `<ol class="objs battling">${blist.map((o) => objRow(o, false)).join("")}</ol>`
      : "";

    return `<div class="prog-top"><div class="prog-head">${escapeHtml(done)} of ${escapeHtml(total)} story objectives done</div>`
      + `<div class="pips">${pips}</div></div>`
      + (chapters || `<div class="empty">No story objectives.</div>`) + battling;
  }

  return {
    escapeHtml, parseUtc, fmtClock, fmtElapsed, fmtCall,
    renderFeed, renderParty, renderProgress, groupChapters,
  };
})();

if (typeof module !== "undefined") module.exports = PlayPanels;
else Object.assign(globalThis, PlayPanels);
