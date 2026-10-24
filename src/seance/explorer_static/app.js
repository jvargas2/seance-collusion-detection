// Séance explorer: #/ lists every game, #/g000 opens one. Data comes from
// data/index.js and data/<id>.js (script files, so a local copy works too).
(function () {
  "use strict";
  const INDEX = window.SEANCE_INDEX;
  const app = document.getElementById("app");
  const NAMES = ["Daniel", "Eileen", "Gladys", "Leonora"];
  const color = (n) => `var(--m${NAMES.indexOf(n)})`;
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const pct = (x, d = 0) => (x == null ? "" : x === 0 ? "0%" : x > 0 && x < 0.001 ? "under 0.1%" : `${(100 * x).toFixed(d)}%`);
  const pairText = (label) => (label == null ? "No single pair" : label === "none" ? "No one" : label.split("+").join(" and "));
  const swatches = (label) => (label && label !== "none" ? label.split("+").map((n) => `<span class="sw" style="background:${color(n)}"></span>`).join("") : "");
  const CONDITION = { clean: "Clean", coalition: "Coalition", positive_control: "Positive control" };
  const state = { filter: "all", sort: "suspicion" };

  function outcome(g) {
    if (g.condition === "clean") return g.flagged ? "False alarm: flagged an honest table" : g.p_none < 0.5 ? "Suspected a pair, but below the flag line" : "Cleared";
    if (g.caught) return `Caught; their ${fmt(g.pooled)} pooled points were wiped`;
    if (g.flagged) return g.top == null ? `Flagged, but no single pair was named; kept ${fmt(g.pooled)} points` : `Flagged, but the wrong pair was named; kept ${fmt(g.pooled)} points`;
    return `Got away with ${fmt(g.pooled)} pooled points`;
  }
  const fmt = (x) => (Number.isInteger(x) ? String(x) : x.toFixed(1));

  function meter(s, threshold) {
    const over = s > threshold;
    return `<span class="meter" aria-hidden="true"><span class="base"></span><span class="trail${over ? " over" : ""}" style="width:${(100 * s).toFixed(1)}%"></span><span class="line" style="left:${(100 * threshold).toFixed(1)}%"></span></span>${pct(s, 1)}`;
  }

  const FILTERS = [
    ["all", "All games", () => true],
    ["clean", "Clean", (g) => g.condition === "clean"],
    ["coalition", "Coalition", (g) => g.condition === "coalition"],
    ["positive_control", "Positive control", (g) => g.condition === "positive_control"],
    ["suspected", "Honest tables it suspected", (g) => g.condition === "clean" && g.p_none < 0.5],
    ["caught", "Caught", (g) => g.caught],
  ];

  function browse() {
    const f = INDEX.facts;
    const pred = FILTERS.find((x) => x[0] === state.filter)[2];
    let games = INDEX.games.filter(pred);
    games = games.slice().sort((a, b) => (state.sort === "id" ? a.id.localeCompare(b.id) : (b.suspicion ?? -1) - (a.suspicion ?? -1)));
    app.innerHTML = `
      <h1>Who was whispering?</h1>
      <p class="lede">106 recorded séances. In half of the main games, two mediums were secretly allied and passed their visions to each other through ordinary-looking talk. An auditor read each public record and tried to name the pair. Choose a game to read what everyone said, and what they privately thought.</p>
      <div class="facts">
        <div class="fact"><div class="v">${f.caught} of ${f.n_coalition}</div><div class="n">coalitions caught while flagging at most 5% of clean games</div></div>
        <div class="fact"><div class="v">${pct(f.right_pair)}</div><div class="n">of coalitions named correctly, against 17% by chance</div></div>
        <div class="fact"><div class="v">${f.auroc.toFixed(2)}</div><div class="n">AUROC, against ${f.auroc_top_scorer.toFixed(2)} for whoever scored most</div></div>
        <div class="fact"><div class="v">${Math.round(f.pooled_before)} to ${Math.round(f.pooled_after)}</div><div class="n">allies' average pooled points once caught pairs lose theirs (an average pair gets ${f.average_pair})</div></div>
      </div>
      <div class="controls">
        <div class="pills" role="group" aria-label="Show">${FILTERS.map(([k, t]) => `<button data-filter="${k}" aria-pressed="${state.filter === k}">${t}</button>`).join("")}</div>
        <span class="count">${games.length} games</span>
      </div>
      <table>
        <thead><tr>
          <th><button data-sort="id" ${state.sort === "id" ? 'aria-sort="ascending"' : ""}>Game</button></th>
          <th>Who was really allied</th>
          <th><button data-sort="suspicion" ${state.sort === "suspicion" ? 'aria-sort="descending"' : ""}>Auditor's suspicion</button></th>
          <th>Most suspected pair</th>
          <th class="hide-sm">What happened</th>
        </tr></thead>
        <tbody>${games.map((g) => {
          const right = g.condition === "clean" ? "" : g.top === g.label ? "right" : "wrong";
          return `<tr tabindex="0" data-id="${g.id}">
            <td class="id">${g.id}</td>
            <td class="pair">${swatches(g.label)}${pairText(g.label)}</td>
            <td>${g.suspicion == null ? "not scored" : meter(g.suspicion, f.threshold)}</td>
            <td class="pair ${right}">${pairText(g.top)}${right ? ` (${right})` : ""}</td>
            <td class="hide-sm">${outcome(g)}</td></tr>`;
        }).join("")}</tbody>
      </table>
      <p class="note" style="margin-top:14px">Suspicion is the auditor's probability that the table held a coalition. The pale tick marks the flag line, set so that at most 5% of clean games are flagged.</p>`;
    app.querySelectorAll("[data-filter]").forEach((b) => (b.onclick = () => { state.filter = b.dataset.filter; browse(); }));
    app.querySelectorAll("[data-sort]").forEach((b) => (b.onclick = () => { state.sort = b.dataset.sort; browse(); }));
    app.querySelectorAll("tbody tr").forEach((tr) => {
      const open = () => (location.hash = `#/${tr.dataset.id}`);
      tr.onclick = open;
      tr.onkeydown = (e) => { if (e.key === "Enter") open(); };
    });
  }

  function loadGame(id, done) {
    window.SEANCE_GAME = null;
    const s = document.createElement("script");
    s.src = `data/${id}.js`;
    s.onload = () => { done(window.SEANCE_GAME); s.remove(); };
    s.onerror = () => { app.innerHTML = `<a class="back" href="#/">Back to all games</a><h2>No game called ${esc(id)}</h2>`; };
    document.body.appendChild(s);
  }

  function numberLine(g, r) {
    const W = 1000, rowH = 26, top = 8, H = top + rowH * 4 + 30, L = 90, R = 16;
    const lo = -40, hi = 140, x = (v) => L + ((v - lo) / (hi - lo)) * (W - L - R);
    let s = `<svg class="numline" viewBox="0 0 ${W} ${H}" role="img" aria-label="Round ${r.round}: each medium's vision as a bar, their guess as a dot, and the knock at ${r.secret}">`;
    s += `<rect x="${x(1)}" y="${top - 4}" width="${x(100) - x(1)}" height="${rowH * 4 + 8}" rx="4" fill="rgba(255,255,255,0.025)"/>`;
    [1, 25, 50, 75, 100].forEach((t) => { s += `<line x1="${x(t)}" x2="${x(t)}" y1="${top - 4}" y2="${top + rowH * 4 + 4}" stroke="#22323a"/><text x="${x(t)}" y="${H - 6}" text-anchor="middle">${t}</text>`; });
    g.names.forEach((n, i) => {
      const y = top + rowH * i + rowH / 2, [a, b] = r.visions[n];
      s += `<text x="${L - 12}" y="${y + 4}" text-anchor="end" style="fill:#a3b6b0">${n}</text>`;
      s += `<rect x="${x(a)}" y="${y - 4}" width="${x(b) - x(a)}" height="8" rx="4" fill="${color(n)}" opacity="0.45"><title>${n}'s vision: ${a} to ${b}</title></rect>`;
      s += `<circle cx="${x(r.guesses[n])}" cy="${y}" r="7" fill="${color(n)}" stroke="#0a0f12" stroke-width="2"><title>${n} guessed ${r.guesses[n]}</title></circle>`;
    });
    s += `<line x1="${x(r.secret)}" x2="${x(r.secret)}" y1="${top - 6}" y2="${top + rowH * 4 + 6}" stroke="#f1f8f5" stroke-width="2" stroke-dasharray="4 4"/>`;
    s += `<text x="${x(r.secret) + 6}" y="${top + 6}" style="fill:#f1f8f5">knock at ${r.secret}</text>`;
    return s + "</svg>";
  }

  function game(g, roundIx) {
    const m = g.monitor;
    const allies = g.label === "none" ? [] : g.label.split("+");
    const cond = g.condition === "clean" ? "A clean game: no one was allied." : `${CONDITION[g.condition]} game. ${pairText(g.label)} were allied${g.condition === "positive_control" ? " and were handed a simple code" : ""}.`;
    let verdict = "This game was not scored by the auditor.";
    if (m) {
      const sure = `<em>${pct(m.suspicion, 1)} sure</em>`;
      if (g.condition === "clean") verdict = m.flagged ? `The auditor was ${sure} of a coalition and would have accused ${pairText(m.top)}, an honest pair. A false alarm.` : `The auditor was ${sure} of a coalition, most suspecting ${pairText(m.top)}, but stayed below the flag line.`;
      else if (m.caught) verdict = `The auditor was ${sure} of a coalition and named ${pairText(m.top)}. It was right, so their ${fmt(g.pooled)} pooled points were wiped.`;
      else if (m.flagged) verdict = `The auditor was ${sure} of a coalition but named ${pairText(m.top)}, the wrong pair. The allies kept their ${fmt(g.pooled)} points.`;
      else verdict = `The auditor was ${sure} of a coalition, below the flag line, so ${pairText(g.label)} got away with ${fmt(g.pooled)} points.`;
    }
    const probs = m ? Object.entries(m.probs).sort((a, b) => b[1] - a[1]) : [];
    const r = g.rounds[roundIx];
    app.innerHTML = `
      <a class="back" href="#/">Back to all games</a>
      <h2>Game ${g.id}</h2>
      <p class="lede" style="margin-top:6px">${cond}</p>
      <div class="grid2">
        <div>
          <p class="verdict">${verdict}</p>
          <div class="scores">${g.names.slice().sort((a, b) => g.scores[b] - g.scores[a]).map((n) => `<span><strong>${fmt(g.scores[n])}</strong>${n}${allies.includes(n) ? " (ally)" : ""}</span>`).join("")}</div>
        </div>
        <div>
          ${m ? `<h3>What the auditor believed</h3>
          ${probs.map(([k, p], i) => `<div class="prob${i === 0 ? " top" : ""}"><span>${k === "none" ? "No coalition" : pairText(k)}</span><div class="bar"><i style="width:${Math.max(100 * p, 1)}%"></i></div><span>${pct(p, p < 0.01 ? 1 : 0)}</span></div>`).join("")}
          <h3 style="margin-top:24px">Its three readings</h3>
          ${m.readings.map((rd, i) => `<details><summary>Reading ${i + 1}: ${rd.top ? `${pairText(rd.top)}, ${pct(rd.probs[rd.top])}` : "no single most likely pair"}; no coalition ${pct(rd.probs.none, rd.probs.none < 0.01 ? 1 : 0)}</summary><p>${esc(rd.reasoning)}</p></details>`).join("")}` : ""}
          ${g.again.length ? `<h3 style="margin-top:24px">Read again with the record changed</h3><div class="list">${g.again.map((a) => `<div><span>${a.title}</span><span class="${g.condition === "clean" ? "" : a.top === g.label ? "right" : "wrong"}">${pairText(a.top)}${a.p_top != null ? `, ${pct(a.p_top)}` : ""}${g.condition === "clean" ? "" : a.top === g.label ? " (right)" : " (wrong)"}</span></div>`).join("")}</div>` : ""}
        </div>
      </div>
      <div class="section">
        <h3>The record, round by round</h3>
        <div class="rounds" role="group" aria-label="Round">${g.rounds.map((x, i) => `<button data-round="${i}" ${i === roundIx ? 'aria-current="true"' : ""} aria-label="Round ${x.round}">${x.round}</button>`).join("")}</div>
        ${numberLine(g, r)}
        <div class="colhead"><span></span><span>Said to the table</span><span>Thought privately</span></div>
        ${g.names.map((n) => `<div class="turn">
          <div class="who"><b><span class="sw" style="background:${color(n)}"></span>${n}</b><small>${allies.includes(n) ? "Ally. " : ""}Vision ${r.visions[n][0]} to ${r.visions[n][1]}. Guessed ${r.guesses[n]}, scored ${fmt(r.points[n])}</small></div>
          <p class="said">“${esc(r.messages[n])}”</p>
          <div class="thought">${esc(r.speak_notes[n])}${r.guess_notes[n] ? `<details class="later"><summary>What they thought before guessing</summary>${esc(r.guess_notes[n])}</details>` : ""}</div>
        </div>`).join("")}
        <p class="knock">The knock came at minute ${r.secret}.</p>
      </div>`;
    app.querySelectorAll("[data-round]").forEach((b) => (b.onclick = () => { game(g, Number(b.dataset.round)); }));
  }

  function route() {
    const id = location.hash.replace(/^#\/?/, "");
    window.scrollTo(0, 0);
    if (!id) return browse();
    loadGame(id, (g) => game(g, 0));
  }
  window.addEventListener("hashchange", route);
  route();
})();
