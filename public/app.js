const $ = (id) => document.getElementById(id);

let META = {};
let HOSTS = new Set();
const isHost = (t) => HOSTS.has(t);

async function api(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    const e = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(e.detail || `HTTP ${r.status}`);
  }
  return r.json();
}

async function init() {
  const { teams, meta } = await api('/api/teams');
  META = meta || {};
  HOSTS = new Set(META.hosts || []);
  const opts = teams.map((t) => `<option>${t}</option>`).join('');
  $('home').innerHTML = opts;
  $('away').innerHTML = opts;
  $('away').selectedIndex = Math.min(1, teams.length - 1);
  updateVenue();
  await Promise.all([loadXI('home'), loadXI('away')]);
  $('home').onchange = () => { updateVenue(); loadXI('home'); };
  $('away').onchange = () => { updateVenue(); loadXI('away'); };
  $('neutral').onchange = renderVenueNote;
  $('go').onclick = predict;
  renderMethod();
}

// --- venue: neutral by default; a single 2026 host gets home advantage ---------------
function updateVenue() {
  const oneHost = isHost($('home').value) !== isHost($('away').value); // XOR
  $('neutral').checked = !oneHost;
  renderVenueNote();
}

function renderVenueNote() {
  const h = $('home').value, a = $('away').value;
  if ($('neutral').checked) {
    $('venueNote').textContent = 'Neutral venue — World Cup default (no home advantage).';
  } else {
    const host = isHost(h) ? h : isHost(a) ? a : h;
    $('venueNote').textContent = `🏠 Home advantage: ${host}${isHost(host) ? ' (2026 host)' : ''}`;
  }
}

async function loadXI(side) {
  const team = $(side).value;
  $(side === 'home' ? 'homeName' : 'awayName').textContent = team;
  const { players, default_xi, info } = await api(`/api/squad?team=${encodeURIComponent(team)}`);
  const def = new Set(default_xi);
  const row = (p) => {
    const i = info && info[p];
    const xa = i && i.club_xa != null ? ` · xA ${i.club_xa}` : '';
    const meta = i && i.caps != null
      ? `<span class="pmeta">${i.pos || ''} · ${i.caps} caps · ${i.goals} gls${xa}</span>` : '';
    return `<label><input type="checkbox" value="${p}" ${def.has(p) ? 'checked' : ''}/>
      <span class="pname">${p}</span>${meta}</label>`;
  };
  const starters = players.slice(0, 11).map(row).join('');
  const bench = players.slice(11).map(row).join('');
  $(side === 'home' ? 'homeXI' : 'awayXI').innerHTML =
    starters + (bench ? `<div class="benchsep">Bench</div>${bench}` : '');
}

const xiOf = (side) =>
  [...document.querySelectorAll(`#${side === 'home' ? 'homeXI' : 'awayXI'} input:checked`)].map((i) => i.value);

const pct = (x) => (x * 100).toFixed(0) + '%';

function bars(el, arr) {
  el.innerHTML = arr
    .map((o) => `<div class="bar"><span class="lbl">${o.player}</span>
      <span class="track"><span class="fill" style="width:${(o.prob * 100).toFixed(0)}%"></span></span>
      <span class="pct">${pct(o.prob)}</span></div>`)
    .join('');
}

async function predict() {
  $('err').textContent = '';
  $('go').disabled = true;
  $('go').textContent = 'Simulating…';
  try {
    let homeTeam = $('home').value, awayTeam = $('away').value;
    let homeXI = xiOf('home'), awayXI = xiOf('away');
    const neutral = $('neutral').checked;
    // If not neutral, the advantaged (home) side must be the host — reorder if the host was picked as away.
    if (!neutral && isHost(awayTeam) && !isHost(homeTeam)) {
      [homeTeam, awayTeam] = [awayTeam, homeTeam];
      [homeXI, awayXI] = [awayXI, homeXI];
    }
    const p = await api('/api/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam,
        home_xi: homeXI, away_xi: awayXI, neutral }),
    });
    $('results').classList.remove('hidden');
    $('matchTitle').textContent = `${p.home_team} vs ${p.away_team}`;
    const rows = [
      [p.home_team, p.prob_home], ['Draw', p.prob_draw], [p.away_team, p.prob_away],
    ];
    $('outcome').innerHTML = rows
      .map(([n, v]) => `<div class="obar"><span class="oname">${n}</span>
        <span class="track"><span class="fill" style="width:${(v * 100).toFixed(0)}%"></span></span>
        <span class="pct">${pct(v)}</span></div>`)
      .join('');
    $('score').innerHTML = `<b>${p.most_likely_score[0]}–${p.most_likely_score[1]}</b> most likely ·
      expected goals ${p.exp_home_goals.toFixed(2)}–${p.exp_away_goals.toFixed(2)} ·
      top: ${p.top_scores.map((s) => `${s.home}-${s.away} ${pct(s.prob)}`).join(', ')}`;
    $('hsTitle').textContent = `${p.home_team} — scorers`;
    $('asTitle').textContent = `${p.away_team} — scorers`;
    $('haTitle').textContent = `${p.home_team} — assisters`;
    $('aaTitle').textContent = `${p.away_team} — assisters`;
    bars($('homeScorers'), p.home_scorers);
    bars($('awayScorers'), p.away_scorers);
    bars($('homeAssists'), p.home_assisters);
    bars($('awayAssists'), p.away_assisters);
  } catch (e) {
    $('err').textContent = 'Error: ' + e.message;
  } finally {
    $('go').disabled = false;
    $('go').textContent = 'Predict';
  }
}

function renderMethod() {
  const m = META.method || {}, b = META.backtest || {};
  const li = (k, v) => (v ? `<li><b>${k}:</b> ${v}</li>` : '');
  $('methodBody').innerHTML =
    '<ul>' +
    li('Scoreline', m.scoreline) + li('Scorers', m.scorer) + li('Assists', m.assist) +
    li('Default XI', m.default_xi) + li('Venue', m.venue) + '</ul>' +
    (b.test_rps ? `<p class="mnote">Team layer held-out backtest on international matches:
       RPS ${b.test_rps} (lower is better — beats Elo &amp; base-rate). The scorer &amp; assist
       layers are history/prior-based and are <b>not</b> validated on 2026 outcomes.</p>` : '') +
    (META.note ? `<p class="mnote">${META.note}</p>` : '');
}

init().catch((e) => ($('err').textContent = 'Init error: ' + e.message));
