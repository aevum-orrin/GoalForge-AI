const $ = (id) => document.getElementById(id);

async function api(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    const e = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(e.detail || r.statusText);
  }
  return r.json();
}

async function init() {
  const { teams } = await api('/api/teams');
  const opts = teams.map((t) => `<option>${t}</option>`).join('');
  $('home').innerHTML = opts;
  $('away').innerHTML = opts;
  $('away').selectedIndex = Math.min(1, teams.length - 1);
  await Promise.all([loadXI('home'), loadXI('away')]);
  $('home').onchange = () => loadXI('home');
  $('away').onchange = () => loadXI('away');
  $('nsims').oninput = (e) => ($('nsimsv').textContent = e.target.value);
  $('go').onclick = predict;
}

async function loadXI(side) {
  const team = $(side).value;
  $(side === 'home' ? 'homeName' : 'awayName').textContent = team;
  const { players, default_xi } = await api(`/api/squad?team=${encodeURIComponent(team)}`);
  const def = new Set(default_xi);
  $(side === 'home' ? 'homeXI' : 'awayXI').innerHTML = players
    .map((p) => `<label><input type="checkbox" value="${p}" ${def.has(p) ? 'checked' : ''}/> ${p}</label>`)
    .join('');
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
    const body = {
      home_team: $('home').value, away_team: $('away').value,
      home_xi: xiOf('home'), away_xi: xiOf('away'),
      neutral: $('neutral').checked, n_sims: +$('nsims').value,
    };
    const p = await api('/api/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
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

init().catch((e) => ($('err').textContent = 'Init error: ' + e.message));
