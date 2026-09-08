'use strict';

const $ = (id) => document.getElementById(id);

let current = null;      // the match payload last loaded
let sortKey = 'kills';
let sortDesc = true;
let selectedRound = -1;

const TEAM_CT = 3;
const TEAM_T = 2;

function teamClass(team) {
  if (team === TEAM_CT) return 'ct';
  if (team === TEAM_T) return 't';
  return 'none';
}

function setStatus(text, isError) {
  const el = $('status');
  el.textContent = text || '';
  el.classList.toggle('error', !!isError);
}

function formatDuration(seconds) {
  if (!seconds || seconds < 0) return '—';
  const total = Math.round(seconds);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}m ${String(s).padStart(2, '0')}s`;
}

async function load(target) {
  setStatus('Loading… (a first-time download can take a while)');
  $('match').hidden = true;
  try {
    const response = await fetch('/api/match?code=' + encodeURIComponent(target));
    const payload = await response.json();
    if (!response.ok) {
      setStatus(payload.error || `request failed with ${response.status}`, true);
      return;
    }
    current = payload;
    selectedRound = -1;
    render();
    setStatus('');
  } catch (err) {
    setStatus(String(err), true);
  }
}

function render() {
  const m = current.match;
  $('map-name').textContent = m.map || 'unknown map';
  const bits = [];
  if (m.roundsPlayed) bits.push(`${m.roundsPlayed} rounds`);
  if (m.playbackTime) bits.push(formatDuration(m.playbackTime));
  if (m.tickRate) bits.push(`${Math.round(m.tickRate)} tick`);
  if (m.serverName) bits.push(m.serverName);
  $('match-meta').textContent = bits.join(' · ');

  $('score-ct').textContent = m.scoreCt;
  $('score-t').textContent = m.scoreT;
  $('match-id').textContent = m.matchId !== '0' ? m.matchId : '—';
  $('outcome-id').textContent = m.outcomeId !== '0' ? m.outcomeId : '—';
  $('demo-source').textContent = m.demoSource || '—';

  renderScoreboard();
  renderRounds();

  const warnings = current.warnings || [];
  $('warnings-box').hidden = warnings.length === 0;
  $('warnings').innerHTML = '';
  for (const warning of warnings) {
    const li = document.createElement('li');
    li.textContent = warning;
    $('warnings').appendChild(li);
  }

  $('match').hidden = false;
}

function renderScoreboard() {
  const body = $('scoreboard').querySelector('tbody');
  body.innerHTML = '';

  const players = current.players.filter((p) => !p.hltv);
  const groups = [
    ['Counter-Terrorists', players.filter((p) => p.team === TEAM_CT)],
    ['Terrorists', players.filter((p) => p.team === TEAM_T)],
    ['Other', players.filter((p) => p.team !== TEAM_CT && p.team !== TEAM_T)],
  ];

  for (const [label, group] of groups) {
    if (group.length === 0) continue;
    group.sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey];
      const cmp = typeof av === 'number' ? bv - av : String(av).localeCompare(String(bv));
      return sortDesc ? cmp : -cmp;
    });

    const header = document.createElement('tr');
    header.className = 'team-header';
    header.innerHTML = `<td colspan="11">${label}</td>`;
    body.appendChild(header);

    for (const p of group) {
      const row = document.createElement('tr');
      row.className = 'team-' + teamClass(p.team);
      const name = document.createElement('td');
      name.className = 'name';
      name.textContent = p.name || `slot ${p.slot}`;
      if (p.bot) {
        const tag = document.createElement('span');
        tag.className = 'bot';
        tag.textContent = 'BOT';
        name.appendChild(tag);
      }
      row.appendChild(name);
      const cells = [
        p.kills, p.deaths, p.assists,
        p.kd.toFixed(2), p.adr.toFixed(1), p.hsPercent.toFixed(0) + '%',
        p.mvps, p.entryKills, p.enemiesFlashed, p.utilityDamage,
      ];
      for (const value of cells) {
        const td = document.createElement('td');
        td.textContent = value;
        row.appendChild(td);
      }
      body.appendChild(row);
    }
  }
}

function renderRounds() {
  const box = $('rounds');
  box.innerHTML = '';
  current.rounds.forEach((round, index) => {
    const el = document.createElement('div');
    el.className = 'round ' + teamClass(round.winner);
    el.textContent = round.number;
    el.title = `${round.winnerName} win — ${round.reasonText || 'round ' + round.number}` +
               ` (${round.scoreCt}:${round.scoreT})`;
    el.addEventListener('click', () => {
      selectedRound = selectedRound === index ? -1 : index;
      renderRounds();
      renderRoundDetail();
    });
    if (index === selectedRound) el.classList.add('selected');
    box.appendChild(el);
  });
  renderRoundDetail();
}

function playerSpan(index, fallback) {
  const span = document.createElement('span');
  if (index < 0 || index >= current.players.length) {
    span.className = 'player-none';
    span.textContent = fallback;
    return span;
  }
  const p = current.players[index];
  span.className = 'player-' + teamClass(p.team);
  span.textContent = p.name || `slot ${p.slot}`;
  return span;
}

function renderRoundDetail() {
  const box = $('round-detail');
  if (selectedRound < 0 || !current.rounds[selectedRound]) {
    box.hidden = true;
    return;
  }
  const round = current.rounds[selectedRound];
  box.hidden = false;
  box.innerHTML = '';

  const title = document.createElement('h4');
  const notes = [];
  if (round.bombPlanted) notes.push('bomb planted');
  if (round.bombDefused) notes.push('defused');
  if (round.bombExploded) notes.push('exploded');
  title.textContent =
    `Round ${round.number} — ${round.winnerName} ` +
    `(${round.reasonText || 'no reason recorded'})` +
    (notes.length ? ' · ' + notes.join(', ') : '');
  box.appendChild(title);

  const list = document.createElement('ul');
  list.className = 'feed';
  if (round.kills.length === 0) {
    const li = document.createElement('li');
    li.textContent = 'No kills recorded in this round.';
    list.appendChild(li);
  }
  for (const kill of round.kills) {
    const li = document.createElement('li');

    const time = document.createElement('span');
    time.className = 'time';
    time.textContent = formatClock(kill.time);
    li.appendChild(time);

    li.appendChild(playerSpan(kill.attacker, 'world'));
    const weapon = document.createElement('span');
    weapon.className = 'weapon';
    weapon.textContent = kill.weapon || 'killed';
    li.appendChild(weapon);
    li.appendChild(playerSpan(kill.victim, 'unknown'));

    if (kill.assister >= 0) {
      const assist = document.createElement('span');
      assist.className = 'weapon';
      assist.textContent = '+';
      li.appendChild(assist);
      li.appendChild(playerSpan(kill.assister, ''));
    }
    for (const [flag, label] of [
      ['headshot', 'hs'], ['wallbang', 'wallbang'], ['noscope', 'noscope'],
      ['throughSmoke', 'smoke'], ['attackerBlind', 'blind'],
    ]) {
      if (!kill[flag]) continue;
      const tag = document.createElement('span');
      tag.className = 'tag';
      tag.textContent = label;
      li.appendChild(tag);
    }
    list.appendChild(li);
  }
  box.appendChild(list);
}

function formatClock(seconds) {
  if (!isFinite(seconds) || seconds < 0) return '0:00';
  const total = Math.floor(seconds);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}

$('lookup').addEventListener('submit', (event) => {
  event.preventDefault();
  const value = $('code').value.trim();
  if (!value) return;
  const url = new URL(window.location);
  url.searchParams.set('code', value);
  window.history.replaceState({}, '', url);
  load(value);
});

document.querySelectorAll('th[data-sort]').forEach((th) => {
  th.addEventListener('click', () => {
    const key = th.dataset.sort;
    if (key === sortKey) {
      sortDesc = !sortDesc;
    } else {
      sortKey = key;
      sortDesc = true;
    }
    if (current) renderScoreboard();
  });
});

// Deep link support: /?code=CSGO-... opens straight away.
const initial = new URL(window.location).searchParams.get('code');
if (initial) {
  $('code').value = initial;
  load(initial);
}
