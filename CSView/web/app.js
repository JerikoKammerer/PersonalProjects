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
  box.appendChild(roundSeekCommand(round));
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

// --- Steam sign-in -------------------------------------------------------
//
// Only a QR challenge reaches this page. The phone talks to Steam directly and
// the helper keeps the resulting token, so nothing secret is handled here.

let steamPoll = null;

async function steamStatus() {
  try {
    const response = await fetch('/api/steam/status');
    const s = await response.json();
    const box = $('steam');
    box.hidden = false;

    if (!s.helperConfigured) {
      $('steam-state').textContent = 'Steam helper not configured — matches you have downloaded in CS2 still work.';
      $('steam-state').title = s.detail || '';
      $('steam-signin').hidden = true;
      $('steam-signout').hidden = true;
      return;
    }
    if (s.signedIn) {
      $('steam-state').innerHTML = 'Steam: <span class="on">signed in' +
        (s.account ? ' as ' + escapeHtml(s.account) : '') + '</span>';
      $('steam-signin').hidden = true;
      $('steam-signout').hidden = false;
      $('steam-qr').hidden = true;
    } else {
      $('steam-state').textContent = s.detail ||
        'Steam: not signed in — sign in to fetch matches you have not downloaded.';
      $('steam-signin').hidden = false;
      $('steam-signout').hidden = true;
    }
  } catch (err) {
    // The server is what serves this page, so a failure here is not worth
    // shouting about; the panel just stays hidden.
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

async function steamSignIn() {
  $('steam-signin').disabled = true;
  try {
    const response = await fetch('/api/steam/login', { method: 'POST' });
    const body = await response.json();
    if (!response.ok) {
      $('steam-state').textContent = body.error || 'could not start sign-in';
      return;
    }
    $('steam-qr').hidden = false;
    $('steam-qr-progress').textContent = 'Asking Steam for a code…';
    if (steamPoll) clearInterval(steamPoll);
    steamPoll = setInterval(pollSteamLogin, 1000);
  } finally {
    $('steam-signin').disabled = false;
  }
}

async function pollSteamLogin() {
  let s;
  try {
    s = await (await fetch('/api/steam/login/status')).json();
  } catch (err) {
    return;
  }

  // A data: URL is the only thing accepted here, so a helper cannot inject
  // markup into the page through this field.
  if (s.qrPng && s.qrPng.startsWith('data:image/png;base64,')) {
    $('steam-qr-img').src = s.qrPng;
  }
  if (s.qrUrl) $('steam-qr-link').href = s.qrUrl;

  const progress = $('steam-qr-progress');
  if (s.state === 'waiting') {
    progress.textContent = 'Waiting for a phone…';
  } else if (s.state === 'scanned') {
    progress.textContent = 'Scanned — approve the sign-in on your phone.';
  } else if (s.state === 'done') {
    clearInterval(steamPoll);
    steamPoll = null;
    $('steam-qr').hidden = true;
    steamStatus();
  } else if (s.state === 'error') {
    clearInterval(steamPoll);
    steamPoll = null;
    progress.textContent = s.message || 'sign-in failed';
  }
}

async function steamSignOut() {
  if (steamPoll) { clearInterval(steamPoll); steamPoll = null; }
  await fetch('/api/steam/logout', { method: 'POST' });
  $('steam-qr').hidden = true;
  steamStatus();
}

$('steam-signin').addEventListener('click', steamSignIn);
$('steam-signout').addEventListener('click', steamSignOut);
steamStatus();

// --- Your matches --------------------------------------------------------
//
// Demos CS2 has already downloaded. Listing them needs no Steam and no share
// code: the folders are known, and each demo's first frame names its map.

function formatBytes(n) {
  if (!n) return '';
  const mb = n / (1024 * 1024);
  return mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : Math.round(mb) + ' MB';
}

function formatWhen(unix) {
  if (!unix) return '';
  const then = new Date(unix * 1000);
  const days = Math.floor((Date.now() - then.getTime()) / 86400000);
  if (days <= 0) return 'today';
  if (days === 1) return 'yesterday';
  if (days < 30) return days + ' days ago';
  return then.toLocaleDateString();
}

async function loadRecent() {
  let payload;
  try {
    payload = await (await fetch('/api/matches')).json();
  } catch (err) {
    return;
  }
  const box = $('recent');
  const list = $('recent-list');
  list.innerHTML = '';
  box.hidden = false;

  if (!payload.matches || payload.matches.length === 0) {
    const empty = document.createElement('p');
    empty.className = 'recent-empty';
    empty.textContent =
      'No demos downloaded yet. In CS2, open Watch → Your Matches and click ' +
      'Download on a match; it will show up here.';
    list.appendChild(empty);
    return;
  }

  for (const m of payload.matches) {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'recent-item';

    const map = document.createElement('span');
    map.className = 'map';
    map.textContent = m.map || 'unknown map';
    const when = document.createElement('span');
    when.className = 'when';
    when.textContent = formatWhen(m.modified);
    const size = document.createElement('span');
    size.className = 'size';
    size.textContent = formatBytes(m.sizeBytes);

    row.appendChild(map);
    row.appendChild(when);
    row.appendChild(size);
    row.title = m.path;
    row.addEventListener('click', () => {
      $('code').value = m.path;
      load(m.path);
    });
    list.appendChild(row);
  }
}

loadRecent();

// Match history from the game coordinator: everything played recently, not
// just what happens to be on disk. Needs the signed-in helper, so it is behind
// a button rather than loaded with the page.
async function loadRemote() {
  const button = $('recent-fetch');
  const note = $('recent-fetch-note');
  button.disabled = true;
  note.textContent = 'Asking Steam…';
  try {
    const response = await fetch('/api/matches/remote');
    const payload = await response.json();
    if (!response.ok) {
      note.textContent = payload.error || 'could not reach the game coordinator';
      return;
    }
    const list = $('remote-list');
    list.innerHTML = '';
    note.textContent = `${payload.matches.length} from Steam — click to fetch and open`;
    for (const m of payload.matches) {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'recent-item' + (m.expired ? ' expired' : '');
      const code = document.createElement('span');
      code.className = 'map';
      code.textContent = m.shareCode;
      const when = document.createElement('span');
      when.className = 'when';
      when.textContent = m.matchTime ? formatWhen(m.matchTime) : '';
      const state = document.createElement('span');
      state.className = 'size';
      state.textContent = m.expired ? 'expired' : 'available';
      row.appendChild(code);
      row.appendChild(when);
      row.appendChild(state);
      if (!m.expired) {
        row.addEventListener('click', () => {
          $('code').value = m.shareCode;
          load(m.shareCode);
        });
      } else {
        row.title = 'Valve keeps match demos for about 30 days';
      }
      list.appendChild(row);
    }
  } catch (err) {
    note.textContent = String(err);
  } finally {
    button.disabled = false;
  }
}

$('recent-fetch').addEventListener('click', loadRemote);

// --- Playback in CS2 -----------------------------------------------------
//
// A demo holds no video, only per-tick state, so watching it means letting the
// game render it. This hands the file to CS2 and offers the console command
// that seeks to a given round.

async function watchInCs2() {
  const path = current && current.match ? current.match.demoSource : '';
  if (!path) return;
  const note = $('watch-note');
  const button = $('watch-match');
  button.disabled = true;
  note.textContent = 'Handing the demo to CS2…';
  try {
    const response = await fetch('/api/watch?demo=' + encodeURIComponent(path),
                                 { method: 'POST' });
    const body = await response.json();
    note.textContent = response.ok
      ? 'CS2 is opening the demo. Use the round buttons below for the tick to seek to.'
      : (body.error || 'could not start playback');
  } catch (err) {
    note.textContent = String(err);
  } finally {
    button.disabled = false;
  }
}

$('watch-match').addEventListener('click', watchInCs2);

// Seeking is a console command rather than a launch argument: playdemo loads
// asynchronously, so a tick passed at startup is applied before the demo is
// ready. Clicking copies it.
function roundSeekCommand(round) {
  const wrap = document.createElement('div');
  wrap.className = 'round-watch';

  const label = document.createElement('span');
  label.className = 'muted';
  label.textContent = 'Jump to this round in CS2:';

  const cmd = document.createElement('button');
  cmd.type = 'button';
  cmd.className = 'tick-cmd';
  const text = `demo_gototick ${round.startTick}`;
  cmd.textContent = text;
  cmd.title = 'Copy, then paste into the CS2 console (~)';
  cmd.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(text);
      cmd.textContent = 'copied — paste into the CS2 console';
      setTimeout(() => { cmd.textContent = text; }, 1600);
    } catch (err) {
      cmd.textContent = text;
    }
  });

  wrap.appendChild(label);
  wrap.appendChild(cmd);
  return wrap;
}
