'use strict';

// The 2D replay: a round played back on a map drawn from where players
// actually walked. Everything here is driven by /api/replay, one round at a
// time, and /api/replay/map, the walkable-area image for the match.
//
// Frame layout, as the server sends it (compact arrays):
//   [tick, [[player, x, y, z, yaw, health, armor, team, flags, weapon]...],
//          [[kind, x, y, z, active, thrower]...],
//          [bombState, x, y, z, carrier, site, defusing]]

const ReplayView = (() => {
  const FLAG_ALIVE = 1, FLAG_BOMB = 2, FLAG_DEFUSING = 4, FLAG_SCOPED = 8;
  const GRENADE_SMOKE = 0, GRENADE_HE = 1, GRENADE_FLASH = 2, GRENADE_MOLOTOV = 3,
        GRENADE_DECOY = 4, GRENADE_FIRE = 5;
  const BOMB_DROPPED = 1, BOMB_PLANTED = 2, BOMB_DEFUSED = 3, BOMB_EXPLODED = 4;
  const SMOKE_RADIUS = 144;   // units; a CS2 smoke is roughly this wide
  const FIRE_RADIUS = 48;

  const state = {
    match: null,        // the match payload from /api/match
    target: null,       // what was typed: share code or path
    round: null,        // the selected round from the match payload
    replay: null,       // /api/replay payload for that round
    map: null,          // Image of /api/replay/map
    mapTarget: null,
    frames: [],
    time: 0,            // seconds since the round's first frame
    duration: 0,
    playing: false,
    speed: 2,
    raf: 0,
    lastStamp: 0,
    byRound: new Map(), // round number -> replay payload
    matchToReplay: new Map(), // match player index -> replay player index
    deaths: [],         // [{tick, x, y, team}] for the selected round
  };

  const el = (id) => document.getElementById(id);

  // ---------------------------------------------------------------- loading

  async function open(match, round, target) {
    state.match = match;
    state.round = round;
    if (state.target !== target) {
      state.target = target;
      state.byRound.clear();
    }
    stop();
    el('replay').hidden = false;
    el('replay-note').textContent = '';

    if (state.mapTarget !== target) {
      state.map = null;
      state.mapTarget = target;
      loadMap(target);
    }

    let payload = state.byRound.get(round.number);
    if (!payload) {
      el('replay-note').textContent =
        'Building the replay for this match… the first round takes a few seconds, ' +
        'the rest open at once.';
      el('replay-play').disabled = true;
      try {
        const response = await fetch(
          `/api/replay?code=${encodeURIComponent(target)}&round=${round.number}`);
        payload = await response.json();
        if (!response.ok) throw new Error(payload.error || `request failed with ${response.status}`);
      } catch (err) {
        el('replay-note').textContent = String(err);
        return;
      }
      state.byRound.set(round.number, payload);
      // The user may have moved on while this loaded.
      if (state.round !== round) return;
    }
    el('replay-play').disabled = false;
    el('replay-note').textContent = '';
    show(payload);
  }

  function loadMap(target) {
    const image = new Image();
    image.onload = () => {
      if (state.mapTarget === target) {
        state.map = image;
        draw();
      }
    };
    image.src = `/api/replay/map?code=${encodeURIComponent(target)}`;
  }

  function close() {
    stop();
    el('replay').hidden = true;
  }

  function show(payload) {
    state.replay = payload;
    state.frames = payload.frames;
    const first = state.frames.length ? state.frames[0][0] : state.round.startTick;
    const last = state.frames.length ? state.frames[state.frames.length - 1][0] : first;
    state.duration = (last - first) / payload.tickRate;
    state.time = 0;

    // Match players and replay players are different lists; Steam ids join
    // them, so kills from the match can be placed on the replay.
    state.matchToReplay.clear();
    const bySteam = new Map();
    payload.players.forEach((p, i) => { if (p.steamId !== '0') bySteam.set(p.steamId, i); });
    state.match.players.forEach((p, i) => {
      if (bySteam.has(p.steamId)) state.matchToReplay.set(i, bySteam.get(p.steamId));
    });
    state.deaths = [];
    for (const kill of state.round.kills) {
      const victim = state.matchToReplay.get(kill.victim);
      if (victim === undefined) continue;
      const frame = frameAtTick(kill.tick);
      const sample = frame ? frame[1].find((s) => s[0] === victim) : null;
      if (sample) state.deaths.push({ tick: kill.tick, x: sample[1], y: sample[2], team: sample[7] });
    }

    const seek = el('replay-seek');
    seek.max = Math.max(0, state.frames.length - 1);
    seek.value = 0;
    renderLegend();
    resize();
    draw();
  }

  // --------------------------------------------------------------- playback

  function play() {
    if (!state.frames.length) return;
    if (state.time >= state.duration) state.time = 0;
    state.playing = true;
    state.lastStamp = 0;
    el('replay-play').textContent = 'Pause';
    state.raf = requestAnimationFrame(tick);
  }

  function stop() {
    state.playing = false;
    if (state.raf) cancelAnimationFrame(state.raf);
    state.raf = 0;
    el('replay-play').textContent = 'Play';
  }

  function tick(stamp) {
    if (!state.playing) return;
    if (state.lastStamp) {
      state.time += ((stamp - state.lastStamp) / 1000) * state.speed;
    }
    state.lastStamp = stamp;
    if (state.time >= state.duration) {
      state.time = state.duration;
      draw();
      stop();
      return;
    }
    draw();
    state.raf = requestAnimationFrame(tick);
  }

  function seekTo(seconds) {
    state.time = Math.max(0, Math.min(state.duration, seconds));
    draw();
  }

  // The frame index just before `time`, by binary search on ticks.
  function frameIndexAt(time) {
    const frames = state.frames;
    if (!frames.length) return -1;
    const tick = frames[0][0] + time * state.replay.tickRate;
    let lo = 0, hi = frames.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (frames[mid][0] <= tick) lo = mid; else hi = mid - 1;
    }
    return lo;
  }

  function frameAtTick(tick) {
    const frames = state.frames;
    if (!frames.length) return null;
    let lo = 0, hi = frames.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (frames[mid][0] <= tick) lo = mid; else hi = mid - 1;
    }
    return frames[lo];
  }

  // ---------------------------------------------------------------- drawing

  const canvas = () => el('replay-canvas');

  function resize() {
    const c = canvas();
    const r = state.replay;
    if (!r) return;
    const box = c.parentElement.getBoundingClientRect();
    const worldW = r.bounds.max[0] - r.bounds.min[0];
    const worldH = r.bounds.max[1] - r.bounds.min[1];
    const width = Math.max(320, Math.floor(box.width));
    const height = Math.round(width * worldH / worldW);
    const dpr = window.devicePixelRatio || 1;
    c.style.height = `${height}px`;
    c.width = Math.round(width * dpr);
    c.height = Math.round(height * dpr);
  }

  // World to canvas: x grows right, y grows up in the game and down on screen.
  function makeTransform() {
    const c = canvas();
    const r = state.replay;
    const sx = c.width / (r.bounds.max[0] - r.bounds.min[0]);
    const sy = c.height / (r.bounds.max[1] - r.bounds.min[1]);
    return {
      x: (wx) => (wx - r.bounds.min[0]) * sx,
      y: (wy) => (r.bounds.max[1] - wy) * sy,
      s: (units) => units * sx,
      px: (n) => n * (window.devicePixelRatio || 1),
    };
  }

  function lerpAngle(a, b, t) {
    let d = ((b - a + 540) % 360) - 180;
    return a + d * t;
  }

  function draw() {
    const c = canvas();
    const ctx = c.getContext('2d');
    const r = state.replay;
    ctx.fillStyle = '#0d1016';
    ctx.fillRect(0, 0, c.width, c.height);
    if (!r) return;
    const T = makeTransform();

    // Map, then bomb sites on top of it.
    if (state.map) {
      const g = r.grid;
      const left = T.x(r.bounds.min[0]);
      const top = T.y(r.bounds.min[1] + g.height * g.cell);
      const w = T.s(g.width * g.cell);
      const h = T.s(g.height * g.cell);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(state.map, left, top, w, h);
    }
    for (const site of r.sites) {
      const x0 = T.x(site.min[0]), x1 = T.x(site.max[0]);
      const y0 = T.y(site.max[1]), y1 = T.y(site.min[1]);
      ctx.fillStyle = 'rgba(248, 113, 113, 0.18)';
      ctx.strokeStyle = 'rgba(248, 113, 113, 0.7)';
      ctx.lineWidth = T.px(1);
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      if (site.letter) {
        ctx.font = `bold ${T.px(18)}px system-ui, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.lineWidth = T.px(3);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.strokeText(site.letter, (x0 + x1) / 2, (y0 + y1) / 2);
        ctx.fillStyle = 'rgba(255, 190, 190, 0.95)';
        ctx.fillText(site.letter, (x0 + x1) / 2, (y0 + y1) / 2);
      }
    }

    const frames = state.frames;
    if (!frames.length) return;
    const i = frameIndexAt(state.time);
    const a = frames[i];
    const b = frames[Math.min(i + 1, frames.length - 1)];
    const tickNow = frames[0][0] + state.time * r.tickRate;
    const span = b[0] - a[0];
    const t = span > 0 ? Math.max(0, Math.min(1, (tickNow - a[0]) / span)) : 0;

    // Deaths so far this round.
    for (const death of state.deaths) {
      if (death.tick > tickNow) continue;
      drawCross(ctx, T.x(death.x), T.y(death.y), T.px(5), teamColor(death.team, 0.55));
    }

    // Grenades: from the earlier frame; they are not interpolated.
    for (const g of a[2]) {
      const x = T.x(g[1]), y = T.y(g[2]);
      const kind = g[0], active = g[4];
      if (kind === GRENADE_SMOKE && active) {
        ctx.fillStyle = 'rgba(200, 205, 215, 0.35)';
        ctx.beginPath();
        ctx.arc(x, y, T.s(SMOKE_RADIUS), 0, Math.PI * 2);
        ctx.fill();
      } else if (kind === GRENADE_FIRE) {
        ctx.fillStyle = active ? 'rgba(255, 140, 40, 0.45)' : 'rgba(255, 140, 40, 0.15)';
        ctx.beginPath();
        ctx.arc(x, y, T.s(FIRE_RADIUS), 0, Math.PI * 2);
        ctx.fill();
      } else {
        const color = kind === GRENADE_MOLOTOV ? '#ff8c28' : kind === GRENADE_FLASH ? '#fff2a8'
          : kind === GRENADE_HE ? '#c0c8d4' : kind === GRENADE_DECOY ? '#9aa4b8' : '#d0d6e0';
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, T.px(3), 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // The bomb, when it is not in someone's hands.
    const bomb = a[3];
    if (bomb[0] === BOMB_DROPPED || bomb[0] === BOMB_PLANTED || bomb[0] === BOMB_DEFUSED) {
      const x = T.x(bomb[1]), y = T.y(bomb[2]);
      const planted = bomb[0] === BOMB_PLANTED;
      const blink = planted && Math.floor(state.time * 2) % 2 === 0;
      ctx.fillStyle = bomb[0] === BOMB_DEFUSED ? '#4ade80' : blink ? '#ff5050' : '#c03030';
      ctx.beginPath();
      ctx.arc(x, y, T.px(planted ? 6 : 4), 0, Math.PI * 2);
      ctx.fill();
      if (planted) {
        ctx.fillStyle = '#fff';
        ctx.font = `bold ${T.px(8)}px system-ui, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('C4', x, y);
      }
    }

    // Players: interpolated between the two frames around the clock.
    const next = new Map(b[1].map((s) => [s[0], s]));
    for (const s of a[1]) {
      const n = next.get(s[0]) || s;
      const alive = (s[8] & FLAG_ALIVE) !== 0;
      const x = T.x(s[1] + (n[1] - s[1]) * t);
      const y = T.y(s[2] + (n[2] - s[2]) * t);
      const yaw = lerpAngle(s[4], n[4], t);
      const player = r.players[s[0]];
      drawPlayer(ctx, T, x, y, yaw, s, alive, player ? player.name : '');
    }

    // Clock and slider.
    const clock = el('replay-clock');
    clock.textContent = `${formatClock(state.time)} / ${formatClock(state.duration)}`;
    const seek = el('replay-seek');
    if (document.activeElement !== seek) seek.value = i;
    updateLegend(a);
    updateFeed(tickNow);
  }

  function teamColor(team, alpha) {
    const rgb = team === 3 ? '107, 157, 234' : team === 2 ? '224, 176, 74' : '160, 160, 160';
    return `rgba(${rgb}, ${alpha === undefined ? 1 : alpha})`;
  }

  function drawCross(ctx, x, y, r, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1, r / 3);
    ctx.beginPath();
    ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r);
    ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r);
    ctx.stroke();
  }

  function drawPlayer(ctx, T, x, y, yaw, s, alive, name) {
    const team = s[7];
    const radius = T.px(7);
    if (!alive) {
      drawCross(ctx, x, y, T.px(5), teamColor(team, 0.6));
      return;
    }
    // View direction: Source yaw is counter-clockwise from +x, and the
    // canvas has y pointing down.
    const rad = yaw * Math.PI / 180;
    const dx = Math.cos(rad), dy = -Math.sin(rad);

    // A soft view cone, then the body.
    ctx.fillStyle = teamColor(team, 0.15);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.arc(x, y, radius * 4.5, Math.atan2(dy, dx) - 0.55, Math.atan2(dy, dx) + 0.55);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = teamColor(team);
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.75)';
    ctx.lineWidth = T.px(1.5);
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = T.px(2);
    ctx.beginPath();
    ctx.moveTo(x + dx * radius * 0.4, y + dy * radius * 0.4);
    ctx.lineTo(x + dx * radius * 1.7, y + dy * radius * 1.7);
    ctx.stroke();

    // Health as an arc around the body.
    const health = Math.max(0, Math.min(100, s[5]));
    ctx.strokeStyle = health > 50 ? '#4ade80' : health > 25 ? '#facc15' : '#f87171';
    ctx.lineWidth = T.px(2);
    ctx.beginPath();
    ctx.arc(x, y, radius + T.px(2.5), -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * health / 100);
    ctx.stroke();

    if (s[8] & FLAG_BOMB) {
      ctx.fillStyle = '#ff5050';
      ctx.beginPath();
      ctx.arc(x - radius * 0.9, y - radius * 0.9, T.px(3.5), 0, Math.PI * 2);
      ctx.fill();
    }
    if (s[8] & FLAG_DEFUSING) {
      ctx.strokeStyle = '#4ade80';
      ctx.lineWidth = T.px(2);
      ctx.setLineDash([T.px(3), T.px(3)]);
      ctx.beginPath();
      ctx.arc(x, y, radius + T.px(6), 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.fillStyle = 'rgba(230, 233, 239, 0.9)';
    ctx.font = `${T.px(10)}px system-ui, sans-serif`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(name, x + radius + T.px(6), y);
  }

  // ------------------------------------------------------------- side panel

  function renderLegend() {
    const box = el('replay-players');
    box.innerHTML = '';
    const r = state.replay;
    r.players.forEach((p, i) => {
      const row = document.createElement('div');
      row.className = 'replay-player';
      row.dataset.index = i;
      row.innerHTML =
        `<span class="dot"></span><span class="name"></span>` +
        `<span class="weapon"></span><span class="hp"></span>`;
      row.querySelector('.name').textContent = p.name || `player ${p.id}`;
      box.appendChild(row);
    });
  }

  function updateLegend(frame) {
    const r = state.replay;
    const rows = el('replay-players').children;
    const samples = new Map(frame[1].map((s) => [s[0], s]));
    for (const row of rows) {
      const s = samples.get(Number(row.dataset.index));
      const alive = s && (s[8] & FLAG_ALIVE);
      row.classList.toggle('dead', !alive);
      row.classList.toggle('ct', !!s && s[7] === 3);
      row.classList.toggle('t', !!s && s[7] === 2);
      row.querySelector('.hp').textContent = s ? (alive ? `${s[5]}` : '') : '';
      const weapon = s && s[9] >= 0 ? r.weapons[s[9]] : '';
      row.querySelector('.weapon').textContent =
        alive ? weapon + ((s[8] & FLAG_BOMB) ? ' · C4' : '') : '';
    }
  }

  function updateFeed(tickNow) {
    const rows = el('replay-feed').children;
    for (const row of rows) {
      row.classList.toggle('done', Number(row.dataset.tick) <= tickNow);
    }
  }

  function renderFeed() {
    const box = el('replay-feed');
    box.innerHTML = '';
    if (!state.round) return;
    for (const kill of state.round.kills) {
      const row = document.createElement('div');
      row.className = 'replay-kill';
      row.dataset.tick = kill.tick;
      const attacker = state.match.players[kill.attacker];
      const victim = state.match.players[kill.victim];
      row.innerHTML =
        `<span class="time"></span><span class="who a"></span>` +
        `<span class="weapon"></span><span class="who v"></span>`;
      row.querySelector('.time').textContent = formatClock(kill.time);
      const a = row.querySelector('.a');
      a.textContent = attacker ? attacker.name : 'world';
      a.classList.add(attacker && attacker.team === 3 ? 'ct' : 't');
      row.querySelector('.weapon').textContent = kill.weapon + (kill.headshot ? ' (hs)' : '');
      const v = row.querySelector('.v');
      v.textContent = victim ? victim.name : '?';
      v.classList.add(kill.victimTeam === 3 ? 'ct' : 't');
      row.addEventListener('click', () => {
        const first = state.frames.length ? state.frames[0][0] : kill.tick;
        seekTo((kill.tick - first) / state.replay.tickRate - 3);
      });
      box.appendChild(row);
    }
  }

  function formatClock(seconds) {
    if (!isFinite(seconds) || seconds < 0) return '0:00';
    const total = Math.floor(seconds);
    return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
  }

  // ------------------------------------------------------------------ wiring

  function wire() {
    el('replay-play').addEventListener('click', () => (state.playing ? stop() : play()));
    el('replay-speed').addEventListener('change', (e) => { state.speed = Number(e.target.value); });
    el('replay-seek').addEventListener('input', (e) => {
      const index = Number(e.target.value);
      const frames = state.frames;
      if (!frames.length) return;
      seekTo((frames[index][0] - frames[0][0]) / state.replay.tickRate);
    });
    window.addEventListener('resize', () => { if (state.replay) { resize(); draw(); } });
    document.addEventListener('keydown', (e) => {
      if (el('replay').hidden || e.target.tagName === 'INPUT') return;
      if (e.code === 'Space') {
        e.preventDefault();
        state.playing ? stop() : play();
      } else if (e.code === 'ArrowLeft') {
        seekTo(state.time - (e.shiftKey ? 10 : 2));
      } else if (e.code === 'ArrowRight') {
        seekTo(state.time + (e.shiftKey ? 10 : 2));
      }
    });
  }

  document.addEventListener('DOMContentLoaded', wire);
  if (document.readyState !== 'loading') wire();

  return {
    open: (match, round, target) => { open(match, round, target); renderFeed(); },
    close,
  };
})();
