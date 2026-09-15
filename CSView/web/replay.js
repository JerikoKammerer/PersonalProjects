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

  // The server names weapons after their entity classes; these are the names
  // people use. Anything not listed is shown upper-cased.
  const WEAPON_NAMES = {
    ak47: 'AK-47', m4a1: 'M4A4', m4a1silencer: 'M4A1-S', awp: 'AWP', ssg08: 'Scout',
    galilar: 'Galil', famas: 'FAMAS', aug: 'AUG', sg556: 'SG 553', scar20: 'SCAR-20',
    g3sg1: 'G3SG1', m249: 'M249', negev: 'Negev', mac10: 'MAC-10', mp9: 'MP9', mp7: 'MP7',
    mp5sd: 'MP5-SD', ump45: 'UMP-45', p90: 'P90', bizon: 'PP-Bizon', nova: 'Nova',
    xm1014: 'XM1014', mag7: 'MAG-7', sawedoff: 'Sawed-Off', glock: 'Glock', hkp2000: 'P2000',
    uspsilencer: 'USP-S', p250: 'P250', fiveseven: 'Five-SeveN', tec9: 'Tec-9', cz75a: 'CZ75',
    deagle: 'Deagle', revolver: 'R8', elite: 'Dual Berettas', knife: 'Knife', taser: 'Zeus',
    c4: 'C4', hegrenade: 'HE', flashbang: 'Flash', smokegrenade: 'Smoke',
    molotovgrenade: 'Molotov', incendiarygrenade: 'Incendiary', decoygrenade: 'Decoy',
  };
  const weaponName = (key) => WEAPON_NAMES[key] || (key ? key.toUpperCase() : '');

  // Radar images, by map. Each is placed in the world by a top-left corner
  // (posX, posY) and a scale in units per pixel of a 1024 wide image - the
  // convention the game's own overview files use. Those numbers are only a
  // starting point: whenever a match is opened, the placement is refined
  // against where the players actually walked (see fitRadar), which also
  // handles an image that is a crop, a thumbnail, or has no numbers at all.
  const RADARS = {
    // A crop with no overview numbers; these were fitted from a match.
    de_ancient: { file: 'de_ancient.jpg', posX: -2477, posY: 1933, scale: 3.94 },
    de_anubis: { file: 'de_anubis.jpg', posX: -2796, posY: 3328, scale: 5.22 },
    de_cache: { file: 'de_cache.jpg', posX: -2000, posY: 3250, scale: 5.5 },
    de_dust2: { file: 'de_dust2.jpg', posX: -2476, posY: 3239, scale: 4.4 },
    de_inferno: { file: 'de_inferno.jpg', posX: -2087, posY: 3870, scale: 4.9 },
    de_mirage: { file: 'de_mirage.jpg', posX: -3230, posY: 1713, scale: 5.0 },
    // Upper floor only, cropped to the drawing. Anything below `lowerBelow`
    // is the floor beneath, which the drawing does not show: it is left out
    // of the fit, and players down there are drawn hollow with a marker.
    de_nuke: { file: 'de_nuke.jpg', lowerBelow: -495 },
  };
  const MIN_FIT = 0.9;  // share of walked cells that must land on the drawing

  const state = {
    match: null,        // the match payload from /api/match
    target: null,       // what was typed: share code or path
    round: null,        // the selected round from the match payload
    replay: null,       // /api/replay payload for that round
    map: null,          // Image of /api/replay/map: where players walked
    radarEntry: null,   // the RADARS entry in use, for what it says about floors
    mapTarget: null,
    radar: null,        // Image of the map's radar, once it is placed
    radarFit: null,     // {posX, posY, s (units per pixel), score}
    radarNote: '',
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
    clockOffset: 0,     // seconds of freeze time before the round's clock starts
  };

  const el = (id) => document.getElementById(id);

  // ---------------------------------------------------------------- loading

  async function open(match, round, target) {
    state.match = match;
    state.round = round;
    if (state.target !== target) {
      state.target = target;
      state.byRound.clear();
      state.replay = null;
      state.frames = [];
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
    state.radar = null;
    state.radarFit = null;
    state.radarEntry = null;
    state.radarNote = '';
    state.radarTried = false;
    const image = new Image();
    image.onload = () => {
      if (state.mapTarget !== target) return;
      state.map = image;
      draw();
    };
    image.src = `/api/replay/map?code=${encodeURIComponent(target)}`;
  }

  // Loads the radar image for the current map, if there is one, and works
  // out where it sits in the world from the walked footprint. The footprint
  // is asked for here rather than with the map, because a drawing of one
  // floor of a two-floor map must only be fitted to that floor.
  function placeRadar(target) {
    const r = state.replay;
    if (!r || state.radarTried) return;
    state.radarTried = true;
    const entry = RADARS[r.map];
    if (!entry) {
      state.radarNote = 'no radar image for this map; showing where players walked';
      updateNote();
      return;
    }
    let footprint = `/api/replay/map?code=${encodeURIComponent(target)}&raw=1`;
    if (entry.lowerBelow !== undefined) footprint += `&zmin=${entry.lowerBelow}`;
    const walked = new Image();
    walked.onload = () => {
      if (state.mapTarget !== target) return;
      const image = new Image();
      image.onload = () => fitAndPlace(target, image, entry, walked, r);
      image.onerror = () => { state.radarNote = 'radar image missing'; updateNote(); };
      image.src = `/maps/${entry.file}`;
    };
    walked.src = footprint;
  }

  function fitAndPlace(target, image, entry, walked, r) {
    if (state.mapTarget !== target) return;
    const fit = fitRadar(image, entry, walked, r);
    state.lastFit = fit;
    if (fit && fit.score >= MIN_FIT) {
      state.radar = keyOutBackground(image);
      state.radarFit = fit;
      state.radarEntry = entry;
      state.radarNote = `radar placed from play (${Math.round(fit.score * 100)}% of walked ground on the drawing)`;
      console.log(`radar fit for ${r.map}: posX=${fit.posX.toFixed(1)} posY=${fit.posY.toFixed(1)} ` +
                  `scale@1024=${(fit.s * image.width / 1024).toFixed(4)} score=${fit.score.toFixed(4)}`);
    } else {
      state.radarNote = 'the radar image could not be matched to this match; showing where players walked';
    }
    updateNote();
    resize();
    draw();
  }

  function updateNote() {
    const note = el('replay-note');
    if (!note.textContent || note.textContent.startsWith('radar') ||
        note.textContent.startsWith('no radar') || note.textContent.startsWith('the radar')) {
      note.textContent = state.radarNote;
    }
  }

  // The radar drawings come on white; the page is dark. Near-white pixels
  // become transparent so the drawing sits on the page's own ground, with a
  // soft ramp so anti-aliased edges do not turn into a halo.
  function keyOutBackground(image) {
    const c = document.createElement('canvas');
    c.width = image.width;
    c.height = image.height;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(image, 0, 0);
    const data = ctx.getImageData(0, 0, c.width, c.height);
    const px = data.data;
    for (let i = 0; i < px.length; i += 4) {
      const darkest = Math.min(px[i], px[i + 1], px[i + 2]);
      if (darkest >= 215) {
        const alpha = Math.max(0, Math.min(1, (240 - darkest) / 25));
        px[i + 3] = Math.round(px[i + 3] * alpha);
      }
    }
    ctx.putImageData(data, 0, 0);
    return c;
  }

  // ------------------------------------------------------------ radar fitting
  //
  // The walked-area image is a grid of cells over known world bounds; the
  // radar is a drawing with a white or transparent background. The placement
  // that puts the most walked cells onto drawn pixels is the right one, and a
  // pattern search from a decent first guess finds it in well under a second.

  function radarMask(image, entry) {
    const limit = 384;
    const k = Math.min(1, limit / Math.max(image.width, image.height));
    const w = Math.max(1, Math.round(image.width * k));
    const h = Math.max(1, Math.round(image.height * k));
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(image, 0, 0, w, h);
    const px = ctx.getImageData(0, 0, w, h).data;
    const mask = new Uint8Array(w * h);
    const ignore = entry.ignore ? entry.ignore.map((f, i) => f * (i % 2 ? h : w)) : null;
    for (let y = 0; y < h; ++y) {
      for (let x = 0; x < w; ++x) {
        if (ignore && x >= ignore[0] && x < ignore[2] && y >= ignore[1] && y < ignore[3]) continue;
        const i = (y * w + x) * 4;
        const dark = Math.min(px[i], px[i + 1], px[i + 2]) < 225;
        mask[y * w + x] = px[i + 3] > 40 && dark ? 1 : 0;
      }
    }
    return { mask, w, h, k };
  }

  function walkedCells(mapImage, r) {
    const c = document.createElement('canvas');
    c.width = mapImage.width;
    c.height = mapImage.height;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(mapImage, 0, 0);
    const px = ctx.getImageData(0, 0, c.width, c.height).data;
    const xs = [], ys = [];
    const g = r.grid;
    for (let row = 0; row < c.height; ++row) {
      for (let col = 0; col < c.width; ++col) {
        if (px[(row * c.width + col) * 4 + 3] === 0) continue;
        xs.push(r.bounds.min[0] + (col + 0.5) * g.cell);
        ys.push(r.bounds.min[1] + (c.height - 1 - row + 0.5) * g.cell);
      }
    }
    return { xs, ys };
  }

  function fitRadar(image, entry, mapImage, r) {
    const m = radarMask(image, entry);
    const cells = walkedCells(mapImage, r);
    const n = cells.xs.length;
    if (n < 50) return null;

    const score = (posX, posY, s) => {
      let hit = 0;
      const sx = m.k / s;
      for (let i = 0; i < n; ++i) {
        const px = ((cells.xs[i] - posX) * sx) | 0;
        const py = ((posY - cells.ys[i]) * sx) | 0;
        if (px >= 0 && py >= 0 && px < m.w && py < m.h && m.mask[py * m.w + px]) ++hit;
      }
      return hit / n;
    };

    // First guess: the walked area's extent against the drawing's extent.
    let ix0 = m.w, ix1 = 0, iy0 = m.h, iy1 = 0;
    for (let y = 0; y < m.h; ++y) {
      for (let x = 0; x < m.w; ++x) {
        if (!m.mask[y * m.w + x]) continue;
        if (x < ix0) ix0 = x;
        if (x > ix1) ix1 = x;
        if (y < iy0) iy0 = y;
        if (y > iy1) iy1 = y;
      }
    }
    const sortedX = Array.from(cells.xs).sort((a, b) => a - b);
    const sortedY = Array.from(cells.ys).sort((a, b) => a - b);
    const q = (arr, f) => arr[Math.min(arr.length - 1, Math.floor(f * arr.length))];
    const wx0 = q(sortedX, 0.002), wx1 = q(sortedX, 0.998);
    const wy0 = q(sortedY, 0.002), wy1 = q(sortedY, 0.998);
    const sGuess = ((wx1 - wx0) / ((ix1 - ix0) / m.k) + (wy1 - wy0) / ((iy1 - iy0) / m.k)) / 2;
    const starts = [[wx0 - (ix0 / m.k) * sGuess, wy1 + (iy0 / m.k) * sGuess, sGuess]];
    if (entry.scale) starts.push([entry.posX, entry.posY, entry.scale * 1024 / image.width]);

    const refine = (start) => {
      let best = start;
      let bestScore = score(...best);
      for (const [dp, ds] of [[40, 0.03], [10, 0.01], [2, 0.002], [0.5, 0.0005]]) {
        let improved = true;
        while (improved) {
          improved = false;
          for (let dx = -1; dx <= 1; ++dx) {
            for (let dy = -1; dy <= 1; ++dy) {
              for (let dk = -1; dk <= 1; ++dk) {
                const cand = [best[0] + dx * dp, best[1] + dy * dp, best[2] * (1 + dk * ds)];
                const sc = score(...cand);
                if (sc > bestScore + 1e-9) {
                  best = cand;
                  bestScore = sc;
                  improved = true;
                }
              }
            }
          }
        }
      }
      return { posX: best[0], posY: best[1], s: best[2], score: bestScore };
    };
    let result = null;
    for (const start of starts) {
      const fit = refine(start);
      if (!result || fit.score > result.score) result = fit;
    }
    return result;
  }

  function close() {
    stop();
    el('replay').hidden = true;
  }

  // The world rectangle on view: the radar when it is placed, else the game's
  // own radar bounds.
  function viewBounds() {
    const r = state.replay;
    if (state.radar && state.radarFit) {
      const f = state.radarFit;
      return {
        min: [f.posX, f.posY - state.radar.height * f.s],
        max: [f.posX + state.radar.width * f.s, f.posY],
      };
    }
    return r.bounds;
  }

  function show(payload) {
    state.replay = payload;
    state.frames = payload.frames;
    placeRadar(state.target);
    const first = state.frames.length ? state.frames[0][0] : state.round.startTick;
    const last = state.frames.length ? state.frames[state.frames.length - 1][0] : first;
    state.duration = (last - first) / payload.tickRate;
    state.time = 0;
    // The kill feed's clock starts when freeze time ends, so the replay's
    // does too; before that it counts down.
    state.clockOffset = state.round.liveTick > first
      ? (state.round.liveTick - first) / payload.tickRate : 0;

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
      if (!sample) continue;
      const death = { tick: kill.tick, x: sample[1], y: sample[2], team: sample[7], from: null };
      const attacker = state.matchToReplay.get(kill.attacker);
      const shooter = attacker !== undefined ? frame[1].find((s) => s[0] === attacker) : null;
      if (shooter && shooter !== sample) death.from = { x: shooter[1], y: shooter[2] };
      state.deaths.push(death);
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
    const view = viewBounds();
    const worldW = view.max[0] - view.min[0];
    const worldH = view.max[1] - view.min[1];
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
    const view = viewBounds();
    const sx = c.width / (view.max[0] - view.min[0]);
    const sy = c.height / (view.max[1] - view.min[1]);
    return {
      x: (wx) => (wx - view.min[0]) * sx,
      y: (wy) => (view.max[1] - wy) * sy,
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

    // Map, then bomb sites on top of it. The radar drawing when it has been
    // placed; otherwise the ground the players revealed.
    if (state.radar && state.radarFit) {
      const f = state.radarFit;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(state.radar, T.x(f.posX), T.y(f.posY),
                    T.s(state.radar.width * f.s), T.s(state.radar.height * f.s));
    } else if (state.map) {
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

    // Deaths so far this round, and for a moment after each, a line from
    // whoever did it.
    for (const death of state.deaths) {
      if (death.tick > tickNow) continue;
      drawCross(ctx, T.x(death.x), T.y(death.y), T.px(5), teamColor(death.team, 0.55));
      const age = (tickNow - death.tick) / r.tickRate;
      if (death.from && age < 2.5) {
        ctx.strokeStyle = `rgba(248, 113, 113, ${0.9 * (1 - age / 2.5)})`;
        ctx.lineWidth = T.px(1.5);
        ctx.setLineDash([T.px(4), T.px(3)]);
        ctx.beginPath();
        ctx.moveTo(T.x(death.from.x), T.y(death.from.y));
        ctx.lineTo(T.x(death.x), T.y(death.y));
        ctx.stroke();
        ctx.setLineDash([]);
      }
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
      const below = state.radarEntry && state.radarEntry.lowerBelow !== undefined &&
                    s[3] < state.radarEntry.lowerBelow;
      drawPlayer(ctx, T, x, y, yaw, s, alive, player ? player.name : '', below);
    }

    // Clock and slider.
    const clock = el('replay-clock');
    const shown = state.time - state.clockOffset;
    clock.textContent = (shown < 0 ? `freeze ${formatClock(-shown)}` : formatClock(shown)) +
      ` / ${formatClock(state.duration - state.clockOffset)}`;
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

  function drawPlayer(ctx, T, x, y, yaw, s, alive, name, below) {
    const team = s[7];
    const radius = T.px(7);
    if (!alive) {
      drawCross(ctx, x, y, T.px(5), teamColor(team, 0.6));
      return;
    }
    if (below) {
      // On the floor beneath the one drawn: a hollow body, a "down" mark
      // above it, and no view cone, so it cannot be mistaken for someone
      // standing on the drawing.
      const rad = yaw * Math.PI / 180;
      const dx = Math.cos(rad), dy = -Math.sin(rad);
      ctx.fillStyle = 'rgba(13, 16, 22, 0.7)';
      ctx.strokeStyle = teamColor(team, 0.95);
      ctx.lineWidth = T.px(2);
      ctx.setLineDash([T.px(3), T.px(2)]);
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(x + dx * radius * 0.4, y + dy * radius * 0.4);
      ctx.lineTo(x + dx * radius * 1.7, y + dy * radius * 1.7);
      ctx.stroke();
      ctx.fillStyle = teamColor(team, 0.95);
      ctx.font = `bold ${T.px(9)}px system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('\u25BC', x, y - radius - T.px(6));
      ctx.fillStyle = 'rgba(230, 233, 239, 0.75)';
      ctx.font = `${T.px(10)}px system-ui, sans-serif`;
      ctx.textAlign = 'left';
      ctx.fillText(name, x + radius + T.px(6), y);
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
      const weapon = s && s[9] >= 0 ? weaponName(r.weapons[s[9]]) : '';
      row.querySelector('.weapon').textContent =
        alive ? weapon + ((s[8] & FLAG_BOMB) && weapon !== 'C4' ? ' · C4' : '') : '';
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
      // Sides as they were in this round, not where each player ended up.
      const a = row.querySelector('.a');
      a.textContent = attacker ? attacker.name : 'world';
      a.classList.add(sideClass(kill.attackerTeam));
      row.querySelector('.weapon').textContent = kill.weapon + (kill.headshot ? ' (hs)' : '');
      const v = row.querySelector('.v');
      v.textContent = victim ? victim.name : '?';
      v.classList.add(sideClass(kill.victimTeam));
      row.addEventListener('click', () => {
        const first = state.frames.length ? state.frames[0][0] : kill.tick;
        seekTo((kill.tick - first) / state.replay.tickRate - 3);
      });
      box.appendChild(row);
    }
  }

  function sideClass(team) {
    return team === 3 ? 'ct' : team === 2 ? 't' : 'none';
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
    // For poking at the radar placement from the console.
    lastFit: () => state.lastFit,
  };
})();
