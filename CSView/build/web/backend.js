'use strict';

// Where the page gets its data, which is one of three things.
//
//   live     - served by cs2mv itself, which parses demos on demand. Every
//              feature works: any share code, Steam sign-in, Watch in CS2.
//   static   - served as plain files by a host that cannot run anything,
//              GitHub Pages being the point of the exercise. Only matches
//              that were published with `cs2mv publish` exist, but they need
//              no server at all and stay up forever.
//   bridge   - a static page that has found a cs2mv running on the reader's
//              own machine. The published matches stay available and their
//              own demos become available too.
//
// Everything addresses data through this module, so the rest of the page does
// not care which of the three it is talking to. URLs are deliberately
// relative: a project page lives under /<repo>/, not at the root.

const Backend = (() => {
  // Where a local cs2mv would be listening. Checked in order, briefly.
  const BRIDGE_ORIGINS = ['http://127.0.0.1:8080', 'http://127.0.0.1:8081'];
  const BRIDGE_TIMEOUT_MS = 1200;

  const state = {
    mode: 'live',
    catalog: [],    // static: the published matches, newest first
    bridge: '',     // origin of a local cs2mv, when one answered
  };

  // The published bundle carries data/index.json; a live server does not.
  async function detect() {
    try {
      const response = await fetch('data/index.json', { cache: 'no-cache' });
      if (response.ok) {
        const body = await response.json();
        state.catalog = Array.isArray(body.matches) ? body.matches : [];
        state.mode = 'static';
      }
    } catch (_) { /* no bundle: a live server, then */ }
    if (state.mode === 'static') await findBridge();
    return state.mode;
  }

  // A published page can still drive a cs2mv on the reader's own machine,
  // which is what lets the public site open demos that were never published.
  // The server only answers a cross-origin caller when it was started with
  // --allow-origin, so this quietly finds nothing unless that was meant.
  async function findBridge() {
    for (const origin of BRIDGE_ORIGINS) {
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), BRIDGE_TIMEOUT_MS);
        const response = await fetch(origin + '/api/health', { signal: controller.signal });
        clearTimeout(timer);
        if (response.ok) {
          state.bridge = origin;
          return;
        }
      } catch (_) { /* nothing listening there, or it refused us */ }
    }
  }

  // A published match, by its id or by the share code it was published under.
  function findPublished(target) {
    if (!target) return null;
    const key = String(target).trim();
    const upper = key.toUpperCase();
    return state.catalog.find(
      (m) => m.id === key || (m.shareCode || '').toUpperCase() === upper) || null;
  }

  // Which source answers for this target. A published match is served from
  // files even when a bridge exists, because files are faster and cannot
  // fail; anything else needs a server.
  function routeFor(target) {
    if (state.mode === 'static') {
      const published = findPublished(target);
      if (published) return { kind: 'files', id: published.id };
      if (state.bridge) return { kind: 'api', origin: state.bridge };
      return { kind: 'none' };
    }
    return { kind: 'api', origin: '' };
  }

  function api(origin, path) {
    return origin ? origin + '/' + path : path;
  }

  // Why a target cannot be opened, in words the reader can act on.
  function explain(target) {
    if (state.catalog.length === 0) {
      return 'This site has no published matches yet.';
    }
    return 'That match is not published on this site. Published matches are ' +
      'listed above. To open your own demos here, run cs2mv on your machine ' +
      'with --allow-origin ' + window.location.origin + ' and reload.';
  }

  return {
    detect,
    mode: () => state.mode,
    isStatic: () => state.mode === 'static',
    bridge: () => state.bridge,
    // Live, or static with a local server behind it: either way the server
    // features (Steam sign-in, your own demos, Watch in CS2) are available.
    hasServer: () => state.mode === 'live' || !!state.bridge,
    serverOrigin: () => (state.mode === 'live' ? '' : state.bridge),
    catalog: () => state.catalog,
    findPublished,
    routeFor,
    explain,

    serverUrl: (path) => api(state.mode === 'live' ? '' : state.bridge, path),

    matchUrl(target) {
      const route = routeFor(target);
      if (route.kind === 'files') return `data/${route.id}/match.json`;
      if (route.kind === 'api') {
        return api(route.origin, 'api/match?code=' + encodeURIComponent(target));
      }
      return '';
    },

    replayUrl(target, round) {
      const route = routeFor(target);
      if (route.kind === 'files') return `data/${route.id}/round-${round}.json`;
      if (route.kind === 'api') {
        return api(route.origin,
                   `api/replay?code=${encodeURIComponent(target)}&round=${round}`);
      }
      return '';
    },

    // The blurred walkable-area image the replay falls back to.
    mapUrl(target) {
      const route = routeFor(target);
      if (route.kind === 'files') return `data/${route.id}/map.png`;
      if (route.kind === 'api') {
        return api(route.origin, 'api/replay/map?code=' + encodeURIComponent(target));
      }
      return '';
    },

    // The unblurred footprint the radar is fitted against. `zMin` drops a
    // lower floor; publishing writes exactly one height-filtered footprint,
    // for Nuke, so any zMin maps to that one.
    walkedUrl(target, zMin) {
      const route = routeFor(target);
      if (route.kind === 'files') {
        return `data/${route.id}/${zMin === undefined ? 'walked' : 'walked-upper'}.png`;
      }
      if (route.kind === 'api') {
        let url = api(route.origin, 'api/replay/map?code=' + encodeURIComponent(target) + '&raw=1');
        if (zMin !== undefined) url += '&zmin=' + zMin;
        return url;
      }
      return '';
    },
  };
})();
