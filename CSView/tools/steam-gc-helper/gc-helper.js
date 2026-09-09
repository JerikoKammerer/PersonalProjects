#!/usr/bin/env node
//
// Turns a CS2 share code into a demo download URL by asking the game
// coordinator, so that matches can be pulled without downloading them in CS2
// first.
//
// Why this is a separate program
// ------------------------------
// Only the CS2 game coordinator can turn a share code into a URL, and it
// answers only a logged-in Steam client. That means Steam credentials, and
// credentials should live in exactly one place, owned by the thing that needs
// them - not in a C++ web server, and never in a browser form. cs2mv shells out
// to this and reads a URL back; it never sees an account.
//
// What is stored
// --------------
// A refresh token, and nothing else. Logging in is done by scanning a QR code
// with the Steam mobile app, so no password is ever typed, let alone saved.
// The token goes to ~/.cs2mv-steam.json with owner-only permissions. Revoke it
// any time from Steam > Settings > Security, or by deleting that file.
//
// Usage
// -----
//   node gc-helper.js login                     one-time, shows a QR code
//   node gc-helper.js <matchid> <outcomeid> <token>
//   node gc-helper.js CSGO-xxxxx-...            a share code works too
//
// On success the demo URL is printed on stdout and the exit status is 0. That
// is the whole contract cs2mv depends on.
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');

const CONFIG = process.env.CS2MV_STEAM_CONFIG ||
    path.join(os.homedir(), '.cs2mv-steam.json');
const APPID = 730;
const GC_TIMEOUT_MS = 45000;

function die(message) {
  console.error(message);
  process.exit(1);
}

function requireDeps() {
  try {
    return {
      SteamUser: require('steam-user'),
      GlobalOffensive: require('globaloffensive'),
    };
  } catch (err) {
    die('Missing dependencies. Run this once, in this directory:\n' +
        '  npm install\n\n' + err.message);
  }
}

function readToken() {
  try {
    const raw = JSON.parse(fs.readFileSync(CONFIG, 'utf8'));
    if (raw && typeof raw.refreshToken === 'string' && raw.refreshToken) {
      return raw.refreshToken;
    }
  } catch (err) {
    if (err.code !== 'ENOENT') die(`Cannot read ${CONFIG}: ${err.message}`);
  }
  return null;
}

function writeToken(refreshToken, accountName) {
  const body = JSON.stringify({ refreshToken, accountName }, null, 2);
  // Create with owner-only permissions rather than relaxing them afterwards,
  // so the token is never briefly world readable.
  fs.writeFileSync(CONFIG, body, { mode: 0o600 });
  try {
    fs.chmodSync(CONFIG, 0o600);
  } catch (_) {
    // Windows does not implement POSIX modes; the file inherits the user's
    // profile ACL, which is already owner-only.
  }
}

// ---------------------------------------------------------------- login

// The browser sign-in drives this same flow through a line protocol on stdout,
// one event per line, so the C++ server needs no JSON parser and no Steam
// knowledge:
//
//   QR <data:image/png;base64,...>   the challenge, ready to put in an <img>
//   URL <https://s.team/q/...>       the same challenge as a link
//   SCANNED                          phone has it, waiting for approval
//   DONE <account name>              token written
//   ERROR <message>
//
// Nothing secret crosses this channel: the QR is a challenge, and the phone
// talks to Steam directly. The refresh token is written to disk by this
// process and never printed.
async function loginStream() {
  let LoginSession, EAuthTokenPlatformType;
  let QRCode;
  try {
    ({ LoginSession, EAuthTokenPlatformType } = require('steam-session'));
    QRCode = require('qrcode');
  } catch (err) {
    console.log(`ERROR missing dependencies, run npm install (${err.message})`);
    process.exit(1);
  }

  const emit = (line) => {
    process.stdout.write(line + '\n');
  };

  try {
    const session = new LoginSession(EAuthTokenPlatformType.SteamClient);
    const started = await session.startWithQR();

    const png = await QRCode.toDataURL(started.qrChallengeUrl,
                                       { margin: 2, width: 320 });
    emit(`QR ${png}`);
    emit(`URL ${started.qrChallengeUrl}`);

    session.on('remoteInteraction', () => emit('SCANNED'));

    await new Promise((resolve, reject) => {
      session.on('authenticated', resolve);
      session.on('timeout', () => reject(new Error('the QR code expired')));
      session.on('error', reject);
    });

    writeToken(session.refreshToken, session.accountName);
    session.cancelLoginAttempt();
    emit(`DONE ${session.accountName}`);
    process.exit(0);
  } catch (err) {
    emit(`ERROR ${err.message}`);
    process.exit(1);
  }
}

function status() {
  const token = readToken();
  if (!token) {
    console.log('SIGNEDOUT');
    return;
  }
  let account = '';
  try {
    account = JSON.parse(fs.readFileSync(CONFIG, 'utf8')).accountName || '';
  } catch (_) { /* the token is what matters */ }
  console.log(`SIGNEDIN ${account}`);
}

function logout() {
  try {
    fs.unlinkSync(CONFIG);
    console.log('SIGNEDOUT');
  } catch (err) {
    if (err.code === 'ENOENT') {
      console.log('SIGNEDOUT');
      return;
    }
    die(`Cannot remove ${CONFIG}: ${err.message}`);
  }
}

async function login() {
  let LoginSession, EAuthTokenPlatformType, LoginApprover;
  try {
    ({ LoginSession, EAuthTokenPlatformType } = require('steam-session'));
  } catch (err) {
    die('Missing dependencies. Run: npm install\n\n' + err.message);
  }
  let qrcode;
  try {
    qrcode = require('qrcode-terminal');
  } catch (_) {
    qrcode = null;
  }

  const session = new LoginSession(EAuthTokenPlatformType.SteamClient);

  const started = await session.startWithQR();
  console.error('\nScan this with the Steam mobile app (Steam Guard > scan QR):\n');
  if (qrcode) {
    qrcode.generate(started.qrChallengeUrl, { small: true },
                    (art) => console.error(art));
  }
  console.error(`\nOr open this link on your phone:\n  ${started.qrChallengeUrl}\n`);

  session.on('remoteInteraction', () => {
    console.error('QR scanned - approve the sign-in on your phone.');
  });

  await new Promise((resolve, reject) => {
    session.on('authenticated', resolve);
    session.on('timeout', () => reject(new Error('login timed out')));
    session.on('error', reject);
  });

  writeToken(session.refreshToken, session.accountName);
  console.error(`\nSigned in as ${session.accountName}.`);
  console.error(`Refresh token saved to ${CONFIG} (no password was stored).`);
  console.error('\nNow point cs2mv at this helper:');
  console.error(`  cs2mv serve --gc-helper "node ${path.resolve(__filename)}"`);
  session.cancelLoginAttempt();
  process.exit(0);
}

// ------------------------------------------------------------ url lookup

function demoUrlFrom(match) {
  // CDataGCCStrike15_v2_MatchInfo.roundstatsall: the last entry's `map` field
  // holds the demo URL for a finished match rather than a map name.
  const rounds = (match && match.roundstatsall) || [];
  for (let i = rounds.length - 1; i >= 0; --i) {
    const map = rounds[i] && rounds[i].map;
    if (typeof map === 'string' && map.startsWith('http')) return map;
  }
  return null;
}

// Lists the account's recent matches, the way CS2's own "Your Matches" tab
// does. One line each, so cs2mv can read it without a JSON parser:
//
//   MATCH <match id> <outcome id> <token> <unix time> <demo url or ->
//
// cs2mv rebuilds the share code from the first three, which is why no code has
// to be typed anywhere.
async function recent() {
  const { SteamUser, GlobalOffensive } = requireDeps();
  const refreshToken = readToken();
  if (!refreshToken) {
    console.log('ERROR not signed in');
    process.exit(1);
  }

  const user = new SteamUser();
  const cs = new GlobalOffensive(user);
  const finish = (code, line) => {
    if (line) console.log(line);
    try { user.logOff(); } catch (_) { /* already gone */ }
    process.exit(code);
  };

  const timer = setTimeout(() => {
    finish(1, 'ERROR timed out waiting for the CS2 game coordinator');
  }, GC_TIMEOUT_MS);

  user.on('error', (err) => {
    clearTimeout(timer);
    if (err && err.eresult === 6) {
      finish(1, 'ERROR this account is playing CS2 elsewhere; close the game ' +
                'and try again');
    }
    finish(1, `ERROR ${err.message}`);
  });

  user.on('loggedOn', () => user.gamesPlayed([APPID]));

  cs.on('connectedToGC', () => {
    if (cs.requestRecentGames(user.steamID) === false) {
      clearTimeout(timer);
      finish(1, 'ERROR could not ask for recent games');
    }
  });

  cs.on('matchList', (matches) => {
    clearTimeout(timer);
    for (const m of matches || []) {
      const rounds = m.roundstatsall || [];
      const last = rounds.length ? rounds[rounds.length - 1] : null;
      const outcome = (last && last.reservationid) || m.matchid;
      const token = (m.watchablematchinfo && m.watchablematchinfo.tv_port) || 0;
      const url = demoUrlFrom(m) || '-';
      console.log(`MATCH ${m.matchid} ${outcome} ${token} ${m.matchtime || 0} ${url}`);
    }
    finish(0);
  });

  user.logOn({ refreshToken });
}

async function lookup(shareCode) {
  const { SteamUser, GlobalOffensive } = requireDeps();
  const refreshToken = readToken();
  if (!refreshToken) {
    die(`Not signed in yet. Run:\n  node ${path.resolve(__filename)} login`);
  }

  const user = new SteamUser();
  const cs = new GlobalOffensive(user);

  const finish = (code, message) => {
    if (message) console.error(message);
    try { user.logOff(); } catch (_) { /* already gone */ }
    process.exit(code);
  };

  const timer = setTimeout(() => {
    finish(1, 'Timed out waiting for the CS2 game coordinator. It is sometimes ' +
              'down or slow; try again in a minute.');
  }, GC_TIMEOUT_MS);

  user.on('error', (err) => {
    clearTimeout(timer);
    // EResult 6 is LoggedInElsewhere, which is a session conflict rather than a
    // bad token - telling someone to sign in again would send them the wrong
    // way. It happens when this account is already playing CS2 somewhere,
    // because the lookup has to "play" 730 to get a game coordinator session.
    if (err && err.eresult === 6) {
      finish(1, 'Steam signed this session out because the account is in use ' +
                'elsewhere. Close CS2 (or stop playing on the other device) and ' +
                'try again; the saved token is still fine.');
    }
    if (err && (err.eresult === 5 || err.eresult === 16 || err.eresult === 24)) {
      finish(1, `Steam rejected the stored token (${err.message}). Sign in again.`);
    }
    finish(1, `Steam login failed: ${err.message}`);
  });

  user.on('loggedOn', () => {
    // Being "in game" is what gets a game coordinator session.
    user.gamesPlayed([APPID]);
  });

  cs.on('connectedToGC', () => {
    cs.requestGame(shareCode);
  });

  cs.on('matchList', (matches) => {
    clearTimeout(timer);
    const url = matches && matches.length ? demoUrlFrom(matches[0]) : null;
    if (!url) {
      finish(1, 'The game coordinator returned no demo URL. Valve keeps match ' +
                'demos for about 30 days, so this match has probably expired.');
    }
    console.log(url);          // stdout: the one thing cs2mv reads
    finish(0);
  });

  user.logOn({ refreshToken });
}

// ----------------------------------------------------------------- main

function main() {
  const args = process.argv.slice(2);
  if (args.length === 0 || args[0] === '--help' || args[0] === '-h') {
    console.error(
        'usage:\n' +
        '  node gc-helper.js login\n' +
        '  node gc-helper.js <matchid> <outcomeid> <token>\n' +
        '  node gc-helper.js CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx\n');
    process.exit(2);
  }
  if (args[0] === 'login' && args[1] === '--stream') {
    loginStream();
    return;
  }
  if (args[0] === 'login') {
    login().catch((err) => die(`Login failed: ${err.message}`));
    return;
  }
  if (args[0] === 'status') {
    status();
    return;
  }
  if (args[0] === 'recent') {
    recent().catch((err) => {
      console.log(`ERROR ${err.message}`);
      process.exit(1);
    });
    return;
  }
  if (args[0] === 'logout') {
    logout();
    return;
  }

  // cs2mv passes the three decoded fields; globaloffensive wants either a share
  // code or exactly those, so hand it the object form when given numbers.
  let request;
  if (args.length >= 3) {
    request = { matchId: args[0], outcomeId: args[1], token: parseInt(args[2], 10) };
  } else {
    request = args[0];
  }
  lookup(request).catch((err) => die(`Lookup failed: ${err.message}`));
}

main();
