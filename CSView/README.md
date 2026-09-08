# cs2-match-viewer

Open a Counter-Strike 2 match from its share code and view it in a web browser.
C++17, no third-party dependencies — the protobuf reader, Snappy and bzip2
decompressors, demo parser and HTTP server/client are all in this repository.

```
cs2mv serve                # http://127.0.0.1:8080
cs2mv decode CSGO-Cji4Z-rQMJJ-s6Jyq-ovwoS-mJkDA
cs2mv add    CSGO-Cji4Z-... http://replay191.valve.net/730/003693....dem.bz2
cs2mv parse  match.dem --pretty
```

The UI shows the map and score, a scoreboard (K/D/A, ADR, HS%, MVPs, entry
kills, flashes, utility damage), a round-by-round timeline and a per-round kill
feed.

---

## One correction to the usual description

A CS2 share code is **25 payload characters in five groups**, not 20 in four:

```
CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx
```

25 base-57 characters carry ~146 bits, which is what the 144-bit payload needs.
Four groups could not hold it. The decoder accepts the code with or without the
`CSGO-` prefix and with or without dashes, but a four-group code is rejected
with an explanation rather than silently mis-decoded.

## What a share code actually contains

Three fields, and no URL:

| field | type | meaning |
|---|---|---|
| `match_id` | uint64 | the game coordinator's id for the match |
| `outcome_id` | uint64 | reservation id, identifies the outcome record |
| `token` | uint16 | also called `tv_port`, authorises the lookup |

Decoding is pure arithmetic: strip the prefix and dashes, read the 25
characters as base-57 digits least-significant first into a 144-bit integer,
then read three little-endian fields out of the resulting 18 bytes. See
[`src/sharecode.cpp`](src/sharecode.cpp). This is verified against ten vectors
that both [ValvePython/csgo](https://github.com/ValvePython/csgo) and
[akiver/csgo-sharecode](https://github.com/akiver/csgo-sharecode) agree on,
plus a 500-case encode/decode round trip.

## Getting the demo

This is the part no amount of C++ can shortcut, so it is worth being blunt
about it.

A share code is only a lookup key. Turning it into a download URL means asking
the CS2 game coordinator:

```
client -> GC   k_EMsgGCCStrike15_v2_MatchListRequestFullGameInfo (9147)
               { matchid = 1, outcomeid = 2, token = 3 }
GC -> client   k_EMsgGCCStrike15_v2_MatchList (9139)
               { matches: [ { matchid, matchtime, roundstatsall: [...] } ] }
```

and the demo URL is the `map` field of the last `roundstatsall` entry — for a
finished match it holds `http://replayNNN.valve.net/730/....dem.bz2` rather
than a map name.

Those messages only travel inside an authenticated Steam client session. That
means a Steam login, an app ticket and the Steam GC message framing — a
dependency this project does not take on, and one no library can remove
because Valve requires the account.

So the project splits the problem:

* **Implemented here** — [`include/cs2mv/gc.h`](include/cs2mv/gc.h) builds the
  request body and parses the reply, including the demo URL and the players'
  Steam ids. `cs2mv gc-request <share code>` prints the exact bytes to send.
* **Yours to bridge** — the Steam session itself. Any Steam client library
  works: [SteamKit2](https://github.com/SteamRE/SteamKit) (C#),
  [ValvePython/steam](https://github.com/ValvePython/steam) (Python),
  [go-steam](https://github.com/Philipp15b/go-steam) (Go), or the Steamworks
  SDK. Send message 9147 with the body from `gc-request`, take the URL out of
  the reply, then `cs2mv add <share code> <url>`.

### The shortcut that avoids Steam entirely

**If the match is already downloaded, a share code is all you need.** Click
Download in CS2 under *Watch → Your Matches*, then paste the code into the web
UI or hand it to `cs2mv parse`. There is no `add` step:

* every CS2 replay folder on the machine is searched, across all Steam
  libraries - the Steam install is found through the registry, and
  `steamapps/libraryfolders.vdf` gives the rest, so a library on another drive
  or under OneDrive is found like any other;
* demos are named `match730_<21 digit id>_....dem`, where that id is the share
  code's **outcome** id, and the id is parsed out of the name rather than
  searched for, so a short id cannot match by accident;
* the hit is written into the index, so the next lookup skips the search.

Input is forgiving: a bare code, a code with text around it, or the whole
`steam://rungame/730/.../+csgo_download_match%20CSGO-...` URL that CS2's copy
button produces all work.

Failing that, point at a file or URL yourself - `cs2mv add CSGO-...
C:\path\to\match.dem` - or just `cs2mv parse match.dem`.

Registered URLs are downloaded and unpacked on first use into the cache
(`%LOCALAPPDATA%\cs2-match-viewer` / `~/.cache/cs2-match-viewer`). Valve keeps
match demos for roughly 30 days; after that the URL returns 404 and the viewer
says so.

## Building

Needs a C++17 compiler and CMake 3.15+. Nothing else.

```bash
cmake -B build -S .
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

The CLI lands at `build/cs2mv` and the web assets are copied next to it.

### Windows

`build.bat` handles the parts that are not obvious, and needs no shell setup:

```
build.bat                 :: builds into .\build
build.bat C:\cs2build     :: builds somewhere else
```

It finds Visual Studio with `vswhere`, runs `vcvars64.bat`, and uses the Ninja
generator. Three things it works around, all of which bite a plain
`cmake -B build -S .` on a fresh Build Tools install:

* **MSVC is never on `PATH`.** It only exists inside a developer shell, so
  `vcvars64.bat` has to run first.
* **The default Visual Studio generator can fail** with *"No CMAKE_CXX_COMPILER
  could be found"* even when `cl.exe` is present, because that generator drives
  MSBuild and a bare Build Tools install may lack the VC MSBuild integration.
  Ninja only needs `cl.exe`, and Visual Studio ships both Ninja and CMake under
  `Common7\IDE\CommonExtensions\Microsoft\CMake`.
* **MSVC caps object paths at 250 characters.** A deep source tree fails with
  `fatal error C1083: Cannot open compiler generated file: '': Invalid
  argument`. Pass a short build directory when that happens; the script warns
  first.

If you only have Windows PowerShell 5.1, note it has no `&&` operator — run the
two `cmake` commands on separate lines, or use `build.bat`.

Toolchain, if you need one:

```
winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

## How the demo parsing works

CS2 demos describe a match twice over: as a stream of *game events* (the same
events a server plugin subscribes to) and as a full *entity snapshot* system
that carries every networked property of every entity, tick by tick.

**This parser reads the first and ignores the second.** Game events plus the
`userinfo` string table are enough for a scoreboard, a round timeline and a
kill feed, and they avoid the flattened-serializer decoding that makes up the
bulk — and the version fragility — of a Source 2 demo parser.

The layers, bottom up:

| file | what it does |
|---|---|
| [`src/bzip2.cpp`](src/bzip2.cpp) | unpacks `.dem.bz2` downloads |
| [`src/snappy.cpp`](src/snappy.cpp) | unpacks frames flagged `DEM_IsCompressed` |
| [`src/protobuf.cpp`](src/protobuf.cpp) | protobuf wire reader/writer |
| [`include/cs2mv/bits.h`](include/cs2mv/bits.h) | Source 2's LSB-first bit stream, `ReadUBitVar` |
| [`src/demo.cpp`](src/demo.cpp) | the PBDEMS2 container: varint-framed commands |
| [`src/parser.cpp`](src/parser.cpp) | messages → events → scoreboard and rounds |

Field numbers are taken from Valve's own protobufs (as published in
[SteamDatabase/GameTracking-CS2](https://github.com/SteamDatabase/GameTracking-CS2))
and are documented at the call site of each message.

### Two decisions worth knowing about

**Player identity.** Game events reference players by a user id with a
generation counter packed into the bits above the low byte, so the low byte is
what identifies the player — the same rule
[demoinfocs-golang](https://github.com/markus-wa/demoinfocs-golang) applies to
Source 2 demos. Zero means *nobody* (world damage, an absent assister) and is
never a player. The user id map, built from the `userinfo` string table and
`player_connect` events, is authoritative; a slot-index fallback runs only when
no user ids were learned at all, because slots and user ids share a number
space and guessing between them would silently credit the wrong player. If
resolution fails for most events the parser says so in `warnings` instead of
reporting a plausible but empty scoreboard. All of it lives in one function,
`PlayerFromEvent` in [`src/parser.cpp`](src/parser.cpp).

**Warmup.** Stats reset on `begin_new_match` / `round_announce_match_start`, so
warmup and knife rounds do not reach the scoreboard, and a round only counts
once a `round_end` names a winner. Turn this off with
`ParseOptions::reset_on_match_start`.

## Testing

`tools/make_fixture.py` builds `tests/data/synthetic.dem`: a real PBDEMS2
container with Snappy-compressed frames, bit-packed Source 2 message framing,
protobuf messages, a `userinfo` string table and legacy game events staging a
three-round match (plus warmup activity that must be discarded). The C++ tests
parse it through the ordinary code path and check the resulting scoreboard
value by value, which exercises every layer above.

The bzip2 decoder is checked against a multi-block stream produced by a real
bzip2 compressor; Snappy against hand-computed vectors covering each tag type,
including an overlapping copy.

```bash
python tools/make_fixture.py     # regenerate fixtures
ctest --test-dir build --output-on-failure
```

## What retail demos actually contain

Validated against two real CS2 matchmaking demos (de_mirage and de_ancient,
~220 MB each). The container, Snappy, protobuf and bit-level framing all work
unchanged; player names, kills, deaths, assists, damage, headshots and flashes
all come out consistent (kills plus attacker-less deaths equal total deaths, to
the event). Three things about real demos were not what the documentation
suggests, and each is worth knowing before trusting a number:

* **Players are addressed by slot, not by user id.** The `userinfo` string
  table reports a user id of `0xFF00 | slot`, while game events use the plain
  0-based slot. Both are normalised to the low byte. `0xFFFF` (a `short`'s -1)
  means "no player"; **0 is a real player**, so it must never be read as absent.
* **`round_start` and `round_end` are never sent.** Retail GOTV recordings
  carry `round_prestart` / `round_poststart` / `round_freeze_end` /
  `round_officially_ended` and no winner at all. Round winners are therefore
  *deduced* from the bomb outcome and eliminations, and flagged with
  `winnerInferred` in the JSON.
* **Sides are only announced at the halftime swap.** `player_team` fires once
  per player, at the swap, so nobody has a side during the first half. Two
  consequences. Rounds played is counted for every participant rather than for
  players with a known side, or the whole first half is skipped and every ADR
  comes out inflated fourfold. And round winners cannot be settled while
  reading: they are resolved in a single pass afterwards, because rosters are
  fixed for the match even though sides are not. The swap changes which side a
  roster plays, not who is on it, so once every player has a side the first
  half can be reconstructed by playing the swap backwards.

`Round::winner` therefore names a team the way the scoreboard groups them - by
the side they *finished* on - not the side they were playing that round. That
is what a match score means. Round reasons are worded without naming a side
("Opponents eliminated", "Target bombed") so they stay true in both halves.

Validated end to end: three retail matches resolve every round, two of them
finishing on exactly 13 - which is what a first-to-13 match does - and all
three report the swap at round 13, i.e. halftime after round 12, which is MR12.

## Limitations

* **Round winners are deduced, not read.** The bomb settles a round outright;
  otherwise a side that lost everyone lost the round; otherwise the clock ran
  out and the CTs held. This matches every retail match tested, but it is
  inference, and `winnerInferred` marks it in the JSON. A round won by a team
  the demo never showed dying - a full save with the clock running out, say -
  is the case to distrust.
* The halftime swap is taken to be the first round in which `player_team`
  fires. A demo that announces sides at the start instead is handled (no
  reversal is applied), but a mid-match team change in a demo that never had a
  proper swap would be misread.
* `cs2mv inspect <demo>` reports the frames, message kinds, string tables, game
  events *with their key names*, the userinfo table and the raw player
  references seen in events. That is the first thing to run when a demo parses
  oddly - it is how every item above was diagnosed.
* MVPs are always zero on retail demos: `round_mvp` is not broadcast either.
* `CDemoFileInfo` sits after `DEM_Stop` and so is never reached by a forward
  read; playback length falls back to the last tick seen.
* No entity state, so no positions, economy, equipment, or anything derived
  from them (opening duels by site, trade kills, clutch detection).
* Bit-packed `svc_CreateStringTable` / `svc_UpdateStringTable` deltas are not
  decoded. Player identity comes from the full string table snapshots in
  `DEM_StringTables` / `DEM_FullPacket` frames, which are plain protobuf.
* The whole demo is held in memory while parsing — a few hundred MB for a long
  Premier match.
* The HTTP client is plain HTTP, deliberately: Valve replay URLs are `http://`
  and TLS would mean a crypto dependency. `https://` URLs are refused with an
  explanation.
* The server binds to loopback and is a local viewer, not a public web server.
  Serving it to an untrusted network is not a supported configuration.

## Layout

```
include/cs2mv/   public headers, one per subsystem
src/             implementations, plus main.cpp and the socket shim
web/             the UI (vanilla HTML/CSS/JS, no build step)
tests/           test binary and fixtures
tools/           fixture generator
```
