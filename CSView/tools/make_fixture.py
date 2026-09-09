#!/usr/bin/env python3
"""Builds tests/data/synthetic.dem, a small but structurally real CS2 demo.

Nothing here is a mock: the output is a PBDEMS2 container holding Snappy
compressed frames, bit-packed Source 2 message framing, protobuf encoded
CDemo*/CSVCMsg_*/CMsgSource1Legacy* messages and a userinfo string table. The
C++ test parses this file with the ordinary code path and checks the scoreboard
it produces, which exercises every layer below the parser.

Regenerate with:  python tools/make_fixture.py
"""

import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "tests", "data")


# --------------------------------------------------------------- primitives

def varint(v):
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


class Pb:
    """Protobuf wire-format writer."""

    def __init__(self):
        self.b = bytearray()

    def _tag(self, field, wire):
        self.b += varint((field << 3) | wire)

    def var(self, field, value):
        self._tag(field, 0)
        self.b += varint(value)
        return self

    def boolean(self, field, value):
        return self.var(field, 1 if value else 0)

    def fixed64(self, field, value):
        self._tag(field, 1)
        self.b += struct.pack("<Q", value)
        return self

    def f32(self, field, value):
        self._tag(field, 5)
        self.b += struct.pack("<f", value)
        return self

    def raw(self, field, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        self._tag(field, 2)
        self.b += varint(len(data)) + data
        return self

    def msg(self, field, sub):
        return self.raw(field, bytes(sub.b))

    def bytes(self):
        return bytes(self.b)


class BitOut:
    """Least-significant-bit-first writer, the mirror of cs2mv::BitReader."""

    def __init__(self):
        self.buf = bytearray()
        self.cur = 0
        self.n = 0

    def bits(self, value, count):
        for i in range(count):
            self.cur |= ((value >> i) & 1) << self.n
            self.n += 1
            if self.n == 8:
                self.buf.append(self.cur)
                self.cur = 0
                self.n = 0

    def ubitvar(self, v):
        if v < 16:
            self.bits(v, 6)
        elif v < 256:
            self.bits((v & 15) | 0x10, 6)
            self.bits(v >> 4, 4)
        elif v < 4096:
            self.bits((v & 15) | 0x20, 6)
            self.bits(v >> 4, 8)
        else:
            self.bits((v & 15) | 0x30, 6)
            self.bits(v >> 4, 28)

    def varint(self, v):
        for byte in varint(v):
            self.bits(byte, 8)

    def raw(self, data):
        for byte in data:
            self.bits(byte, 8)

    def done(self):
        if self.n:
            self.buf.append(self.cur)
            self.cur = 0
            self.n = 0
        return bytes(self.buf)


# ------------------------------------------------------------------ snappy

sys.path.insert(0, HERE)


def snappy_compress(data):
    """Literal + copy encoder; enough to exercise the decoder's tag paths."""
    out = bytearray()
    out += varint(len(data))
    table = {}
    i = 0
    lit_start = 0
    n = len(data)

    def emit_literal(lit):
        ln = len(lit) - 1
        if ln < 60:
            out.append(ln << 2)
        elif ln < 256:
            out.append(60 << 2)
            out.append(ln)
        else:
            out.append(61 << 2)
            out.append(ln & 0xFF)
            out.append((ln >> 8) & 0xFF)
        out.extend(lit)

    def emit_copy(off, ln):
        while ln > 0:
            take = min(ln, 64)
            if 4 <= take <= 11 and off < 2048:
                out.append(0x01 | ((take - 4) << 2) | ((off >> 8) << 5))
                out.append(off & 0xFF)
            else:
                out.append(0x02 | ((take - 1) << 2))
                out.append(off & 0xFF)
                out.append((off >> 8) & 0xFF)
            ln -= take

    while i + 4 <= n:
        key = data[i:i + 4]
        cand = table.get(key, -1)
        table[key] = i
        if cand >= 0 and i - cand < 65536:
            ln = 0
            while i + ln < n and ln < 64 and data[cand + ln] == data[i + ln]:
                ln += 1
            if ln >= 4:
                if i > lit_start:
                    emit_literal(data[lit_start:i])
                emit_copy(i - cand, ln)
                i += ln
                lit_start = i
                continue
        i += 1
    if lit_start < n:
        emit_literal(data[lit_start:n])
    return bytes(out)


# ------------------------------------------------------------- demo pieces

DEM_STOP = 0
DEM_FILE_HEADER = 1
DEM_FILE_INFO = 2
DEM_STRING_TABLES = 6
DEM_PACKET = 7
DEM_SIGNON_PACKET = 8
DEM_IS_COMPRESSED = 64

SVC_SERVER_INFO = 40
GE_EVENT_LIST = 205
GE_EVENT = 207

TICK_INTERVAL = 1.0 / 64.0

# Game events this fixture uses, with their key names in wire order.
EVENTS = {
    1: ("player_connect", ["name", "userid", "xuid", "bot"]),
    2: ("player_team", ["userid", "team", "oldteam"]),
    3: ("round_start", ["timelimit", "fraglimit", "objective"]),
    4: ("round_end", ["winner", "reason", "message"]),
    5: ("player_death", ["userid", "attacker", "assister", "weapon", "headshot",
                         "penetrated", "noscope", "thrusmoke", "attackerblind",
                         "assistedflash"]),
    6: ("player_hurt", ["userid", "attacker", "health", "weapon", "dmg_health",
                        "hitgroup"]),
    7: ("round_mvp", ["userid", "reason"]),
    8: ("bomb_planted", ["userid", "site"]),
    9: ("begin_new_match", []),
    10: ("player_blind", ["userid", "attacker", "blind_duration"]),
}

# Key value types, matching CMsgSource1LegacyGameEvent.key_t field numbers.
STRING, FLOAT, LONG, SHORT, BYTE, BOOL, UINT64 = 2, 3, 4, 5, 6, 7, 8


def event_list_message():
    pb = Pb()
    for event_id, (name, keys) in sorted(EVENTS.items()):
        desc = Pb().var(1, event_id).raw(2, name)
        for key in keys:
            desc.msg(3, Pb().var(1, 1).raw(2, key))
        pb.msg(1, desc)
    return pb.bytes()


def event_message(event_id, values):
    """values: list of (wire_field, python value) in the descriptor's key order."""
    name, keys = EVENTS[event_id]
    assert len(values) == len(keys), (name, len(values), len(keys))
    pb = Pb().raw(1, name).var(2, event_id)
    for field, value in values:
        key = Pb().var(1, field - 1)
        if field == STRING:
            key.raw(STRING, value)
        elif field == FLOAT:
            key.f32(FLOAT, value)
        elif field == BOOL:
            key.boolean(BOOL, value)
        elif field == UINT64:
            key.var(UINT64, value)
        else:
            key.var(field, value)
        pb.msg(3, key)
    return pb.bytes()


def packet_frame(messages):
    """Wraps bit-packed messages in a CDemoPacket."""
    bits = BitOut()
    for kind, payload in messages:
        bits.ubitvar(kind)
        bits.varint(len(payload))
        bits.raw(payload)
    return Pb().raw(3, bits.done()).bytes()


def frame(kind, tick, body, compress=False):
    if compress:
        body = snappy_compress(body)
        kind |= DEM_IS_COMPRESSED
    return varint(kind) + varint(tick) + varint(len(body)) + body


# ------------------------------------------------------------- the scenario

# Numbering follows retail CS2, as dumped by `cs2mv inspect` from a real demo:
# the userinfo string table carries a user id of 0xFF00 | slot, while game
# events refer to players by the plain 0-based slot.
PLAYERS = [
    # slot, userid, name, steamid, team
    (0, 0xFF00, "Ada", 76561198000000001, 3),
    (1, 0xFF01, "Bo", 76561198000000002, 3),
    (2, 0xFF02, "Cyd", 76561198000000003, 2),
    (3, 0xFF03, "Dex", 76561198000000004, 2),
]
BY_NAME = {name: (slot, userid, steam, team) for slot, userid, name, steam, team in PLAYERS}

# 0xFFFF is -1 read as the `short` these keys are declared as, and is how CS2
# spells "no player". Slot 0 is a real player, so 0 must never mean absent -
# Ada sits in slot 0 here precisely to keep that honest.
NO_PLAYER = 0xFFFF


def uid(name):
    """Game events address players by slot."""
    return BY_NAME[name][0]


def string_tables_frame():
    tables = Pb()
    userinfo = Pb().raw(1, "userinfo")
    for slot, userid, name, steam, _team in PLAYERS:
        info = (Pb().raw(1, name)
                    .fixed64(2, steam)
                    .var(3, userid)
                    .fixed64(4, steam)
                    .boolean(5, False)
                    .boolean(6, False))
        userinfo.msg(2, Pb().raw(1, str(slot)).raw(2, info.bytes()))
    # A second table the parser must ignore.
    tables.msg(1, userinfo)
    tables.msg(1, Pb().raw(1, "instancebaseline").msg(2, Pb().raw(1, "0").raw(2, b"\x00\x01")))
    return tables.bytes()


def build():
    out = bytearray()
    out += b"PBDEMS2\0" + struct.pack("<ii", 0, 0)

    header = (Pb().raw(1, "PBDEMS2\0")
                  .var(2, 13)
                  .raw(3, "Valve CS2 Server (fixture)")
                  .raw(4, "GOTV Demo")
                  .raw(5, "de_dust2")
                  .raw(6, "csgo")
                  .raw(11, "cs2-match-viewer fixture")
                  .var(13, 14000))
    out += frame(DEM_FILE_HEADER, 0, header.bytes())

    server_info = (Pb().var(10, 10)
                       .f32(13, TICK_INTERVAL)
                       .raw(14, "csgo")
                       .raw(15, "de_dust2")
                       .raw(17, "cs2-match-viewer fixture server"))
    out += frame(DEM_SIGNON_PACKET, 0,
                 packet_frame([(SVC_SERVER_INFO, server_info.bytes())]))
    out += frame(DEM_PACKET, 0, packet_frame([(GE_EVENT_LIST, event_list_message())]))
    out += frame(DEM_STRING_TABLES, 0, string_tables_frame(), compress=True)

    tick = [100]
    stats = {name: dict(kills=0, deaths=0, assists=0, headshots=0, damage=0,
                        mvps=0, entry=0, entry_deaths=0, flashed=0, util=0)
             for _, _, name, _, _ in PLAYERS}
    counted = [False]      # stats only count once the match has begun
    pending = []           # messages waiting to be flushed into a frame

    def emit(event_id, values, compress=False):
        pending.append((GE_EVENT, event_message(event_id, values)))
        if len(pending) >= 4:
            flush(compress)

    def flush(compress=False):
        if not pending:
            return
        out.extend(frame(DEM_PACKET, tick[0], packet_frame(pending), compress))
        pending.clear()
        tick[0] += 8

    def connect(name):
        slot, userid, steam, team = BY_NAME[name]
        emit(1, [(STRING, name), (SHORT, uid(name)), (UINT64, steam), (BOOL, False)])
        emit(2, [(SHORT, uid(name)), (BYTE, team), (BYTE, 0)])

    def hurt(attacker, victim, damage, weapon):
        emit(6, [(SHORT, uid(victim)), (SHORT, uid(attacker)), (SHORT, 0),
                 (STRING, weapon), (SHORT, damage), (BYTE, 1)])
        if counted[0]:
            stats[attacker]["damage"] += damage
            if weapon in ("hegrenade", "inferno", "molotov"):
                stats[attacker]["util"] += damage

    def kill(attacker, victim, weapon, headshot=False, assister=None,
             penetrated=0, compress=False):
        hurt(attacker, victim, 100, weapon)
        emit(5, [(SHORT, uid(victim)), (SHORT, uid(attacker)),
                 (SHORT, uid(assister) if assister else NO_PLAYER),
                 (STRING, weapon), (BOOL, headshot), (SHORT, penetrated),
                 (BOOL, False), (BOOL, False), (BOOL, False), (BOOL, False)],
             compress)
        if counted[0]:
            stats[attacker]["kills"] += 1
            stats[victim]["deaths"] += 1
            if headshot:
                stats[attacker]["headshots"] += 1
            if assister:
                stats[assister]["assists"] += 1

    for _, _, name, _, _ in PLAYERS:
        connect(name)
    flush()

    # Warmup activity: must not survive into the scoreboard.
    emit(3, [(FLOAT, 115.0), (SHORT, 0), (STRING, "warmup")])
    kill("Cyd", "Ada", "ak47")
    emit(4, [(BYTE, 2), (BYTE, 0), (STRING, "warmup ends")])
    flush()

    emit(9, [])
    flush()
    counted[0] = True

    rounds = [
        # (winner team, reason, [kill args], mvp, bomb planted)
        (3, 8, [("Ada", "Cyd", "m4a1", True, None), ("Bo", "Dex", "m4a1", False, None)],
         "Ada", False),
        (2, 9, [("Cyd", "Ada", "ak47", False, None), ("Dex", "Bo", "ak47", False, "Cyd")],
         "Dex", True),
        (3, 8, [("Ada", "Dex", "awp", False, None), ("Cyd", "Bo", "ak47", False, None),
                ("Ada", "Cyd", "awp", True, None)],
         "Ada", False),
    ]

    for index, (winner, reason, kills, mvp, planted) in enumerate(rounds):
        emit(3, [(FLOAT, 115.0), (SHORT, 0), (STRING, "round")])
        flush()
        if index == 0:
            # An extra flash and some utility damage, for the derived columns.
            emit(10, [(SHORT, uid("Cyd")), (SHORT, uid("Ada")), (FLOAT, 2.5)])
            stats["Ada"]["flashed"] += 1
            hurt("Ada", "Cyd", 40, "hegrenade")
        if planted:
            emit(8, [(SHORT, uid("Cyd")), (BYTE, 0)])
        for i, (attacker, victim, weapon, headshot, assister) in enumerate(kills):
            if i == 0:
                stats[attacker]["entry"] += 1
                stats[victim]["entry_deaths"] += 1
            kill(attacker, victim, weapon, headshot, assister, compress=(index == 2))
        emit(7, [(SHORT, uid(mvp)), (SHORT, 1)])
        stats[mvp]["mvps"] += 1
        emit(4, [(BYTE, winner), (BYTE, reason), (STRING, "round over")])
        flush(compress=(index == 1))

    out += frame(DEM_FILE_INFO, tick[0],
                 Pb().f32(1, tick[0] * TICK_INTERVAL).var(2, tick[0]).var(3, tick[0]).bytes())
    out += frame(DEM_STOP, tick[0], b"")

    return bytes(out), stats


def bzip2_sample_text():
    """The plaintext tests/test_bzip2.cpp regenerates and compares against.
    Long enough to span several bzip2 blocks at level 1."""
    lines = []
    for i in range(5000):
        lines.append("line %d: the quick brown fox jumps over the lazy dog\n" % i)
    return "".join(lines).encode("ascii")


def main():
    data, stats = build()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "synthetic.dem")
    with open(path, "wb") as f:
        f.write(data)
    print("wrote %s (%d bytes)" % (path, len(data)))

    import bz2
    raw = bzip2_sample_text()
    bz_path = os.path.join(OUT_DIR, "sample.bz2")
    with open(bz_path, "wb") as f:
        f.write(bz2.compress(raw, 1))
    print("wrote %s (%d bytes, %d uncompressed)" %
          (bz_path, os.path.getsize(bz_path), len(raw)))

    # The same demo, bzip2 packed, so the reader's transparent unpacking is
    # covered too.
    bz_demo = os.path.join(OUT_DIR, "synthetic.dem.bz2")
    with open(bz_demo, "wb") as f:
        f.write(bz2.compress(data, 9))
    print("wrote %s (%d bytes)" % (bz_demo, os.path.getsize(bz_demo)))
    print()
    print("expected scoreboard (3 rounds, CT 2 - 1 T):")
    print("%-6s %3s %3s %3s %3s %5s %4s %4s %4s %4s" %
          ("name", "K", "D", "A", "HS", "dmg", "mvp", "ent", "fl", "util"))
    for _, _, name, _, _ in PLAYERS:
        s = stats[name]
        print("%-6s %3d %3d %3d %3d %5d %4d %4d %4d %4d" %
              (name, s["kills"], s["deaths"], s["assists"], s["headshots"],
               s["damage"], s["mvps"], s["entry"], s["flashed"], s["util"]))


if __name__ == "__main__":
    main()
