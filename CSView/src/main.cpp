// cs2mv - open and view CS2 matches from a share code.
//
//   cs2mv serve [--port 8080] [--web <dir>] [--cache <dir>]
//   cs2mv decode CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx
//   cs2mv encode <matchid> <outcomeid> <token>
//   cs2mv add <share code> <demo path or http URL>
//   cs2mv list
//   cs2mv parse <demo.dem | demo.dem.bz2 | share code> [--pretty]
//   cs2mv inspect <demo.dem>
//   cs2mv fetch <url> <dest.dem>
//   cs2mv gc-request <share code>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "cs2mv/demo.h"
#include "cs2mv/gc.h"
#include "cs2mv/http_server.h"
#include "cs2mv/locator.h"
#include "cs2mv/match.h"
#include "cs2mv/parser.h"
#include "cs2mv/json.h"
#include "cs2mv/entities.h"
#include "cs2mv/protobuf.h"
#include "cs2mv/serializers.h"
#include "cs2mv/sharecode.h"
#include "cs2mv/steam_login.h"
#include "cs2mv/watch.h"

namespace {

using namespace cs2mv;

struct Options {
  int port = 8080;
  std::string bind_address = "127.0.0.1";
  std::string web_root = "web";
  std::string cache_dir;
  std::string index_path;
  std::string gc_helper;
  bool pretty = false;
  bool no_download = false;
};

void PrintUsage() {
  std::cout <<
      "cs2mv - view Counter-Strike 2 matches from a share code\n"
      "\n"
      "  cs2mv serve [--port N] [--bind ADDR] [--web DIR] [--cache DIR]\n"
      "        Start the local web UI (default http://127.0.0.1:8080).\n"
      "  cs2mv decode CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx\n"
      "        Show the match id, outcome id and token inside a share code.\n"
      "  cs2mv encode <matchid> <outcomeid> <token>\n"
      "        Build a share code from its three fields.\n"
      "  cs2mv add <share code|match id> <demo path or http URL>\n"
      "        Tell the viewer where a match's demo lives.\n"
      "  cs2mv list\n"
      "        Show the demo index.\n"
      "  cs2mv parse <demo file | share code> [--pretty]\n"
      "        Parse a demo and print the match as JSON.\n"
      "  cs2mv inspect <demo file>\n"
      "        Report which frames, messages and game events a demo contains.\n"
      "  cs2mv fetch <http URL> <destination.dem>\n"
      "        Download a demo, unpacking .bz2 on the way.\n"
      "  cs2mv gc-request <share code>\n"
      "        Print the game coordinator request bytes for this share code.\n"
      "\n"
      "Common flags:\n"
      "  --cache DIR    where downloaded demos are kept\n"
      "  --index FILE   the share code -> demo mapping (default <cache>/index.txt)\n"
      "  --pretty       indent JSON output\n"
      "  --no-download  never fetch over the network\n"
      "  --gc-helper C  command that turns a share code into a demo URL by asking\n"
      "                 the CS2 game coordinator, for matches CS2 has not already\n"
      "                 downloaded. See tools/steam-gc-helper. Also read from the\n"
      "                 CS2MV_GC_HELPER environment variable.\n";
}

// Pulls recognised flags out of `args`, leaving positional arguments behind.
bool ParseFlags(std::vector<std::string>* args, Options* options,
                std::string* error) {
  std::vector<std::string> positional;
  for (std::size_t i = 0; i < args->size(); ++i) {
    const std::string& a = (*args)[i];
    auto value = [&](const char* name) -> bool {
      if (i + 1 >= args->size()) {
        *error = std::string("missing value for ") + name;
        return false;
      }
      return true;
    };
    if (a == "--port") {
      if (!value("--port")) return false;
      options->port = std::atoi((*args)[++i].c_str());
    } else if (a == "--bind") {
      if (!value("--bind")) return false;
      options->bind_address = (*args)[++i];
    } else if (a == "--web") {
      if (!value("--web")) return false;
      options->web_root = (*args)[++i];
    } else if (a == "--cache") {
      if (!value("--cache")) return false;
      options->cache_dir = (*args)[++i];
    } else if (a == "--index") {
      if (!value("--index")) return false;
      options->index_path = (*args)[++i];
    } else if (a == "--gc-helper") {
      if (!value("--gc-helper")) return false;
      options->gc_helper = (*args)[++i];
    } else if (a == "--pretty") {
      options->pretty = true;
    } else if (a == "--no-download") {
      options->no_download = true;
    } else if (a == "-h" || a == "--help") {
      PrintUsage();
      std::exit(0);
    } else if (a.rfind("--", 0) == 0) {
      *error = "unknown flag " + a;
      return false;
    } else {
      positional.push_back(a);
    }
  }
  args->swap(positional);
  return true;
}

ResolveOptions MakeResolveOptions(const Options& options) {
  ResolveOptions resolve;
  resolve.cache_dir = options.cache_dir.empty() ? DefaultCacheDir() : options.cache_dir;
  resolve.index_path = options.index_path;
  resolve.allow_download = !options.no_download;
  resolve.gc_helper = options.gc_helper;
  // An environment variable keeps the flag off every command line once the
  // helper is set up.
  if (resolve.gc_helper.empty()) {
    const char* from_env = std::getenv("CS2MV_GC_HELPER");
    if (from_env != nullptr) resolve.gc_helper = from_env;
  }
  return resolve;
}

bool FileExists(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return static_cast<bool>(f);
}

// A target is treated as a share code when a code can be pulled out of it and
// there is no file by that name, so an oddly named demo still wins. The text
// need not be a bare code: CS2's copy button hands out a steam:// URL, and
// people paste codes with words around them.
bool LooksLikeShareCode(const std::string& s, std::string* code) {
  if (FileExists(s)) return false;
  return ExtractShareCode(s, code);
}

// Loads a match from either a demo file path or a share code.
bool LoadMatch(const std::string& target, const Options& options, Match* match,
               std::string* error) {
  std::string demo_path = target;
  std::string code_text;

  if (LooksLikeShareCode(target, &code_text)) {
    ShareCode code;
    if (!DecodeShareCode(code_text, &code, error)) return false;
    ResolveOptions resolve = MakeResolveOptions(options);
    resolve.progress = [](std::uint64_t done, std::uint64_t total) {
      if (total > 0) {
        std::fprintf(stderr, "\rdownloading %llu/%llu MB",
                     static_cast<unsigned long long>(done >> 20),
                     static_cast<unsigned long long>(total >> 20));
      } else {
        std::fprintf(stderr, "\rdownloading %llu MB",
                     static_cast<unsigned long long>(done >> 20));
      }
    };
    if (!ResolveDemo(code, code_text, resolve, &demo_path, error)) return false;
    std::fprintf(stderr, "\r");

    match->share_code = code_text;
    match->match_id = code.match_id;
    match->outcome_id = code.outcome_id;
    match->token = code.token;
  }

  DemoReader reader;
  if (!reader.Open(demo_path, error)) return false;
  match->demo_source = demo_path;
  return ParseDemo(&reader, ParseOptions(), match, error);
}

int CommandDecode(const std::vector<std::string>& args) {
  if (args.empty()) {
    std::cerr << "usage: cs2mv decode <share code>\n";
    return 2;
  }
  ShareCode code;
  std::string error;
  if (!DecodeShareCode(args[0], &code, &error)) {
    std::cerr << "error: " << error << "\n";
    return 1;
  }
  std::string canonical;
  NormalizeShareCode(args[0], &canonical, nullptr);
  std::cout << "share code : " << canonical << "\n"
            << "match id   : " << code.match_id << "\n"
            << "outcome id : " << code.outcome_id << "\n"
            << "token      : " << code.token << "\n";
  return 0;
}

int CommandEncode(const std::vector<std::string>& args) {
  if (args.size() < 3) {
    std::cerr << "usage: cs2mv encode <matchid> <outcomeid> <token>\n";
    return 2;
  }
  ShareCode code;
  code.match_id = std::strtoull(args[0].c_str(), nullptr, 10);
  code.outcome_id = std::strtoull(args[1].c_str(), nullptr, 10);
  code.token = static_cast<std::uint16_t>(std::strtoul(args[2].c_str(), nullptr, 10));
  std::cout << EncodeShareCode(code) << "\n";
  return 0;
}

int CommandAdd(const std::vector<std::string>& args, const Options& options) {
  if (args.size() < 2) {
    std::cerr << "usage: cs2mv add <share code|match id> <demo path or URL>\n";
    return 2;
  }
  std::string key = args[0];
  std::string canonical;
  if (NormalizeShareCode(args[0], &canonical, nullptr)) key = canonical;

  const ResolveOptions resolve = MakeResolveOptions(options);
  const std::string index_path = resolve.index_path.empty()
                                     ? resolve.cache_dir + "/index.txt"
                                     : resolve.index_path;
  DemoIndex index;
  std::string error;
  if (!index.Load(index_path, &error)) {
    std::cerr << "error: " << error << "\n";
    return 1;
  }
  index.Set(key, args[1]);
  if (!index.Save(index_path, &error)) {
    std::cerr << "error: " << error << "\n";
    return 1;
  }
  std::cout << "recorded " << key << " -> " << args[1] << "\n"
            << "index: " << index_path << "\n";
  return 0;
}

int CommandList(const Options& options) {
  const ResolveOptions resolve = MakeResolveOptions(options);
  const std::string index_path = resolve.index_path.empty()
                                     ? resolve.cache_dir + "/index.txt"
                                     : resolve.index_path;
  DemoIndex index;
  std::string error;
  if (!index.Load(index_path, &error)) {
    std::cerr << "error: " << error << "\n";
    return 1;
  }
  if (index.entries().empty()) {
    std::cout << "the index at " << index_path << " is empty\n";
    return 0;
  }
  for (const auto& entry : index.entries()) {
    std::cout << entry.first << "\t" << entry.second << "\n";
  }
  return 0;
}

int CommandParse(const std::vector<std::string>& args, const Options& options) {
  if (args.empty()) {
    std::cerr << "usage: cs2mv parse <demo file | share code>\n";
    return 2;
  }
  Match match;
  std::string error;
  if (!LoadMatch(args[0], options, &match, &error)) {
    std::cerr << "error: " << error << "\n";
    return 1;
  }
  std::cout << MatchToJson(match, options.pretty) << "\n";
  return 0;
}

int CommandInspect(const std::vector<std::string>& args) {
  if (args.empty()) {
    std::cerr << "usage: cs2mv inspect <demo file>\n";
    return 2;
  }
  DemoReader reader;
  std::string error;
  if (!reader.Open(args[0], &error)) {
    std::cerr << "error: " << error << "\n";
    return 1;
  }
  DemoInventory inventory;
  const bool ok = InspectDemo(&reader, &inventory, &error);

  std::cout << "frames\n";
  for (const auto& entry : inventory.frames) {
    std::cout << "  " << DemoCommandName(entry.first) << " (" << entry.first
              << ") x" << entry.second << "\n";
  }
  std::cout << "  compressed: " << inventory.compressed_frames << "\n"
            << "  payload bytes: " << inventory.total_bytes << "\n"
            << "  last tick: " << inventory.last_tick << "\n";

  std::cout << "packet messages\n";
  for (const auto& entry : inventory.messages) {
    std::cout << "  kind " << entry.first << " x" << entry.second << "\n";
  }
  std::cout << "string tables\n";
  for (const auto& entry : inventory.string_tables) {
    std::cout << "  " << entry.first << " x" << entry.second << "\n";
  }
  std::cout << "game events (name xcount [key names])\n";
  for (const auto& entry : inventory.events) {
    std::cout << "  " << entry.first << " x" << entry.second;
    auto keys = inventory.event_keys.find(entry.first);
    if (keys != inventory.event_keys.end() && !keys->second.empty()) {
      std::cout << " [";
      for (std::size_t i = 0; i < keys->second.size(); ++i) {
        if (i != 0) std::cout << ", ";
        std::cout << keys->second[i];
      }
      std::cout << "]";
    }
    std::cout << "\n";
  }

  // Player identity, the usual culprit when a scoreboard comes out empty.
  std::cout << "userinfo entries (slot / userid / steamid / name)\n";
  for (const auto& entry : inventory.user_info) {
    std::cout << "  " << entry.slot << "\t" << entry.user_id << "\t"
              << entry.steam_id << "\t" << entry.name << "\n";
  }
  std::cout << "player references in game events (raw value x count)\n";
  for (const auto& key : inventory.event_player_refs) {
    std::cout << "  " << key.first << ":";
    for (const auto& value : key.second) {
      std::cout << " " << value.first << "x" << value.second;
    }
    std::cout << "\n";
  }
  if (!ok) {
    std::cerr << "warning: " << error << "\n";
    return 1;
  }
  return 0;
}

// Dumps the entity schema a demo ships with. This is the ground truth the 2D
// replay is built on, so being able to look at it directly matters.
int CommandSchema(const std::vector<std::string>& args) {
  if (args.empty()) {
    std::cerr << "usage: cs2mv schema <demo file> [serializer name]\n";
    return 2;
  }
  DemoReader reader;
  std::string error;
  if (!reader.Open(args[0], &error)) {
    std::cerr << "error: " << error << "\n";
    return 1;
  }

  SerializerSet serializers;
  ClassTable classes;
  bool have_tables = false, have_classes = false;
  DemoFrame frame;
  while (reader.Next(&frame)) {
    if (frame.kind == kDemSendTables && !have_tables) {
      if (!ParseSendTables(frame.body, &serializers, &error)) {
        std::cerr << "error: " << error << "\n";
        return 1;
      }
      have_tables = true;
    } else if (frame.kind == kDemClassInfo && !have_classes) {
      if (!ParseClassInfo(frame.body, &classes, &error)) {
        std::cerr << "error: " << error << "\n";
        return 1;
      }
      have_classes = true;
    }
    if (have_tables && have_classes) break;
  }
  if (!have_tables || !have_classes) {
    std::cerr << "error: demo carried no send tables or class info\n";
    return 1;
  }

  std::cout << "symbols     : " << serializers.symbols.size() << "\n"
            << "fields      : " << serializers.fields.size() << "\n"
            << "serializers : " << serializers.serializers.size() << "\n"
            << "classes     : " << classes.names.size()
            << " (max id " << classes.max_class_id << ", "
            << classes.class_id_bits() << " bits)\n";

  const std::string want = args.size() > 1 ? args[1] : "CCSPlayerPawn";
  const Serializer* s = serializers.Find(want);
  if (s == nullptr) {
    std::cout << "\nno serializer named " << want << "\n";
    return 0;
  }
  std::cout << "\n" << s->name << " v" << s->version << ", " << s->fields.size()
            << " fields:\n";
  for (int index : s->fields) {
    if (index < 0 || static_cast<std::size_t>(index) >= serializers.fields.size()) continue;
    const FieldInfo& f = serializers.fields[static_cast<std::size_t>(index)];
    std::cout << "  " << f.var_name << "  <" << f.var_type << ">";
    if (f.bit_count > 0) std::cout << " bits=" << f.bit_count;
    if (!f.encoder.empty()) std::cout << " enc=" << f.encoder;
    if (f.low != 0.0f || f.high != 0.0f) {
      std::cout << " range=[" << f.low << "," << f.high << "]";
    }
    if (f.has_child()) std::cout << " -> " << f.field_serializer_name;
    std::cout << "\n";
  }
  return 0;
}

// Decodes entity state and reports whether it actually worked. Entity decoding
// has no partial credit - it either stays in sync or produces noise - so the
// useful output is the desync count and a handful of real coordinates.
int CommandEntities(const std::vector<std::string>& args) {
  if (args.empty()) {
    std::cerr << "usage: cs2mv entities <demo file> [max packets]\n";
    return 2;
  }
  const long long limit = args.size() > 1 ? std::atoll(args[1].c_str()) : 400;

  DemoReader reader;
  std::string error;
  if (!reader.Open(args[0], &error)) {
    std::cerr << "error: " << error << "\n";
    return 1;
  }

  SerializerSet serializers;
  ClassTable classes;
  EntityDecoder decoder;
  if (std::getenv("CS2MV_TRACE") != nullptr) decoder.set_trace(1);
  if (std::getenv("CS2MV_NO_SPAWNGROUP") != nullptr) decoder.set_read_spawn_group(false);
  bool ready = false;
  long long packets = 0;
  std::string first_failure;

  DemoFrame frame;
  while (reader.Next(&frame)) {
    if (frame.kind == kDemSendTables) {
      if (!ParseSendTables(frame.body, &serializers, &error)) {
        std::cerr << "error: " << error << "\n";
        return 1;
      }
      continue;
    }
    if (frame.kind == kDemClassInfo) {
      if (!ParseClassInfo(frame.body, &classes, &error)) {
        std::cerr << "error: " << error << "\n";
        return 1;
      }
      continue;
    }
    if (!ready && !serializers.serializers.empty() && !classes.names.empty()) {
      if (!decoder.Init(serializers, classes, &error)) {
        std::cerr << "error: " << error << "\n";
        return 1;
      }
      ready = true;
    }
    if (!ready) continue;
    if (frame.kind != kDemPacket && frame.kind != kDemSignonPacket) continue;

    // Walk the packet for svc_PacketEntities (55).
    pb::Reader r(frame.body);
    pb::Slice payload;
    std::uint32_t field = 0;
    while (r.NextField(&field)) {
      if (field == 3 && r.wire_type() == pb::kLengthDelimited) payload = r.ReadBytes();
    }
    if (payload.data == nullptr) continue;

    BitReader bits(payload.data, payload.size);
    std::string buf;
    while (bits.BitsLeft() > 8) {
      const std::uint32_t kind = bits.ReadUBitVar();
      const std::uint32_t size = bits.ReadVarUInt32();
      if (!bits.ok() || size > (1u << 24)) break;
      buf.resize(size);
      if (size > 0 && !bits.ReadBytes(&buf[0], size)) break;
      if (kind != 55) continue;

      std::string packet_error;
      if (!decoder.ApplyPacket(buf, &packet_error) && first_failure.empty()) {
        first_failure = packet_error;
      }
      ++packets;
    }
    if (limit > 0 && packets >= limit) break;
  }

  std::cout << "packets decoded : " << packets << "\n"
            << "field updates   : " << decoder.updates_applied() << "\n"
            << "desyncs         : " << decoder.packets_failed() << "\n"
            << "entities alive  : " << decoder.entities().size() << "\n";
  if (!first_failure.empty()) {
    std::cout << "first failure   : " << first_failure << "\n";
  }

  // Player positions are the point of all this: cell plus offset, where a cell
  // is 512 units and the world is centred on 16384.
  std::cout << "\nplayer pawns:\n";
  int shown = 0;
  for (const auto& entry : decoder.entities()) {
    const Entity& entity = entry.second;
    if (entity.class_name != "CCSPlayerPawn") continue;
    const FieldValue* cell_x = entity.Get("CBodyComponent.m_cellX");
    const FieldValue* cell_y = entity.Get("CBodyComponent.m_cellY");
    const FieldValue* vec_x = entity.Get("CBodyComponent.m_vecX");
    const FieldValue* vec_y = entity.Get("CBodyComponent.m_vecY");
    if (cell_x == nullptr || vec_x == nullptr) continue;

    const float x = static_cast<float>(cell_x->AsInt()) * 512.0f - 16384.0f + vec_x->AsFloat();
    const float y = cell_y != nullptr && vec_y != nullptr
                        ? static_cast<float>(cell_y->AsInt()) * 512.0f - 16384.0f + vec_y->AsFloat()
                        : 0.0f;
    const FieldValue* health = entity.Get("m_iHealth");
    std::printf("  entity %-5d  x=%9.1f  y=%9.1f  health=%lld\n", entity.index, x, y,
                health != nullptr ? health->AsInt() : -1);
    if (++shown >= 12) break;
  }
  if (shown == 0) std::cout << "  (none decoded)\n";
  return first_failure.empty() ? 0 : 1;
}

int CommandFetch(const std::vector<std::string>& args, const Options& options) {
  if (args.size() < 2) {
    std::cerr << "usage: cs2mv fetch <http URL> <destination.dem>\n";
    return 2;
  }
  ResolveOptions resolve = MakeResolveOptions(options);
  resolve.progress = [](std::uint64_t done, std::uint64_t total) {
    std::fprintf(stderr, "\r%llu/%llu MB",
                 static_cast<unsigned long long>(done >> 20),
                 static_cast<unsigned long long>(total >> 20));
  };
  std::string error;
  if (!FetchDemo(args[0], args[1], resolve, &error)) {
    std::fprintf(stderr, "\n");
    std::cerr << "error: " << error << "\n";
    return 1;
  }
  std::fprintf(stderr, "\n");
  std::cout << "wrote " << args[1] << "\n";
  return 0;
}

int CommandGcRequest(const std::vector<std::string>& args) {
  if (args.empty()) {
    std::cerr << "usage: cs2mv gc-request <share code>\n";
    return 2;
  }
  ShareCode code;
  std::string error;
  if (!DecodeShareCode(args[0], &code, &error)) {
    std::cerr << "error: " << error << "\n";
    return 1;
  }
  const std::string payload = BuildMatchListRequest(code);
  std::cout << "message id : " << kMsgMatchListRequestFullGameInfo
            << " (k_EMsgGCCStrike15_v2_MatchListRequestFullGameInfo)\n"
            << "app id     : 730\n"
            << "body       : ";
  for (unsigned char c : payload) std::printf("%02x", c);
  std::cout << "\n"
            << "\nSend this body as the named GC message through a logged-in\n"
               "Steam client; the reply is k_EMsgGCCStrike15_v2_MatchList ("
            << kMsgMatchList << ").\n";
  return 0;
}

int CommandServe(const Options& options) {
  HttpServer server;
  std::string error;
  if (!server.Start(options.bind_address, options.port, &error)) {
    std::cerr << "error: " << error << "\n";
    return 1;
  }
  server.ServeStatic(options.web_root);

  server.Route("/api/decode", [](const HttpRequest& request, HttpResponse* response) {
    ShareCode code;
    std::string error;
    const std::string input = request.Param("code");
    if (!DecodeShareCode(input, &code, &error)) {
      response->SetError(400, error);
      return;
    }
    std::string canonical;
    NormalizeShareCode(input, &canonical, nullptr);
    response->SetJson("{\"shareCode\":\"" + canonical + "\",\"matchId\":\"" +
                      std::to_string(code.match_id) + "\",\"outcomeId\":\"" +
                      std::to_string(code.outcome_id) + "\",\"token\":" +
                      std::to_string(code.token) + "}");
  });

  server.Route("/api/match", [&options](const HttpRequest& request,
                                        HttpResponse* response) {
    std::string target = request.Param("code");
    if (target.empty()) target = request.Param("demo");
    if (target.empty()) {
      response->SetError(400, "pass ?code=<share code> or ?demo=<path>");
      return;
    }
    // Parsing a demo costs hundreds of megabytes; one at a time keeps a
    // browser's parallel requests from multiplying that, and keeps two
    // requests for the same match from racing on the cache file.
    static std::mutex parse_mutex;
    std::lock_guard<std::mutex> lock(parse_mutex);

    Match match;
    std::string error;
    if (!LoadMatch(target, options, &match, &error)) {
      response->SetError(404, error);
      return;
    }
    response->SetJson(MatchToJson(match, false));
  });

  // --- Steam sign-in, for the game coordinator helper.
  //
  // Only a QR challenge ever reaches the browser. The phone talks to Steam
  // directly and the helper keeps the resulting token to itself, so no
  // credential passes through this server or the page.
  static SteamLogin steam_login;

  server.Route("/api/steam/status", [&options](const HttpRequest&,
                                               HttpResponse* response) {
    const ResolveOptions resolve = MakeResolveOptions(options);
    JsonWriter w;
    w.BeginObject();
    w.Field("helperConfigured", !resolve.gc_helper.empty());
    if (resolve.gc_helper.empty()) {
      w.Field("signedIn", false);
      w.Field("account", std::string());
      w.Field("detail",
              "No Steam helper configured. Start the server with --gc-helper, "
              "or set CS2MV_GC_HELPER. Matches already downloaded in CS2 work "
              "without it.");
    } else {
      bool signed_in = false;
      std::string account;
      std::string error;
      if (SteamStatus(resolve.gc_helper, &signed_in, &account, &error)) {
        w.Field("signedIn", signed_in);
        w.Field("account", account);
        w.Field("detail", std::string());
      } else {
        w.Field("signedIn", false);
        w.Field("account", std::string());
        w.Field("detail", error);
      }
    }
    w.EndObject();
    response->SetJson(w.str());
  });

  server.Route("/api/steam/login", [&options](const HttpRequest&,
                                              HttpResponse* response) {
    const ResolveOptions resolve = MakeResolveOptions(options);
    std::string error;
    if (!steam_login.Start(resolve.gc_helper, &error)) {
      response->SetError(400, error);
      return;
    }
    response->SetJson("{\"started\":true}");
  });

  server.Route("/api/steam/login/status", [](const HttpRequest&,
                                             HttpResponse* response) {
    const SteamLogin::Snapshot snapshot = steam_login.Get();
    JsonWriter w;
    w.BeginObject();
    w.Field("state", SteamLogin::StateName(snapshot.state));
    w.Field("qrPng", snapshot.qr_png);
    w.Field("qrUrl", snapshot.qr_url);
    w.Field("account", snapshot.account);
    w.Field("message", snapshot.message);
    w.EndObject();
    response->SetJson(w.str());
  });

  server.Route("/api/steam/logout", [&options](const HttpRequest&,
                                               HttpResponse* response) {
    const ResolveOptions resolve = MakeResolveOptions(options);
    std::string error;
    steam_login.Cancel();
    if (!SteamLogout(resolve.gc_helper, &error)) {
      response->SetError(400, error);
      return;
    }
    response->SetJson("{\"signedIn\":false}");
  });

  // Demos CS2 has already downloaded, newest first. No Steam involved: this is
  // just what is on disk, labelled by reading each demo's first frame.
  server.Route("/api/matches", [](const HttpRequest&, HttpResponse* response) {
    JsonWriter w;
    w.BeginObject();
    w.Key("matches");
    w.BeginArray();
    for (const LocalDemo& demo : ListDownloadedDemos()) {
      DemoSummary summary;
      std::string ignored;
      ReadDemoSummary(demo.path, &summary, &ignored);

      w.BeginObject();
      w.Field("path", demo.path);
      w.FieldId("id", demo.id);
      w.Field("map", summary.map_name);
      w.Field("server", summary.server_name);
      w.Field("sizeBytes", static_cast<long long>(demo.size_bytes));
      w.Field("modified", static_cast<long long>(demo.modified_unix));
      w.EndObject();
    }
    w.EndArray();
    w.EndObject();
    response->SetJson(w.str());
  });

  // The account's match history, straight from the game coordinator - the same
  // list CS2 shows under Watch > Your Matches. Needs the signed-in helper, and
  // takes a few seconds, so the page asks for it separately.
  server.Route("/api/matches/remote", [&options](const HttpRequest&,
                                                 HttpResponse* response) {
    const ResolveOptions resolve = MakeResolveOptions(options);
    std::vector<RemoteMatch> matches;
    std::string error;
    if (!ListRecentMatches(resolve.gc_helper, &matches, &error)) {
      response->SetError(400, error);
      return;
    }

    JsonWriter w;
    w.BeginObject();
    w.Key("matches");
    w.BeginArray();
    for (const RemoteMatch& match : matches) {
      w.BeginObject();
      w.Field("shareCode", EncodeShareCode(match.code));
      w.FieldId("matchId", match.code.match_id);
      w.FieldId("outcomeId", match.code.outcome_id);
      w.Field("matchTime", match.match_time);
      w.Field("expired", match.demo_url.empty());
      w.EndObject();
    }
    w.EndArray();
    w.EndObject();
    response->SetJson(w.str());
  });

  // Hands a demo to CS2 for playback. This starts a program, so unlike the
  // read-only routes it refuses a cross-origin caller: a page on the open web
  // can POST to loopback, and should not get to launch things here.
  server.Route("/api/watch", [](const HttpRequest& request,
                                HttpResponse* response) {
    auto origin = request.headers.find("origin");
    if (origin != request.headers.end() &&
        origin->second.rfind("http://127.0.0.1:", 0) != 0 &&
        origin->second.rfind("http://localhost:", 0) != 0) {
      response->SetError(403, "cross-origin requests cannot start playback");
      return;
    }

    const std::string demo = request.Param("demo");
    if (demo.empty()) {
      response->SetError(400, "pass ?demo=<path to a .dem>");
      return;
    }
    const bool dry_run = !request.Param("dry").empty();

    std::string command;
    std::string error;
    bool requires_copy = false;
    if (!LaunchCs2Playback(demo, dry_run, &command, &requires_copy, &error)) {
      response->SetError(400, error);
      return;
    }
    JsonWriter w;
    w.BeginObject();
    w.Field("launched", !dry_run);
    w.Field("command", command);
    w.Field("requiresCopy", requires_copy);
    w.EndObject();
    response->SetJson(w.str());
  });

  server.Route("/api/index", [&options](const HttpRequest&, HttpResponse* response) {
    const ResolveOptions resolve = MakeResolveOptions(options);
    const std::string index_path = resolve.index_path.empty()
                                       ? resolve.cache_dir + "/index.txt"
                                       : resolve.index_path;
    DemoIndex index;
    std::string error;
    index.Load(index_path, &error);
    std::string json = "{\"indexPath\":\"";
    for (char c : index_path) {
      if (c == '\\' || c == '"') json += '\\';
      json += c;
    }
    json += "\",\"entries\":[";
    bool first = true;
    for (const auto& entry : index.entries()) {
      if (!first) json += ',';
      first = false;
      json += "{\"key\":\"" + entry.first + "\",\"location\":\"";
      for (char c : entry.second) {
        if (c == '\\' || c == '"') json += '\\';
        json += c;
      }
      json += "\"}";
    }
    json += "]}";
    response->SetJson(json);
  });

  std::cout << "cs2-match-viewer listening on http://" << options.bind_address
            << ":" << server.port() << "\n"
            << "serving the UI from " << options.web_root << "\n"
            << "press Ctrl+C to stop\n";
  server.Serve();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> args(argv + 1, argv + argc);
  if (args.empty()) {
    PrintUsage();
    return 0;
  }

  const std::string command = args[0];
  args.erase(args.begin());

  Options options;
  std::string error;
  if (!ParseFlags(&args, &options, &error)) {
    std::cerr << "error: " << error << "\n";
    return 2;
  }

  if (command == "serve") return CommandServe(options);
  if (command == "decode") return CommandDecode(args);
  if (command == "encode") return CommandEncode(args);
  if (command == "add") return CommandAdd(args, options);
  if (command == "list") return CommandList(options);
  if (command == "parse") return CommandParse(args, options);
  if (command == "inspect") return CommandInspect(args);
  if (command == "schema") return CommandSchema(args);
  if (command == "entities") return CommandEntities(args);
  if (command == "fetch") return CommandFetch(args, options);
  if (command == "gc-request") return CommandGcRequest(args);
  if (command == "help" || command == "--help" || command == "-h") {
    PrintUsage();
    return 0;
  }

  std::cerr << "unknown command '" << command << "'\n\n";
  PrintUsage();
  return 2;
}
