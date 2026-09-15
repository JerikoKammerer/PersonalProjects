#include "cs2mv/entities.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cs2mv/protobuf.h"

namespace cs2mv {
namespace {

bool Err(std::string* error, const std::string& msg) {
  if (error != nullptr) *error = msg;
  return false;
}

bool StartsWith(const std::string& s, const char* prefix) {
  return s.rfind(prefix, 0) == 0;
}

std::string Trim(const std::string& s) {
  std::size_t a = 0;
  std::size_t b = s.size();
  while (a < b && (s[a] == ' ' || s[a] == '\t')) ++a;
  while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t')) --b;
  return s.substr(a, b - a);
}

// Quantisation flags, from the field's encode_flags.
constexpr int kRoundDown = 1;
constexpr int kRoundUp = 2;
constexpr int kEncodeZero = 4;
constexpr int kEncodeIntegers = 8;

// Source 2's coordinate encoding: two selector bits say whether an integer
// and/or fractional part follows.
float ReadCoord(BitReader* bits) {
  const bool has_int = bits->ReadBit();
  const bool has_fraction = bits->ReadBit();
  if (!has_int && !has_fraction) return 0.0f;

  const bool negative = bits->ReadBit();
  int integer = 0;
  int fraction = 0;
  if (has_int) integer = static_cast<int>(bits->ReadBits(14)) + 1;
  if (has_fraction) fraction = static_cast<int>(bits->ReadBits(5));

  float value = static_cast<float>(integer) + static_cast<float>(fraction) * (1.0f / 32.0f);
  return negative ? -value : value;
}

// A component of a unit vector: a sign and eleven bits of magnitude.
float ReadNormal(BitReader* bits) {
  const bool negative = bits->ReadBit();
  const std::uint32_t raw = bits->ReadBits(11);
  float value = static_cast<float>(raw) * (1.0f / ((1 << 11) - 1));
  return negative ? -value : value;
}

// An angle packed into `count` bits over a full turn.
float ReadAngle(BitReader* bits, int count) {
  const double raw = static_cast<double>(bits->ReadBits(count));
  return static_cast<float>(raw * 360.0 / static_cast<double>(1ull << count));
}

float BitsToFloat(std::uint32_t raw) {
  float f = 0.0f;
  std::memcpy(&f, &raw, sizeof(f));
  return f;
}

// Struct-valued fields that stand for one struct rather than an array of
// them. Most are spelled with a pointer in the schema; these are not, and are
// known by name instead.
bool IsPointerType(const std::string& base) {
  static const char* const kNames[] = {
      "CBodyComponent",     "CLightComponent",   "CPhysicsComponent",
      "CRenderComponent",   "CEntityIdentity",   "PhysicsRagdollPose_t",
      "CPlayerLocalData",   "CPlayer_CameraServices",
  };
  for (const char* name : kNames) {
    if (base == name) return true;
  }
  return false;
}

bool IsVectorType(const std::string& base) {
  return base == "CUtlVector" || base == "CNetworkUtlVectorBase" ||
         base == "CUtlVectorEmbeddedNetworkVar";
}

}  // namespace

TypeSpec ParseTypeSpec(const std::string& type) {
  TypeSpec spec;
  std::string s = Trim(type);
  if (!s.empty() && s.back() == ']') {
    const std::size_t open = s.rfind('[');
    if (open != std::string::npos) {
      spec.count = std::atoi(s.c_str() + open + 1);
      s = Trim(s.substr(0, open));
    }
  }
  if (!s.empty() && s.back() == '*') {
    spec.pointer = true;
    s = Trim(s.substr(0, s.size() - 1));
  }
  const std::size_t lt = s.find('<');
  if (lt != std::string::npos) {
    const std::size_t gt = s.rfind('>');
    if (gt != std::string::npos && gt > lt) {
      spec.generic = Trim(s.substr(lt + 1, gt - lt - 1));
    }
    s = Trim(s.substr(0, lt));
  }
  spec.base = s;
  return spec;
}

// ------------------------------------------------------------- field decoder

FieldDecoder FieldDecoder::Bool() {
  FieldDecoder d;
  d.kind_ = kBool;
  d.description_ = "bool";
  return d;
}

FieldDecoder FieldDecoder::VarUInt() {
  FieldDecoder d;
  d.kind_ = kVarUInt32;
  d.description_ = "count";
  return d;
}

FieldDecoder FieldDecoder::Polymorphic() {
  FieldDecoder d;
  d.kind_ = kPolymorphic;
  d.description_ = "polymorphic";
  return d;
}

FieldDecoder::FieldDecoder(const std::string& type, const FieldInfo& info) {
  description_ = type;
  if (!info.encoder.empty()) description_ += " " + info.encoder;

  if (type == "bool") {
    kind_ = kBool;
  } else if (type == "CUtlString" || type == "CUtlSymbolLarge" ||
             type == "CGlobalSymbol" || type == "char") {
    kind_ = kString;
  } else if (info.encoder == "fixed64") {
    // Sixty four literal bits rather than a varint.
    kind_ = kFixed64;
  } else if (type == "int64") {
    kind_ = kVarInt64;
  } else if (type == "HSequence") {
    kind_ = kSequence;
  } else if (type == "int8" || type == "int16" || type == "int32") {
    kind_ = kVarInt32;
  } else if (type == "float32" || type == "GameTime_t" ||
             type == "CNetworkedQuantizedFloat") {
    kind_ = kFloats;
    components_ = 1;
    InitFloat(info);
  } else if (type == "Vector" || type == "VectorWS") {
    if (info.encoder == "normal") {
      kind_ = kVectorNormal;
    } else {
      kind_ = kFloats;
      components_ = 3;
      InitFloat(info);
    }
  } else if (type == "Vector2D") {
    kind_ = kFloats;
    components_ = 2;
    InitFloat(info);
  } else if (type == "Vector4D" || type == "Quaternion") {
    kind_ = kFloats;
    components_ = 4;
    InitFloat(info);
  } else if (type == "QAngle") {
    kind_ = kQAngle;
    bits_ = info.bit_count;
    if (info.encoder == "qangle_pitch_yaw") {
      qangle_ = kQAnglePitchYaw;
    } else if (info.encoder == "qangle_precise") {
      qangle_ = kQAnglePrecise;
    } else if (info.bit_count >= 32) {
      // Three plain floats. Established against an aim punch angle whose
      // three components read as sane floats only at this alignment, with a
      // place name string sitting exactly after them.
      kind_ = kFloats;
      components_ = 3;
      float_ = kNoScale;
    } else if (info.bit_count != 0) {
      qangle_ = kQAngleFixedBits;
    } else {
      qangle_ = kQAngleCoord;
    }
  } else if (type == "CUtlBinaryBlock") {
    kind_ = kBlob;
  } else if (type == "CTransform") {
    // Its bit layout is not established. Reading it wrong would be worse
    // than refusing, because the refusal at least says where it happened.
    kind_ = kUnsupported;
  } else {
    // Everything else - integers, enums, handles, tokens, colours, ticks,
    // resource ids - is an unsigned varint. Some of those are 64 bits wide
    // (a ResourceId_t is a hash, a uint64 a mask) and nothing in the schema
    // says which, so every one is read as 64 bits: a value that fits in 32
    // decodes identically, and one that does not no longer desynchronises.
    kind_ = kVarUInt64;
  }
}

void FieldDecoder::InitFloat(const FieldInfo& info) {
  // Two fields lie about themselves. The schema declares them as plain
  // float32 with no encoder, but the game always writes them as simulation
  // time, and reading them as raw floats desynchronises the stream on the
  // very first entity - CWorld carries m_flSimulationTime as its second field.
  std::string encoder = info.encoder;
  if (info.var_name == "m_flSimulationTime" || info.var_name == "m_flAnimTime") {
    encoder = "simtime";
  }

  if (encoder == "coord") {
    float_ = kCoord;
    return;
  }
  if (encoder == "simtime" || encoder == "runetime") {
    float_ = kSimulationTime;
    return;
  }
  if (encoder == "normal") {
    float_ = kNormal;
    return;
  }
  if (info.bit_count <= 0 || info.bit_count >= 32) {
    float_ = kNoScale;
    return;
  }

  float_ = kQuantized;
  bits_ = info.bit_count;
  flags_ = info.encode_flags;
  float low = info.low;
  float high = info.high;
  // A range the schema leaves out is the unit interval.
  if (low == 0.0f && high == 0.0f) high = 1.0f;

  // What the flags do on the wire was settled against real packets rather
  // than taken from a description, because the bit budget has to be exact:
  //
  //  * RoundDown and RoundUp spend no bit. They shrink the range by one step
  //    so that its low (or high) end lands exactly on a code: a friction of
  //    [0,4] in 8 bits then reads 1.0 rather than 1.0039, and a max speed of
  //    [0,2048] in 12 bits reads 260 rather than 260.06.
  //  * EncodeZero spends one presence bit, unless zero already lands exactly
  //    on a code, in which case it is redundant and the game omits it. That
  //    covers a stashed speed declared [0,16384] (zero is code 0), and a
  //    playback rate declared [-4,12] with RoundDown (zero is code 64), but
  //    not a view offset declared [-64,64] in 10 bits.
  //  * EncodeIntegers overrides everything else.
  if ((flags_ & kRoundDown) && (flags_ & kRoundUp)) flags_ &= ~(kRoundDown | kRoundUp);
  if (low > 0.0f || high < 0.0f) flags_ &= ~kEncodeZero;
  if (flags_ & kEncodeIntegers) flags_ &= ~(kRoundUp | kRoundDown | kEncodeZero);

  int steps = 1 << bits_;
  if (flags_ & kRoundDown) {
    high -= (high - low) / static_cast<float>(steps);
  } else if (flags_ & kRoundUp) {
    low += (high - low) / static_cast<float>(steps);
  }
  if (flags_ & kEncodeIntegers) {
    float delta = high - low;
    if (delta < 1.0f) delta = 1.0f;
    const int span = 1 << static_cast<int>(std::ceil(std::log2(delta)));
    int bits = bits_;
    while ((1 << bits) <= span) ++bits;
    if (bits > bits_) {
      bits_ = bits;
      steps = 1 << bits_;
    }
    high = low + static_cast<float>(span) - static_cast<float>(span) / static_cast<float>(steps);
  }

  low_ = low;
  high_ = high;
  interval_ = (steps > 1) ? (high - low) / static_cast<float>(steps - 1) : 0.0f;

  if ((flags_ & kEncodeZero) && interval_ > 0.0f) {
    const float code = -low_ / interval_;
    if (std::fabs(code - std::round(code)) < 1e-4f) flags_ &= ~kEncodeZero;
  }
}

float FieldDecoder::DecodeFloat(BitReader* bits) const {
  switch (float_) {
    case kCoord:
      return ReadCoord(bits);
    case kNormal:
      return ReadNormal(bits);
    case kSimulationTime:
      return static_cast<float>(bits->ReadVarUInt32()) * (1.0f / 64.0f);
    case kQuantized: {
      if ((flags_ & kEncodeZero) && bits->ReadBit()) return 0.0f;
      const std::uint32_t raw = bits->ReadBits(bits_);
      return low_ + static_cast<float>(raw) * interval_;
    }
    case kNoScale:
    default:
      return BitsToFloat(bits->ReadBits(32));
  }
}

FieldValue FieldDecoder::Decode(BitReader* bits) const {
  FieldValue value;
  switch (kind_) {
    case kBool:
      value.kind = FieldValue::kUInt;
      value.u = bits->ReadBit() ? 1 : 0;
      break;
    case kVarUInt32:
      value.kind = FieldValue::kUInt;
      value.u = bits->ReadVarUInt32();
      break;
    case kVarInt32:
      // Read wide for the same reason as the unsigned case.
      value.kind = FieldValue::kInt;
      value.i = bits->ReadVarInt64();
      break;
    case kVarUInt64:
      value.kind = FieldValue::kUInt;
      value.u = bits->ReadVarUInt64();
      break;
    case kVarInt64:
      value.kind = FieldValue::kInt;
      value.i = bits->ReadVarInt64();
      break;
    case kFixed64:
      value.kind = FieldValue::kUInt;
      value.u = static_cast<std::uint64_t>(bits->ReadBits(32)) |
                (static_cast<std::uint64_t>(bits->ReadBits(32)) << 32);
      break;
    case kSequence:
      value.kind = FieldValue::kUInt;
      value.u = bits->ReadVarUInt64() - 1;
      break;
    case kString: {
      // Null terminated, one byte at a time.
      value.kind = FieldValue::kString;
      for (int i = 0; i < 1024; ++i) {
        const std::uint32_t c = bits->ReadBits(8);
        if (!bits->ok() || c == 0) break;
        value.s.push_back(static_cast<char>(c));
      }
      break;
    }
    case kFloats:
      value.kind = FieldValue::kFloat;
      for (int i = 0; i < components_; ++i) value.v[i] = DecodeFloat(bits);
      break;
    case kVectorNormal: {
      value.kind = FieldValue::kFloat;
      const bool has_x = bits->ReadBit();
      const bool has_y = bits->ReadBit();
      const float x = has_x ? ReadNormal(bits) : 0.0f;
      const float y = has_y ? ReadNormal(bits) : 0.0f;
      const bool negative_z = bits->ReadBit();
      const float sum = x * x + y * y;
      float z = sum < 1.0f ? std::sqrt(1.0f - sum) : 0.0f;
      if (negative_z) z = -z;
      value.v[0] = x;
      value.v[1] = y;
      value.v[2] = z;
      break;
    }
    case kQAngle: {
      value.kind = FieldValue::kFloat;
      switch (qangle_) {
        case kQAnglePitchYaw:
          value.v[0] = ReadAngle(bits, bits_);
          value.v[1] = ReadAngle(bits, bits_);
          break;
        case kQAnglePrecise: {
          const bool has_pitch = bits->ReadBit();
          const bool has_yaw = bits->ReadBit();
          const bool has_roll = bits->ReadBit();
          if (has_pitch) value.v[0] = ReadAngle(bits, 20) - 180.0f;
          if (has_yaw) value.v[1] = ReadAngle(bits, 20) - 180.0f;
          if (has_roll) value.v[2] = ReadAngle(bits, 20) - 180.0f;
          break;
        }
        case kQAngleFixedBits:
          for (int i = 0; i < 3; ++i) value.v[i] = ReadAngle(bits, bits_);
          break;
        case kQAngleCoord: {
          const bool has_pitch = bits->ReadBit();
          const bool has_yaw = bits->ReadBit();
          const bool has_roll = bits->ReadBit();
          if (has_pitch) value.v[0] = ReadCoord(bits);
          if (has_yaw) value.v[1] = ReadCoord(bits);
          if (has_roll) value.v[2] = ReadCoord(bits);
          break;
        }
      }
      break;
    }
    case kBlob: {
      // A byte count, then the bytes. Established against a chicken's
      // serialised pose recipe, which is the only place this type appears.
      value.kind = FieldValue::kString;
      const std::uint64_t count = bits->ReadVarUInt64();
      if (count > (1u << 20)) {
        bits->ReadBits(64);  // an absurd length; poison the reader
        break;
      }
      for (std::uint64_t i = 0; i < count && bits->ok(); ++i) {
        value.s.push_back(static_cast<char>(bits->ReadBits(8)));
      }
      break;
    }
    case kPolymorphic:
      // Established against a game rules baseline: after the presence bit,
      // a ubitvar picks the type, with 0 meaning the declared one.
      value.kind = FieldValue::kInt;
      value.i = bits->ReadBit() ? static_cast<long long>(bits->ReadUBitVar()) : -1;
      break;
    case kUnsupported:
      break;
  }
  return value;
}

// ---------------------------------------------------------------- flattening

void EntityDecoder::Flatten(const SerializerSet& set,
                            const Serializer& serializer,
                            const std::string& prefix, int depth,
                            std::vector<FlatField>* out) {
  // Seven is the deepest a field path can address; guard against a schema that
  // refers to itself.
  if (depth > 6) return;

  for (int index : serializer.fields) {
    FlatField flat;
    if (index < 0 || static_cast<std::size_t>(index) >= set.fields.size()) {
      out->push_back(flat);
      continue;
    }
    const FieldInfo& info = set.fields[static_cast<std::size_t>(index)];
    const TypeSpec spec = ParseTypeSpec(info.var_type);
    flat.name = prefix.empty() ? info.var_name : prefix + "." + info.var_name;

    if (info.has_child()) {
      // A struct, or an array of them. The schema spells most single structs
      // with a pointer; a bare struct type that is not a vector is one too.
      const bool single = spec.pointer || IsPointerType(spec.base) ||
                          (spec.count == 0 && !IsVectorType(spec.base));
      flat.model = single ? FlatField::kFixedTable : FlatField::kVariableTable;
      flat.self = single ? FieldDecoder::Bool() : FieldDecoder::VarUInt();
      const Serializer* child =
          set.Find(info.field_serializer_name, info.field_serializer_version);
      if (child != nullptr) {
        Flatten(set, *child, flat.name, depth + 1, &flat.children);
      }
      if (single && !info.polymorphic_types.empty()) {
        flat.self = FieldDecoder::Polymorphic();
        for (const FieldInfo::Polymorphic& type : info.polymorphic_types) {
          std::vector<FlatField> members;
          const Serializer* alt = set.Find(type.serializer_name, type.version);
          if (alt != nullptr) Flatten(set, *alt, flat.name, depth + 1, &members);
          flat.alternatives.push_back(std::move(members));
        }
      }
    } else if (spec.count > 0 && spec.base != "char") {
      // T[N]. char[N] looks the same but is a string, decoded as one.
      flat.model = FlatField::kFixedArray;
      flat.decoder = FieldDecoder(spec.base, info);
      flat.self = flat.decoder;
    } else if (IsVectorType(spec.base)) {
      // A vector of values: its element type is the generic parameter, while
      // the encoding parameters are the field's own.
      flat.model = FlatField::kVariableArray;
      flat.decoder = FieldDecoder(ParseTypeSpec(spec.generic).base, info);
      flat.self = FieldDecoder::VarUInt();
    } else {
      flat.model = FlatField::kSimple;
      flat.decoder = FieldDecoder(spec.base, info);
      flat.self = flat.decoder;
    }
    out->push_back(std::move(flat));
  }
}

bool EntityDecoder::Init(const SerializerSet& serializers,
                         const ClassTable& classes, std::string* error) {
  for (const auto& entry : classes.names) {
    const Serializer* serializer = serializers.Find(entry.second);
    if (serializer == nullptr) continue;
    FlatClass flat;
    flat.name = entry.second;
    Flatten(serializers, *serializer, "", 0, &flat.fields);
    by_class_id_[entry.first] = std::move(flat);
  }
  class_id_bits_ = classes.class_id_bits();
  if (by_class_id_.empty()) {
    return Err(error, "no class resolved against the schema");
  }
  return true;
}

// Walks a field path down the class's tree. Each level's meaning depends on
// the field it lands on: a member index for a struct, an element index for an
// array, and for a path that stops at an array or struct, the field's own
// "self" value (a size, or a presence bit).
const FieldDecoder* EntityDecoder::Resolve(const FlatClass& flat,
                                           const Entity& entity,
                                           const FieldPath& path,
                                           std::string* name) const {
  const std::vector<FlatField>* fields = &flat.fields;
  int level = 0;
  for (;;) {
    const int index = path.path[level];
    if (index < 0 || static_cast<std::size_t>(index) >= fields->size()) return nullptr;
    const FlatField* field = &(*fields)[static_cast<std::size_t>(index)];
    *name = field->name;
    ++level;
    const bool ends_here = level > path.last;

    switch (field->model) {
      case FlatField::kSimple:
        return ends_here ? &field->decoder : nullptr;

      case FlatField::kFixedArray:
      case FlatField::kVariableArray:
        if (ends_here) return &field->self;
        *name += "." + std::to_string(path.path[level]);
        return level == path.last ? &field->decoder : nullptr;

      case FlatField::kFixedTable:
        if (ends_here) return &field->self;
        fields = &field->children;
        if (!field->alternatives.empty()) {
          auto variant = entity.variants.find(field->name);
          if (variant != entity.variants.end() && variant->second > 0) {
            const std::size_t k = static_cast<std::size_t>(variant->second - 1);
            if (k >= field->alternatives.size()) return nullptr;
            fields = &field->alternatives[k];
          }
        }
        continue;

      case FlatField::kVariableTable:
        if (ends_here) return &field->self;
        *name += "." + std::to_string(path.path[level]);
        ++level;
        if (level > path.last) return &field->self;
        fields = &field->children;
        continue;
    }
    return nullptr;
  }
}

// ------------------------------------------------------------------- updates

bool EntityDecoder::ApplyUpdate(Entity* entity, BitReader* bits, bool trace,
                                std::string* error) {
  auto flat = by_class_id_.find(entity->class_id);
  if (flat == by_class_id_.end()) {
    return Err(error, "entity has no flattened class");
  }

  std::vector<FieldPath> paths;
  if (!ReadFieldPaths(bits, &paths)) {
    return Err(error, "field path stream ended badly in " + flat->second.name);
  }
  for (const FieldPath& path : paths) {
    std::string name;
    const FieldDecoder* decoder = Resolve(flat->second, *entity, path, &name);
    if (decoder == nullptr) {
      return Err(error, "field path " + path.ToString() + " does not resolve in " +
                            flat->second.name);
    }
    if (decoder->unsupported()) {
      return Err(error, "no decoder for " + name + " <" + decoder->description() +
                            "> in " + flat->second.name);
    }
    const std::size_t at = bits->BitsConsumed();
    const FieldValue decoded = decoder->Decode(bits);
    if (trace) {
      char text[128];
      switch (decoded.kind) {
        case FieldValue::kFloat:
          std::snprintf(text, sizeof(text), "%g %g %g %g", decoded.v[0],
                        decoded.v[1], decoded.v[2], decoded.v[3]);
          break;
        case FieldValue::kInt:
          std::snprintf(text, sizeof(text), "%lld", decoded.i);
          break;
        case FieldValue::kUInt:
          std::snprintf(text, sizeof(text), "%llu", decoded.u);
          break;
        default:
          std::snprintf(text, sizeof(text), "\"%s\"", decoded.s.c_str());
          break;
      }
      std::printf("      @%-7zu %-12s %-46s = %s  <%s>\n", at,
                  path.ToString().c_str(), name.c_str(), text,
                  decoder->description().c_str());
    }
    entity->values[name] = decoded;
    if (decoder->polymorphic()) {
      entity->variants[name] = decoded.i < 0 ? 0 : static_cast<int>(decoded.i);
    }
    ++updates_applied_;
    if (!bits->ok()) {
      return Err(error, "bit stream exhausted at " + name + " in " + flat->second.name);
    }
  }
  return true;
}

bool EntityDecoder::CheckBaseline(int class_id, Entity* scratch, int* bits_left,
                                  std::string* error) {
  auto baseline = baselines_.find(class_id);
  if (baseline == baselines_.end()) return Err(error, "no baseline");
  auto flat = by_class_id_.find(class_id);
  if (flat == by_class_id_.end()) return Err(error, "class is not in the schema");

  scratch->class_id = class_id;
  scratch->class_name = flat->second.name;
  BitReader bits(baseline->second.data(), baseline->second.size());
  const long long applied = updates_applied_;
  const bool ok = ApplyUpdate(scratch, &bits, trace_ > 0, error);
  updates_applied_ = applied;
  *bits_left = static_cast<int>(bits.BitsLeft());
  return ok;
}

// ------------------------------------------------------------ packet entities

bool EntityDecoder::ApplyPacket(const std::string& message, std::string* error) {
  // CSVCMsg_PacketEntities: entity_data = 7 carries the payload; the newer
  // serialized_entities = 13 exists alongside it but is a much smaller,
  // different thing. Taking the wrong one decodes noise.
  pb::Slice data;
  int updated_entries = 0;
  bool is_delta = false;
  {
    pb::Reader r(message);
    std::uint32_t field = 0;
    while (r.NextField(&field)) {
      if (r.wire_type() == pb::kLengthDelimited) {
        const pb::Slice s = r.ReadBytes();
        if (trace_ > 0) std::printf("    field %-3u len=%zu\n", field, s.size);
        if (field == 7) data = s;
        continue;
      }
      const std::uint64_t v = r.ReadVarint();
      if (trace_ > 0) {
        std::printf("    field %-3u = %llu\n", field, static_cast<unsigned long long>(v));
      }
      if (field == 2) updated_entries = static_cast<int>(v);
      if (field == 3) is_delta = v != 0;
    }
  }
  if (data.data == nullptr || updated_entries <= 0) return true;

  BitReader bits(data.data, data.size);
  int entity_index = -1;
  const bool trace = trace_ > 0;
  if (trace_ > 0) --trace_;
  if (trace) {
    std::printf("  packet: %d updates, %zu bytes, delta=%d\n", updated_entries,
                data.size, is_delta ? 1 : 0);
  }

  for (int i = 0; i < updated_entries; ++i) {
    // Index is delta encoded from the previous entity.
    entity_index += 1 + static_cast<int>(bits.ReadUBitVar());
    if (!bits.ok() || entity_index < 0 || entity_index > 16384) {
      ++packets_failed_;
      return Err(error, "entity index out of range after update " + std::to_string(i));
    }

    const bool leaving = bits.ReadBit();
    const bool creating = bits.ReadBit();

    if (leaving) {
      // Leave, and possibly delete. The second bit distinguishes them.
      if (trace) {
        std::printf("   -%s idx=%d\n", creating ? "delete" : "leave", entity_index);
      }
      if (creating) entities_.erase(entity_index);
      continue;
    }

    Entity* entity = nullptr;
    if (creating) {
      const int class_id = static_cast<int>(bits.ReadBits(class_id_bits_));
      const int serial = static_cast<int>(bits.ReadBits(17));
      if (read_spawn_group_) bits.ReadVarUInt32();

      Entity fresh;
      fresh.index = entity_index;
      fresh.serial = serial;
      fresh.class_id = class_id;
      auto flat = by_class_id_.find(class_id);
      if (flat == by_class_id_.end()) {
        ++packets_failed_;
        return Err(error, "unknown class id " + std::to_string(class_id));
      }
      fresh.class_name = flat->second.name;
      if (trace) {
        std::printf("   +create idx=%-5d class=%-3d %s @%zu\n", entity_index,
                    class_id, fresh.class_name.c_str(), bits.BitsConsumed());
      }
      entities_[entity_index] = std::move(fresh);
      entity = &entities_[entity_index];

      // The class baseline first, then whatever this creation changes. The
      // baseline is its own bit stream, so failing to read it costs values,
      // not synchronisation.
      auto baseline = baselines_.find(class_id);
      if (baseline != baselines_.end()) {
        BitReader base_bits(baseline->second.data(), baseline->second.size());
        std::string base_error;
        if (!ApplyUpdate(entity, &base_bits, false, &base_error) && trace) {
          std::printf("    (baseline for %s failed: %s)\n",
                      entity->class_name.c_str(), base_error.c_str());
        }
      }
    } else {
      auto it = entities_.find(entity_index);
      if (it == entities_.end()) {
        ++packets_failed_;
        return Err(error, "delta for entity " + std::to_string(entity_index) +
                              ", which does not exist");
      }
      entity = &it->second;
      if (trace) {
        std::printf("   ~update idx=%-5d %s @%zu\n", entity_index,
                    entity->class_name.c_str(), bits.BitsConsumed());
      }
    }

    if (!ApplyUpdate(entity, &bits, trace, error)) {
      ++packets_failed_;
      return false;
    }
  }
  return true;
}

}  // namespace cs2mv
