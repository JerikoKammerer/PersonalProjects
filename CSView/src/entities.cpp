#include "cs2mv/entities.h"

#include <cmath>
#include <cstring>

#include <cstdio>

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

float ReadNormal(BitReader* bits) {
  const bool negative = bits->ReadBit();
  const std::uint32_t raw = bits->ReadBits(11);
  float value = static_cast<float>(raw) * (1.0f / ((1 << 11) - 1));
  return negative ? -value : value;
}

float BitsToFloat(std::uint32_t raw) {
  float f = 0.0f;
  std::memcpy(&f, &raw, sizeof(f));
  return f;
}

}  // namespace

FieldDecoder::FieldDecoder(const FieldInfo& info) {
  const std::string& type = info.var_type;

  // Two fields lie about themselves. The schema declares them as plain
  // float32 with no encoder, but the game always writes them as simulation
  // time, and reading them as raw floats desynchronises the stream on the
  // very first entity - CWorld carries m_flSimulationTime as its second field.
  std::string encoder = info.encoder;
  if (info.var_name == "m_flSimulationTime" || info.var_name == "m_flAnimTime") {
    encoder = "simtime";
  }

  // Strings first: they are the only variable length values here.
  if (type == "CUtlString" || type == "CUtlSymbolLarge" ||
      type == "CGlobalSymbol" || StartsWith(type, "char[")) {
    kind_ = kString;
    return;
  }

  if (type == "bool") {
    kind_ = kBool;
    return;
  }

  // 64 bit values. A "fixed64" encoder means the bits are literal, not varint.
  if (type == "uint64" || StartsWith(type, "CStrongHandle")) {
    kind_ = (encoder == "fixed64") ? kFixed64 : kVarUInt;
    return;
  }
  if (type == "int64") {
    kind_ = kVarInt;
    return;
  }

  // Floats, and everything built out of them. A vector is several floats back
  // to back: only the first is kept, but every component must still be read,
  // because leaving 64 bits of a Vector on the wire desynchronises everything
  // after it.
  const bool is_float =
      type == "float32" || type == "GameTime_t" ||
      type == "CNetworkedQuantizedFloat" || type == "Vector" ||
      type == "Vector2D" || type == "Vector4D" || type == "QAngle" ||
      type == "Quaternion" || type == "CTransform";
  if (is_float) {
    if (encoder == "coord") {
      kind_ = kFloatCoord;
    } else if (encoder == "simtime" || encoder == "runetime") {
      kind_ = kFloatSimulationTime;
    } else if (encoder == "normal") {
      kind_ = kNormal;
    } else if (info.bit_count <= 0 || info.bit_count >= 32) {
      kind_ = kFloatNoScale;
    } else {
      kind_ = kFloatQuantized;
    }

    if (type == "Vector" || type == "QAngle") {
      components_ = 3;
    } else if (type == "Vector2D") {
      components_ = 2;
    } else if (type == "Vector4D" || type == "Quaternion") {
      components_ = 4;
    }

    bits_ = info.bit_count;
    flags_ = info.encode_flags;
    low_ = info.low;
    high_ = info.high;
    no_scale_ = (info.bit_count <= 0 || info.bit_count >= 32);

    if (kind_ == kFloatQuantized) {
      // Round up and round down each spend one step, and both being set is a
      // contradiction. Flags that cannot apply to this range are dropped, so
      // that the bit budget below is right.
      if ((flags_ & kRoundDown) && (flags_ & kRoundUp)) flags_ &= ~kRoundUp;
      if (low_ == 0.0f && (flags_ & kRoundDown)) flags_ &= ~kRoundDown;
      if (high_ == 0.0f && (flags_ & kRoundUp)) flags_ &= ~kRoundUp;
      if (low_ > 0.0f || high_ < 0.0f) flags_ &= ~kEncodeZero;

      float low = low_;
      float high = high_;
      int bit_count = bits_;
      float range = high - low;

      if (flags_ & kEncodeIntegers) {
        float delta = high - low;
        if (delta < 1.0f) delta = 1.0f;
        const float log2delta = std::ceil(std::log2(delta));
        const float span = std::pow(2.0f, log2delta);
        int needed = bit_count;
        while ((1 << needed) < static_cast<int>(span)) ++needed;
        if (needed > bit_count) bit_count = needed;
        const float steps = static_cast<float>(1 << bit_count);
        high = low + span - (span / steps);
        range = high - low;
      }

      const int steps = 1 << bit_count;
      if (flags_ & kRoundDown) {
        high -= range / static_cast<float>(steps);
      } else if (flags_ & kRoundUp) {
        low += range / static_cast<float>(steps);
      }

      bits_ = bit_count;
      low_ = low;
      high_ = high;
      interval_ = (steps > 1) ? (high - low) / static_cast<float>(steps - 1) : 0.0f;
    }
    return;
  }

  // Everything else - integers, enums, handles, tokens, resource ids - is a
  // varint. Signed types are zigzagged.
  if (type == "int8" || type == "int16" || type == "int32") {
    kind_ = kVarInt;
    return;
  }
  kind_ = kVarUInt;
}

float FieldDecoder::DecodeFloat(BitReader* bits) const {
  switch (kind_) {
    case kFloatCoord:
      return ReadCoord(bits);
    case kNormal:
      return ReadNormal(bits);
    case kFloatSimulationTime:
      return static_cast<float>(bits->ReadVarUInt32()) * (1.0f / 64.0f);
    case kFloatNoScale:
      return BitsToFloat(bits->ReadBits(32));
    case kFloatQuantized: {
      // The order of these checks is part of the format.
      if (flags_ & kRoundDown) {
        if (bits->ReadBit()) return low_;
      }
      if (flags_ & kRoundUp) {
        if (bits->ReadBit()) return high_;
      }
      if (flags_ & kEncodeZero) {
        if (bits->ReadBit()) return 0.0f;
      }
      const std::uint32_t raw = bits->ReadBits(bits_);
      return low_ + static_cast<float>(raw) * interval_;
    }
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
    case kVarInt:
      value.kind = FieldValue::kInt;
      value.i = bits->ReadVarInt32();
      break;
    case kFixed64:
      value.kind = FieldValue::kUInt;
      value.u = static_cast<std::uint64_t>(bits->ReadBits(32)) |
                (static_cast<std::uint64_t>(bits->ReadBits(32)) << 32);
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
    case kVarUInt:
      value.kind = FieldValue::kUInt;
      value.u = bits->ReadVarUInt32();
      break;
    default: {
      value.kind = FieldValue::kFloat;
      // Read every component; keep the first.
      const float first = DecodeFloat(bits);
      for (int i = 1; i < components_; ++i) DecodeFloat(bits);
      value.f = first;
      break;
    }
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
    flat.name = prefix.empty() ? info.var_name : prefix + "." + info.var_name;
    flat.decoder = FieldDecoder(info);
    // Arrays come in two spellings: the vector templates, and a plain fixed
    // size suffix such as MedalRank_t[6]. char[128] looks like the latter but
    // is a string, and is decoded as one.
    const bool fixed_array = !info.var_type.empty() &&
                             info.var_type.back() == ']' &&
                             !StartsWith(info.var_type, "char[");
    flat.is_array = fixed_array ||
                    StartsWith(info.var_type, "CNetworkUtlVectorBase") ||
                    StartsWith(info.var_type, "CUtlVector");

    if (info.has_child()) {
      const Serializer* child = set.Find(info.field_serializer_name);
      if (child != nullptr) {
        Flatten(set, *child, flat.name, depth + 1, &flat.children);
      }
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

const EntityDecoder::FlatField* EntityDecoder::Resolve(const FlatClass& flat,
                                                       const FieldPath& path,
                                                       std::string* name) const {
  if (path.path[0] < 0 ||
      static_cast<std::size_t>(path.path[0]) >= flat.fields.size()) {
    return nullptr;
  }
  const FlatField* field = &flat.fields[static_cast<std::size_t>(path.path[0])];
  *name = field->name;
  // An array spends one path level on the element index before any member
  // index. A vector of structs therefore uses two levels: which element, then
  // which member of it.
  bool subscript_pending = field->is_array;

  for (int level = 1; level <= path.last; ++level) {
    const int index = path.path[level];
    if (index < 0) return nullptr;

    if (subscript_pending) {
      *name = field->name + "." + std::to_string(index);
      subscript_pending = false;
      continue;
    }
    if (!field->children.empty()) {
      if (static_cast<std::size_t>(index) >= field->children.size()) return nullptr;
      field = &field->children[static_cast<std::size_t>(index)];
      *name = field->name;
      subscript_pending = field->is_array;
      continue;
    }
    return nullptr;
  }
  return field;
}

// ------------------------------------------------------------ packet entities

bool EntityDecoder::ApplyPacket(const std::string& message, std::string* error) {
  // CSVCMsg_PacketEntities: entity_data = 7 in older builds, serialized
  // entities = 13 in current ones.
  pb::Slice data;
  int updated_entries = 0;
  bool is_delta = false;
  {
    pb::Reader r(message);
    std::uint32_t field = 0;
    while (r.NextField(&field)) {
      // Which field actually carries the payload has changed between builds,
      // so when tracing, show what the message really contains rather than
      // assuming.
      if (trace_ > 0 && r.wire_type() == pb::kLengthDelimited) {
        const pb::Slice s = r.ReadBytes();
        std::printf("    field %-3u len=%zu\n", field, s.size);
        if (field == 7) data = s;
        if (field == 13 && data.data == nullptr) data = s;
        continue;
      }
      switch (field) {
        case 2: updated_entries = r.ReadInt32(); break;
        case 3: is_delta = r.ReadBool(); break;
        // entity_data is the payload in current builds; serialized_entities
        // exists alongside it but is a much smaller, different thing. Taking
        // the wrong one decodes noise, so the preference is explicit.
        case 7: data = r.ReadBytes(); break;
        case 13:
          if (data.data == nullptr) data = r.ReadBytes();
          break;
        default:
          if (trace_ > 0) {
            const std::uint64_t v = r.ReadVarint();
            std::printf("    field %-3u = %llu\n", field,
                        static_cast<unsigned long long>(v));
          }
          break;
      }
    }
  }
  if (data.data == nullptr || updated_entries <= 0) return true;
  (void)is_delta;

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
      return Err(error, "entity index out of range");
    }

    const bool leaving = bits.ReadBit();
    const bool creating = bits.ReadBit();

    if (leaving) {
      // Leave, and possibly delete. The second bit distinguishes them.
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
        std::printf("   +create idx=%-5d class=%-3d %s\n", entity_index, class_id,
                    fresh.class_name.c_str());
      }
      entities_[entity_index] = std::move(fresh);
      entity = &entities_[entity_index];
    } else {
      auto it = entities_.find(entity_index);
      if (it == entities_.end()) {
        ++packets_failed_;
        return Err(error, "delta for an entity that does not exist");
      }
      entity = &it->second;
    }

    auto flat = by_class_id_.find(entity->class_id);
    if (flat == by_class_id_.end()) {
      ++packets_failed_;
      return Err(error, "entity has no flattened class");
    }

    std::vector<FieldPath> paths;
    if (!ReadFieldPaths(&bits, &paths)) {
      ++packets_failed_;
      return Err(error, "field path stream ended badly");
    }
    for (const FieldPath& path : paths) {
      std::string name;
      const FlatField* field = Resolve(flat->second, path, &name);
      if (field == nullptr) {
        ++packets_failed_;
        return Err(error, "field path " + path.ToString() + " does not resolve in " +
                              flat->second.name);
      }
      const FieldValue decoded = field->decoder.Decode(&bits);
      if (trace) {
        std::printf("      %-46s = %s\n", name.c_str(),
                    decoded.kind == FieldValue::kString
                        ? decoded.s.c_str()
                        : std::to_string(decoded.AsFloat()).c_str());
      }
      entity->values[name] = decoded;
      ++updates_applied_;
      if (!bits.ok()) {
        ++packets_failed_;
        return Err(error, "bit stream exhausted mid update");
      }
    }
  }
  return true;
}

}  // namespace cs2mv
