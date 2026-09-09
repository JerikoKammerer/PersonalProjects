// Entity decoding: the part of a Source 2 demo that actually knows where
// everyone is standing.
//
// svc_PacketEntities carries a dense bit stream of entity updates. There are no
// length prefixes and no field names - each update is a list of field paths
// (fieldpath.h) with a value after each, and the encoding of a value depends
// entirely on what the schema (serializers.h) says about that field.
//
// The consequence worth understanding before touching this: there is no
// partial decoding. A field cannot be skipped because nothing says how long it
// is, so reading m_cellX correctly requires decoding every field before it
// correctly. One wrong decoder desynchronises the stream and everything after
// it is noise.
#ifndef CS2MV_ENTITIES_H_
#define CS2MV_ENTITIES_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "cs2mv/bits.h"
#include "cs2mv/fieldpath.h"
#include "cs2mv/serializers.h"

namespace cs2mv {

struct FieldValue {
  enum Kind { kNone, kInt, kUInt, kFloat, kString };
  Kind kind = kNone;
  long long i = 0;
  unsigned long long u = 0;
  float f = 0.0f;
  std::string s;

  float AsFloat() const {
    switch (kind) {
      case kFloat: return f;
      case kInt: return static_cast<float>(i);
      case kUInt: return static_cast<float>(u);
      default: return 0.0f;
    }
  }
  long long AsInt() const {
    switch (kind) {
      case kInt: return i;
      case kUInt: return static_cast<long long>(u);
      case kFloat: return static_cast<long long>(f);
      default: return 0;
    }
  }
};

// How one field is read off the wire. Chosen once per schema field, from its
// type, encoder and quantisation parameters, then reused for every update.
class FieldDecoder {
 public:
  FieldDecoder() = default;
  explicit FieldDecoder(const FieldInfo& info);

  FieldValue Decode(BitReader* bits) const;

 private:
  enum Kind {
    kBool,
    kVarUInt,
    kVarInt,      // zigzag
    kFixed64,
    kFloatNoScale,
    kFloatCoord,
    kFloatSimulationTime,
    kFloatQuantized,
    kNormal,
    kString,
    kSkipUnknown,
  };

  float DecodeFloat(BitReader* bits) const;

  Kind kind_ = kVarUInt;
  // Quantisation, precomputed. A bit count of 0 or 32 means no scaling at all.
  int bits_ = 0;
  int flags_ = 0;
  float low_ = 0.0f;
  float high_ = 0.0f;
  float interval_ = 0.0f;
  bool no_scale_ = true;
};

// One entity's current field values, keyed by the dotted path the schema gives
// them ("CBodyComponent.m_cellX").
struct Entity {
  int index = -1;
  int serial = 0;
  int class_id = -1;
  std::string class_name;
  std::map<std::string, FieldValue> values;

  const FieldValue* Get(const std::string& name) const {
    auto it = values.find(name);
    return it == values.end() ? nullptr : &it->second;
  }
};

// Resolves field paths against the schema and applies updates to entities.
class EntityDecoder {
 public:
  bool Init(const SerializerSet& serializers, const ClassTable& classes,
            std::string* error);

  // Applies one svc_PacketEntities message. Returns false when the bit stream
  // desynchronises, which is fatal for the rest of that packet.
  bool ApplyPacket(const std::string& message, std::string* error);

  const std::map<int, Entity>& entities() const { return entities_; }

  // Whether an entity creation carries a spawn group handle after its serial
  // number. Builds differ, and guessing wrong desynchronises immediately, so
  // it is switchable while the right answer is established.
  void set_read_spawn_group(bool on) { read_spawn_group_ = on; }
  // Prints each step of the first packet, which is the only practical way to
  // see where the bit stream starts drifting.
  void set_trace(int packets) { trace_ = packets; }

  // Diagnostics, because "it decoded" and "it decoded correctly" are different
  // claims and only the counts can tell them apart.
  long long updates_applied() const { return updates_applied_; }
  long long packets_failed() const { return packets_failed_; }

 private:
  // A class's field tree, shaped the way field paths address it: path[0]
  // selects a top level field, path[1] a member of that field's struct, and so
  // on. Arrays reuse the element decoder at every index.
  struct FlatField {
    std::string name;
    FieldDecoder decoder;
    bool is_array = false;
    std::vector<FlatField> children;
  };
  struct FlatClass {
    std::string name;
    std::vector<FlatField> fields;
  };

  void Flatten(const SerializerSet& set, const Serializer& serializer,
               const std::string& prefix, int depth,
               std::vector<FlatField>* out);
  const FlatField* Resolve(const FlatClass& flat, const FieldPath& path,
                           std::string* name) const;

  std::map<int, FlatClass> by_class_id_;
  // Entity packets spend exactly enough bits for the largest class id.
  int class_id_bits_ = 8;
  std::map<int, Entity> entities_;
  bool read_spawn_group_ = true;
  int trace_ = 0;
  long long updates_applied_ = 0;
  long long packets_failed_ = 0;
};

}  // namespace cs2mv

#endif  // CS2MV_ENTITIES_H_
