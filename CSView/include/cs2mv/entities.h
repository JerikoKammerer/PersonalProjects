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
  // Up to four components; scalars use v[0]. Vectors keep all of theirs, since
  // a position with only its x is not much of a position.
  float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  std::string s;

  float AsFloat() const {
    switch (kind) {
      case kFloat: return v[0];
      case kInt: return static_cast<float>(i);
      case kUInt: return static_cast<float>(u);
      default: return 0.0f;
    }
  }
  long long AsInt() const {
    switch (kind) {
      case kInt: return i;
      case kUInt: return static_cast<long long>(u);
      case kFloat: return static_cast<long long>(v[0]);
      default: return 0;
    }
  }
};

// The pieces of a schema type string that decide how it is decoded:
// "CNetworkUtlVectorBase< CHandle< CBaseEntity > >" is a vector whose
// elements are handles; "uint32[4]" a fixed array; "CEntityIdentity*" a
// pointer to a struct.
struct TypeSpec {
  std::string base;
  std::string generic;  // inside the outermost < >, if any
  bool pointer = false;
  int count = 0;        // trailing [N], if any
};
TypeSpec ParseTypeSpec(const std::string& type);

// How one value is read off the wire. Chosen once per schema field, from its
// type, encoder and quantisation parameters, then reused for every update.
class FieldDecoder {
 public:
  FieldDecoder() = default;
  // `type` is the type being decoded - the field's own type for a scalar, the
  // element type for an array - while the encoding parameters always come
  // from the field.
  FieldDecoder(const std::string& type, const FieldInfo& info);

  // Fixed-purpose decoders for the structural values an update can carry.
  static FieldDecoder Bool();
  static FieldDecoder VarUInt();
  // A pointer that may point at one of several types: a presence bit, then
  // which type. Decodes to -1 for absent, else the type index.
  static FieldDecoder Polymorphic();

  FieldValue Decode(BitReader* bits) const;
  bool polymorphic() const { return kind_ == kPolymorphic; }
  // Set when the schema declares an encoding nothing here understands. Such a
  // field cannot be read, and the packet is lost from that point.
  bool unsupported() const { return kind_ == kUnsupported; }
  const std::string& description() const { return description_; }

 private:
  enum Kind {
    kBool,
    kVarUInt32,
    kVarInt32,     // zigzag
    kVarUInt64,
    kVarInt64,
    kFixed64,
    kSequence,     // HSequence: varint minus one
    kString,
    kBlob,         // CUtlBinaryBlock: a byte count, then the bytes
    kFloats,       // 1..4 floats, each via float_
    kVectorNormal, // a unit vector: two components and a sign for the third
    kQAngle,
    kPolymorphic,
    kUnsupported,
  };
  enum FloatKind {
    kNoScale,
    kCoord,
    kSimulationTime,
    kNormal,
    kQuantized,
  };
  enum QAngleKind {
    kQAnglePitchYaw,   // two angles of bits_ each
    kQAnglePrecise,    // three presence bits, then 20 bit angles
    kQAngleFixedBits,  // three angles of bits_ each
    kQAngleCoord,      // three presence bits, then coords
  };

  void InitFloat(const FieldInfo& info);
  float DecodeFloat(BitReader* bits) const;

  Kind kind_ = kVarUInt32;
  FloatKind float_ = kNoScale;
  QAngleKind qangle_ = kQAngleCoord;
  int components_ = 1;
  // Quantisation, precomputed. A bit count of 0 or 32 means no scaling at all.
  int bits_ = 0;
  int flags_ = 0;
  float low_ = 0.0f;
  float high_ = 0.0f;
  float interval_ = 0.0f;
  std::string description_;
};

// One entity's current field values, keyed by the dotted path the schema gives
// them ("CBodyComponent.m_cellX"; array elements as "m_iAmmo.3").
struct Entity {
  int index = -1;
  int serial = 0;
  int class_id = -1;
  std::string class_name;
  std::map<std::string, FieldValue> values;
  // Which type each polymorphic pointer currently points at, by field name.
  // 0 is the declared type, k the k-th alternative.
  std::map<std::string, int> variants;

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

  // The instancebaseline string table: the default state of each class, in
  // the same encoding as an update. A created entity starts from it.
  void SetBaseline(int class_id, const std::string& data) {
    baselines_[class_id] = data;
  }
  std::size_t baseline_count() const { return baselines_.size(); }

  // Applies one svc_PacketEntities message. Returns false when the bit stream
  // desynchronises, which is fatal for the rest of that packet.
  bool ApplyPacket(const std::string& message, std::string* error);

  // Decodes one class's baseline into a scratch entity, as a check that the
  // decoders for that class are right: a baseline that decodes cleanly, with
  // only padding left over, is strong evidence. Returns false with a reason
  // otherwise. `bits_left` reports the padding either way.
  bool CheckBaseline(int class_id, Entity* scratch, int* bits_left,
                     std::string* error);

  const std::map<int, Entity>& entities() const { return entities_; }
  const std::map<int, std::string>& baselines() const { return baselines_; }
  const std::string* class_name(int class_id) const {
    auto it = by_class_id_.find(class_id);
    return it == by_class_id_.end() ? nullptr : &it->second.name;
  }

  // Whether an entity creation carries a spawn group handle after its serial
  // number. Builds differ, and guessing wrong desynchronises immediately, so
  // it is switchable while the right answer is established.
  void set_read_spawn_group(bool on) { read_spawn_group_ = on; }
  // Prints each step of the first packets, which is the only practical way to
  // see where the bit stream starts drifting.
  void set_trace(int packets) { trace_ = packets; }

  // Diagnostics, because "it decoded" and "it decoded correctly" are different
  // claims and only the counts can tell them apart.
  long long updates_applied() const { return updates_applied_; }
  long long packets_failed() const { return packets_failed_; }

 private:
  // A class's field tree, shaped the way field paths address it: path[0]
  // selects a top level field, path[1] a member of that field's struct, and so
  // on. What a path level means depends on the field's model.
  struct FlatField {
    enum Model {
      kSimple,         // a value
      kFixedArray,     // T[N]: next level is the element index
      kVariableArray,  // vector of values: next level is the element index,
                       // a path ending here carries the new size
      kFixedTable,     // a struct: next level picks a member, a path ending
                       // here carries one bit
      kVariableTable,  // vector of structs: element index, then member
    };
    std::string name;
    Model model = kSimple;
    FieldDecoder decoder;  // the value, or an element of the array
    FieldDecoder self;     // what a path ending at the field itself carries
    std::vector<FlatField> children;
    // For a polymorphic pointer, the members of each alternative type, in the
    // schema's order. `children` stays the declared type's.
    std::vector<std::vector<FlatField>> alternatives;
  };
  struct FlatClass {
    std::string name;
    std::vector<FlatField> fields;
  };

  void Flatten(const SerializerSet& set, const Serializer& serializer,
               const std::string& prefix, int depth,
               std::vector<FlatField>* out);
  const FieldDecoder* Resolve(const FlatClass& flat, const Entity& entity,
                              const FieldPath& path, std::string* name) const;
  bool ApplyUpdate(Entity* entity, BitReader* bits, bool trace,
                   std::string* error);

  std::map<int, FlatClass> by_class_id_;
  std::map<int, std::string> baselines_;
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
