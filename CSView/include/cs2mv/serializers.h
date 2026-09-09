// The schema half of a Source 2 demo's entity system.
//
// A demo does not describe entities with fixed offsets; it ships its own
// schema. DEM_SendTables carries a CSVCMsg_FlattenedSerializer listing every
// networked class, its fields, and how each field is encoded on the wire.
// DEM_ClassInfo then maps the class ids that appear in packets onto those
// serializers.
//
// Nothing here decodes entity data - it only builds the tables that
// entities.h needs in order to.
#ifndef CS2MV_SERIALIZERS_H_
#define CS2MV_SERIALIZERS_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace cs2mv {

// One networked field, as the demo describes it.
struct FieldInfo {
  std::string var_type;    // "float32", "Vector", "CHandle< CBaseEntity >", ...
  std::string var_name;    // "m_vecOrigin", "m_iHealth", ...
  std::string send_node;   // grouping only; not needed to decode
  std::string encoder;     // "coord", "normal", "simtime", "qangle_pitch_yaw"
  int bit_count = 0;
  float low = 0.0f;
  float high = 0.0f;
  int encode_flags = 0;

  // Set when the field is itself a struct or an array of them.
  std::string field_serializer_name;
  int field_serializer_version = -1;

  bool has_child() const { return !field_serializer_name.empty(); }
};

struct Serializer {
  std::string name;   // "CCSPlayerPawn", "CBodyComponentBaseAnimGraph", ...
  int version = 0;
  std::vector<int> fields;  // indices into SerializerSet::fields
};

struct SerializerSet {
  std::vector<std::string> symbols;
  std::vector<FieldInfo> fields;
  std::vector<Serializer> serializers;

  // Serializer name to index. A demo can carry several versions of a name; the
  // last one wins, which is the one the packets use.
  std::map<std::string, int> by_name;

  const Serializer* Find(const std::string& name) const {
    auto it = by_name.find(name);
    return it == by_name.end() ? nullptr : &serializers[static_cast<std::size_t>(it->second)];
  }
};

// Parses a DEM_SendTables body. The CDemoSendTables `data` field holds a
// varint length followed by the CSVCMsg_FlattenedSerializer itself.
bool ParseSendTables(const std::string& body, SerializerSet* out,
                     std::string* error);

// class id -> serializer name, from a DEM_ClassInfo body.
struct ClassTable {
  std::map<int, std::string> names;
  int max_class_id = 0;

  // Entity packets spend exactly enough bits to hold the largest class id.
  int class_id_bits() const {
    int bits = 1;
    while ((1 << bits) <= max_class_id) ++bits;
    return bits;
  }
};

bool ParseClassInfo(const std::string& body, ClassTable* out, std::string* error);

}  // namespace cs2mv

#endif  // CS2MV_SERIALIZERS_H_
