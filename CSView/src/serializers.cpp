#include "cs2mv/serializers.h"

#include "cs2mv/protobuf.h"

namespace cs2mv {
namespace {

bool Err(std::string* error, const std::string& msg) {
  if (error != nullptr) *error = msg;
  return false;
}

// Symbols are referenced by index throughout the schema; out-of-range indices
// mean a malformed table rather than something to guess at.
const std::string& Symbol(const SerializerSet& set, int index) {
  static const std::string kEmpty;
  if (index < 0 || static_cast<std::size_t>(index) >= set.symbols.size()) {
    return kEmpty;
  }
  return set.symbols[static_cast<std::size_t>(index)];
}

// ProtoFlattenedSerializerField_t
void ParseField(const pb::Slice& body, FieldInfo* out, int* type_sym,
                int* name_sym, int* node_sym, int* encoder_sym,
                int* serializer_sym) {
  pb::Reader r(body);
  std::uint32_t field = 0;
  while (r.NextField(&field)) {
    switch (field) {
      case 1: *type_sym = r.ReadInt32(); break;
      case 2: *name_sym = r.ReadInt32(); break;
      case 3: out->bit_count = r.ReadInt32(); break;
      case 4: out->low = r.ReadFloat(); break;
      case 5: out->high = r.ReadFloat(); break;
      case 6: out->encode_flags = r.ReadInt32(); break;
      case 7: *serializer_sym = r.ReadInt32(); break;
      case 8: out->field_serializer_version = r.ReadInt32(); break;
      case 9: *node_sym = r.ReadInt32(); break;
      case 10: *encoder_sym = r.ReadInt32(); break;
      default: break;
    }
  }
}

}  // namespace

bool ParseSendTables(const std::string& body, SerializerSet* out,
                     std::string* error) {
  // CDemoSendTables { data = 1 }
  pb::Slice data;
  {
    pb::Reader r(body);
    std::uint32_t field = 0;
    while (r.NextField(&field)) {
      if (field == 1 && r.wire_type() == pb::kLengthDelimited) data = r.ReadBytes();
    }
  }
  if (data.data == nullptr) return Err(error, "DEM_SendTables carried no data");

  // The payload is a varint length followed by the message itself.
  pb::Reader outer(data);
  const std::uint64_t length = outer.ReadVarint();
  if (!outer.ok() || length > outer.remaining()) {
    return Err(error, "send tables: bad length prefix");
  }
  pb::Reader r(data.data + (data.size - outer.remaining()),
               static_cast<std::size_t>(length));

  // CSVCMsg_FlattenedSerializer { serializers = 1, symbols = 2, fields = 3 }
  std::vector<pb::Slice> serializer_bodies;
  std::vector<pb::Slice> field_bodies;
  std::uint32_t field = 0;
  while (r.NextField(&field)) {
    if (r.wire_type() != pb::kLengthDelimited) continue;
    switch (field) {
      case 1: serializer_bodies.push_back(r.ReadBytes()); break;
      case 2: out->symbols.push_back(r.ReadString()); break;
      case 3: field_bodies.push_back(r.ReadBytes()); break;
      default: break;
    }
  }
  if (!r.ok()) return Err(error, "send tables: malformed FlattenedSerializer");
  if (out->symbols.empty()) return Err(error, "send tables: no symbol table");

  out->fields.reserve(field_bodies.size());
  for (const pb::Slice& f : field_bodies) {
    FieldInfo info;
    int type_sym = -1, name_sym = -1, node_sym = -1, encoder_sym = -1;
    int serializer_sym = -1;
    ParseField(f, &info, &type_sym, &name_sym, &node_sym, &encoder_sym,
               &serializer_sym);
    info.var_type = Symbol(*out, type_sym);
    info.var_name = Symbol(*out, name_sym);
    info.send_node = Symbol(*out, node_sym);
    info.encoder = Symbol(*out, encoder_sym);
    info.field_serializer_name = Symbol(*out, serializer_sym);
    out->fields.push_back(std::move(info));
  }

  out->serializers.reserve(serializer_bodies.size());
  for (const pb::Slice& s : serializer_bodies) {
    // ProtoFlattenedSerializer_t { name_sym = 1, version = 2, fields_index = 3 }
    Serializer serializer;
    int name_sym = -1;
    pb::Reader sr(s);
    std::uint32_t sf = 0;
    while (sr.NextField(&sf)) {
      switch (sf) {
        case 1: name_sym = sr.ReadInt32(); break;
        case 2: serializer.version = sr.ReadInt32(); break;
        case 3:
          // repeated int32, packed or not
          if (sr.wire_type() == pb::kLengthDelimited) {
            pb::Reader packed(sr.ReadBytes());
            while (packed.remaining() > 0) {
              const int index = static_cast<int>(packed.ReadVarint());
              if (!packed.ok()) break;
              serializer.fields.push_back(index);
            }
          } else {
            serializer.fields.push_back(sr.ReadInt32());
          }
          break;
        default:
          break;
      }
    }
    serializer.name = Symbol(*out, name_sym);
    if (serializer.name.empty()) continue;
    out->by_name[serializer.name] = static_cast<int>(out->serializers.size());
    out->serializers.push_back(std::move(serializer));
  }

  if (out->serializers.empty()) {
    return Err(error, "send tables: no serializers");
  }
  return true;
}

bool ParseClassInfo(const std::string& body, ClassTable* out,
                    std::string* error) {
  // CDemoClassInfo { classes = 1 repeated class_t }
  pb::Reader r(body);
  std::uint32_t field = 0;
  while (r.NextField(&field)) {
    if (field != 1 || r.wire_type() != pb::kLengthDelimited) continue;
    // class_t { class_id = 1, network_name = 2, table_name = 3 }
    pb::Reader c(r.ReadBytes());
    int class_id = -1;
    std::string network_name;
    std::uint32_t cf = 0;
    while (c.NextField(&cf)) {
      if (cf == 1) class_id = c.ReadInt32();
      else if (cf == 2) network_name = c.ReadString();
    }
    if (class_id < 0 || network_name.empty()) continue;
    out->names[class_id] = network_name;
    if (class_id > out->max_class_id) out->max_class_id = class_id;
  }
  if (out->names.empty()) return Err(error, "DEM_ClassInfo listed no classes");
  return true;
}

}  // namespace cs2mv
