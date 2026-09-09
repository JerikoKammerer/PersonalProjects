#include "cs2mv/json.h"

#include <cmath>
#include <cstdio>

namespace cs2mv {

void JsonWriter::Value(double v) {
  Sep();
  if (std::isnan(v) || std::isinf(v)) {
    out_ += "null";
    return;
  }
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.4g", v);
  out_ += buf;
}

void JsonWriter::AppendQuoted(const std::string& s) {
  out_ += '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"': out_ += "\\\""; break;
      case '\\': out_ += "\\\\"; break;
      case '\n': out_ += "\\n"; break;
      case '\r': out_ += "\\r"; break;
      case '\t': out_ += "\\t"; break;
      case '\b': out_ += "\\b"; break;
      case '\f': out_ += "\\f"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out_ += buf;
        } else {
          // Player names arrive as UTF-8 and are emitted unchanged; JSON
          // permits raw UTF-8 in strings.
          out_ += static_cast<char>(c);
        }
        break;
    }
  }
  out_ += '"';
}

}  // namespace cs2mv
