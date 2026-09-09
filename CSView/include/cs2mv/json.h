// A very small JSON writer: enough to serialise the match model for the web UI.
#ifndef CS2MV_JSON_H_
#define CS2MV_JSON_H_

#include <cstdint>
#include <string>

namespace cs2mv {

class JsonWriter {
 public:
  void BeginObject() { Sep(); out_ += '{'; first_ = true; }
  void EndObject() { out_ += '}'; first_ = false; }
  void BeginArray() { Sep(); out_ += '['; first_ = true; }
  void EndArray() { out_ += ']'; first_ = false; }

  void Key(const std::string& k) {
    Sep();
    AppendQuoted(k);
    out_ += ':';
    first_ = true;  // the value that follows must not be preceded by a comma
  }

  void Value(const std::string& v) { Sep(); AppendQuoted(v); }
  void Value(const char* v) { Value(std::string(v)); }
  void Value(bool v) { Sep(); out_ += v ? "true" : "false"; }
  void Value(double v);

  // Integer overloads are spelled with fundamental types rather than the
  // <cstdint> aliases so that the set stays distinct on both LP64 and LLP64.
  void Value(int v) { Number(static_cast<long long>(v)); }
  void Value(long v) { Number(static_cast<long long>(v)); }
  void Value(long long v) { Number(v); }
  void Value(unsigned int v) { Number(static_cast<unsigned long long>(v)); }
  void Value(unsigned long v) { Number(static_cast<unsigned long long>(v)); }
  void Value(unsigned long long v) { Number(v); }

  void Null() { Sep(); out_ += "null"; }

  // 64 bit ids do not survive a round trip through a JavaScript number, so
  // they go over the wire as decimal strings.
  void IdValue(std::uint64_t v) { Value(std::to_string(v)); }

  // Convenience: key plus value in one call.
  template <typename T>
  void Field(const std::string& k, const T& v) {
    Key(k);
    Value(v);
  }
  void FieldId(const std::string& k, std::uint64_t v) {
    Key(k);
    IdValue(v);
  }

  const std::string& str() const { return out_; }

 private:
  void Sep() {
    if (!first_) out_ += ',';
    first_ = false;
  }
  void Number(long long v) { Sep(); out_ += std::to_string(v); }
  void Number(unsigned long long v) { Sep(); out_ += std::to_string(v); }
  void AppendQuoted(const std::string& s);

  std::string out_;
  bool first_ = true;
};

}  // namespace cs2mv

#endif  // CS2MV_JSON_H_
