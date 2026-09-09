// Field path decoding, the addressing scheme inside a Source 2 entity update.
//
// An entity update does not name the fields it carries. It walks a path
// through the class's field tree, encoded as a sequence of operations - "move
// one along", "descend and start at zero", "pop back to the top and move on" -
// and each operation is itself Huffman coded against a fixed weight table
// baked into the game.
//
// The table below is that wire format. It is not tunable: the weights decide
// the code lengths, so a single wrong weight builds a different tree and every
// subsequent bit is misread. That is also why this is worth testing on its own
// rather than only through a decoded demo.
#ifndef CS2MV_FIELDPATH_H_
#define CS2MV_FIELDPATH_H_

#include <string>
#include <vector>

#include "cs2mv/bits.h"

namespace cs2mv {

// A path into a class's field tree. Seven levels is what the format allows.
struct FieldPath {
  int path[7] = {-1, 0, 0, 0, 0, 0, 0};
  int last = 0;      // index of the deepest valid element
  bool done = false;

  int depth() const { return last + 1; }
  std::string ToString() const;
};

// The operations, in the order their weights are declared. The order is part
// of the format: it decides tie-breaking when the Huffman tree is built.
enum FieldPathOp {
  kPlusOne = 0,
  kPlusTwo,
  kPlusThree,
  kPlusFour,
  kPlusN,
  kPushOneLeftDeltaZeroRightZero,
  kPushOneLeftDeltaZeroRightNonZero,
  kPushOneLeftDeltaOneRightZero,
  kPushOneLeftDeltaOneRightNonZero,
  kPushOneLeftDeltaNRightZero,
  kPushOneLeftDeltaNRightNonZero,
  kPushOneLeftDeltaNRightNonZeroPack6Bits,
  kPushOneLeftDeltaNRightNonZeroPack8Bits,
  kPushTwoLeftDeltaZero,
  kPushTwoPack5LeftDeltaZero,
  kPushThreeLeftDeltaZero,
  kPushThreePack5LeftDeltaZero,
  kPushTwoLeftDeltaOne,
  kPushTwoPack5LeftDeltaOne,
  kPushThreeLeftDeltaOne,
  kPushThreePack5LeftDeltaOne,
  kPushTwoLeftDeltaN,
  kPushTwoPack5LeftDeltaN,
  kPushThreeLeftDeltaN,
  kPushThreePack5LeftDeltaN,
  kPushN,
  kPushNAndNonTopological,
  kPopOnePlusOne,
  kPopOnePlusN,
  kPopAllButOnePlusOne,
  kPopAllButOnePlusN,
  kPopAllButOnePlusNPack3Bits,
  kPopAllButOnePlusNPack6Bits,
  kPopNPlusOne,
  kPopNPlusN,
  kPopNAndNonTopographical,
  kNonTopoComplex,
  kNonTopoPenultimatePlusOne,
  kNonTopoComplexPack4Bits,
  kFieldPathEncodeFinish,
  kFieldPathOpCount,
};

// The fixed weight of each operation, indexed by FieldPathOp.
extern const int kFieldPathWeights[kFieldPathOpCount];
const char* FieldPathOpName(int op);

// The Huffman tree over those weights. Built once and shared: it never varies
// between demos.
class FieldPathHuffman {
 public:
  static const FieldPathHuffman& Instance();

  // Reads one operation. Returns -1 if the bit stream ran out.
  int ReadOp(BitReader* bits) const;

  // Test hooks: the code assigned to an operation, as a string of '0'/'1'.
  std::string CodeFor(int op) const;
  int LeafCount() const;

 private:
  FieldPathHuffman();

  struct Node {
    int left = -1;
    int right = -1;
    int op = -1;  // >= 0 for a leaf
  };
  std::vector<Node> nodes_;
  int root_ = -1;
};

// Reads the list of field paths an entity update touches, in order.
bool ReadFieldPaths(BitReader* bits, std::vector<FieldPath>* out);

}  // namespace cs2mv

#endif  // CS2MV_FIELDPATH_H_
