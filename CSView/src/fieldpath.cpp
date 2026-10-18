#include "cs2mv/fieldpath.h"

#include <algorithm>
#include <queue>
#include <vector>

namespace cs2mv {

// Weights baked into the game's encoder. Their only job is to decide code
// lengths, so they are copied exactly and never adjusted.
const int kFieldPathWeights[kFieldPathOpCount] = {
    36271,  // PlusOne
    10334,  // PlusTwo
    1375,   // PlusThree
    646,    // PlusFour
    4128,   // PlusN
    35,     // PushOneLeftDeltaZeroRightZero
    3,      // PushOneLeftDeltaZeroRightNonZero
    521,    // PushOneLeftDeltaOneRightZero
    2942,   // PushOneLeftDeltaOneRightNonZero
    560,    // PushOneLeftDeltaNRightZero
    471,    // PushOneLeftDeltaNRightNonZero
    10530,  // PushOneLeftDeltaNRightNonZeroPack6Bits
    251,    // PushOneLeftDeltaNRightNonZeroPack8Bits
    0,      // PushTwoLeftDeltaZero
    0,      // PushTwoPack5LeftDeltaZero
    0,      // PushThreeLeftDeltaZero
    0,      // PushThreePack5LeftDeltaZero
    0,      // PushTwoLeftDeltaOne
    0,      // PushTwoPack5LeftDeltaOne
    0,      // PushThreeLeftDeltaOne
    0,      // PushThreePack5LeftDeltaOne
    0,      // PushTwoLeftDeltaN
    0,      // PushTwoPack5LeftDeltaN
    0,      // PushThreeLeftDeltaN
    0,      // PushThreePack5LeftDeltaN
    0,      // PushN
    310,    // PushNAndNonTopological
    2,      // PopOnePlusOne
    0,      // PopOnePlusN
    1837,   // PopAllButOnePlusOne
    149,    // PopAllButOnePlusN
    300,    // PopAllButOnePlusNPack3Bits
    634,    // PopAllButOnePlusNPack6Bits
    0,      // PopNPlusOne
    0,      // PopNPlusN
    1,      // PopNAndNonTopographical
    76,     // NonTopoComplex
    271,    // NonTopoPenultimatePlusOne
    99,     // NonTopoComplexPack4Bits
    25474,  // FieldPathEncodeFinish
};

namespace {

const char* const kOpNames[kFieldPathOpCount] = {
    "PlusOne", "PlusTwo", "PlusThree", "PlusFour", "PlusN",
    "PushOneLeftDeltaZeroRightZero", "PushOneLeftDeltaZeroRightNonZero",
    "PushOneLeftDeltaOneRightZero", "PushOneLeftDeltaOneRightNonZero",
    "PushOneLeftDeltaNRightZero", "PushOneLeftDeltaNRightNonZero",
    "PushOneLeftDeltaNRightNonZeroPack6Bits",
    "PushOneLeftDeltaNRightNonZeroPack8Bits",
    "PushTwoLeftDeltaZero", "PushTwoPack5LeftDeltaZero",
    "PushThreeLeftDeltaZero", "PushThreePack5LeftDeltaZero",
    "PushTwoLeftDeltaOne", "PushTwoPack5LeftDeltaOne",
    "PushThreeLeftDeltaOne", "PushThreePack5LeftDeltaOne",
    "PushTwoLeftDeltaN", "PushTwoPack5LeftDeltaN",
    "PushThreeLeftDeltaN", "PushThreePack5LeftDeltaN",
    "PushN", "PushNAndNonTopological",
    "PopOnePlusOne", "PopOnePlusN",
    "PopAllButOnePlusOne", "PopAllButOnePlusN",
    "PopAllButOnePlusNPack3Bits", "PopAllButOnePlusNPack6Bits",
    "PopNPlusOne", "PopNPlusN", "PopNAndNonTopographical",
    "NonTopoComplex", "NonTopoPenultimatePlusOne", "NonTopoComplexPack4Bits",
    "FieldPathEncodeFinish",
};

void Pop(FieldPath* path, int count) {
  for (int i = 0; i < count && path->last > 0; ++i) {
    path->path[path->last] = 0;
    --path->last;
  }
}

// Applies one operation. The bit reads are part of the operation, so this is
// where the format's bit budget is actually spent.
void Apply(int op, BitReader* bits, FieldPath* p) {
  switch (op) {
    case kPlusOne: p->path[p->last] += 1; break;
    case kPlusTwo: p->path[p->last] += 2; break;
    case kPlusThree: p->path[p->last] += 3; break;
    case kPlusFour: p->path[p->last] += 4; break;
    case kPlusN:
      p->path[p->last] += static_cast<int>(bits->ReadUBitVarFieldPath()) + 5;
      break;

    case kPushOneLeftDeltaZeroRightZero:
      p->path[++p->last] = 0;
      break;
    case kPushOneLeftDeltaZeroRightNonZero:
      p->path[++p->last] = static_cast<int>(bits->ReadUBitVarFieldPath());
      break;
    case kPushOneLeftDeltaOneRightZero:
      p->path[p->last] += 1;
      p->path[++p->last] = 0;
      break;
    case kPushOneLeftDeltaOneRightNonZero:
      p->path[p->last] += 1;
      p->path[++p->last] = static_cast<int>(bits->ReadUBitVarFieldPath());
      break;
    case kPushOneLeftDeltaNRightZero:
      p->path[p->last] += static_cast<int>(bits->ReadUBitVarFieldPath());
      p->path[++p->last] = 0;
      break;
    case kPushOneLeftDeltaNRightNonZero:
      p->path[p->last] += static_cast<int>(bits->ReadUBitVarFieldPath()) + 2;
      p->path[++p->last] = static_cast<int>(bits->ReadUBitVarFieldPath()) + 1;
      break;
    case kPushOneLeftDeltaNRightNonZeroPack6Bits:
      p->path[p->last] += static_cast<int>(bits->ReadBits(3)) + 2;
      p->path[++p->last] = static_cast<int>(bits->ReadBits(3)) + 1;
      break;
    case kPushOneLeftDeltaNRightNonZeroPack8Bits:
      p->path[p->last] += static_cast<int>(bits->ReadBits(4)) + 2;
      p->path[++p->last] = static_cast<int>(bits->ReadBits(4)) + 1;
      break;

    case kPushTwoLeftDeltaZero:
      for (int i = 0; i < 2; ++i) {
        p->path[++p->last] += static_cast<int>(bits->ReadUBitVarFieldPath());
      }
      break;
    case kPushTwoPack5LeftDeltaZero:
      for (int i = 0; i < 2; ++i) {
        p->path[++p->last] = static_cast<int>(bits->ReadBits(5));
      }
      break;
    case kPushThreeLeftDeltaZero:
      for (int i = 0; i < 3; ++i) {
        p->path[++p->last] += static_cast<int>(bits->ReadUBitVarFieldPath());
      }
      break;
    case kPushThreePack5LeftDeltaZero:
      for (int i = 0; i < 3; ++i) {
        p->path[++p->last] = static_cast<int>(bits->ReadBits(5));
      }
      break;
    case kPushTwoLeftDeltaOne:
      p->path[p->last] += 1;
      for (int i = 0; i < 2; ++i) {
        p->path[++p->last] += static_cast<int>(bits->ReadUBitVarFieldPath());
      }
      break;
    case kPushTwoPack5LeftDeltaOne:
      p->path[p->last] += 1;
      for (int i = 0; i < 2; ++i) {
        p->path[++p->last] += static_cast<int>(bits->ReadBits(5));
      }
      break;
    case kPushThreeLeftDeltaOne:
      p->path[p->last] += 1;
      for (int i = 0; i < 3; ++i) {
        p->path[++p->last] += static_cast<int>(bits->ReadUBitVarFieldPath());
      }
      break;
    case kPushThreePack5LeftDeltaOne:
      p->path[p->last] += 1;
      for (int i = 0; i < 3; ++i) {
        p->path[++p->last] += static_cast<int>(bits->ReadBits(5));
      }
      break;
    case kPushTwoLeftDeltaN:
      p->path[p->last] += static_cast<int>(bits->ReadUBitVar()) + 2;
      for (int i = 0; i < 2; ++i) {
        p->path[++p->last] += static_cast<int>(bits->ReadUBitVarFieldPath());
      }
      break;
    case kPushTwoPack5LeftDeltaN:
      p->path[p->last] += static_cast<int>(bits->ReadUBitVar()) + 2;
      for (int i = 0; i < 2; ++i) {
        p->path[++p->last] += static_cast<int>(bits->ReadBits(5));
      }
      break;
    case kPushThreeLeftDeltaN:
      p->path[p->last] += static_cast<int>(bits->ReadUBitVar()) + 2;
      for (int i = 0; i < 3; ++i) {
        p->path[++p->last] += static_cast<int>(bits->ReadUBitVarFieldPath());
      }
      break;
    case kPushThreePack5LeftDeltaN:
      p->path[p->last] += static_cast<int>(bits->ReadUBitVar()) + 2;
      for (int i = 0; i < 3; ++i) {
        p->path[++p->last] += static_cast<int>(bits->ReadBits(5));
      }
      break;
    case kPushN: {
      const int count = static_cast<int>(bits->ReadUBitVar());
      p->path[p->last] += static_cast<int>(bits->ReadUBitVar());
      for (int i = 0; i < count; ++i) {
        p->path[++p->last] += static_cast<int>(bits->ReadUBitVarFieldPath());
      }
      break;
    }
    case kPushNAndNonTopological: {
      for (int i = 0; i <= p->last; ++i) {
        if (bits->ReadBit()) p->path[i] += bits->ReadVarInt32() + 1;
      }
      const int count = static_cast<int>(bits->ReadUBitVar());
      for (int i = 0; i < count; ++i) {
        p->path[++p->last] = static_cast<int>(bits->ReadUBitVarFieldPath());
      }
      break;
    }

    case kPopOnePlusOne:
      Pop(p, 1);
      p->path[p->last] += 1;
      break;
    case kPopOnePlusN:
      Pop(p, 1);
      p->path[p->last] += static_cast<int>(bits->ReadUBitVarFieldPath()) + 1;
      break;
    case kPopAllButOnePlusOne:
      Pop(p, p->last);
      p->path[0] += 1;
      break;
    case kPopAllButOnePlusN:
      Pop(p, p->last);
      p->path[0] += static_cast<int>(bits->ReadUBitVarFieldPath()) + 1;
      break;
    case kPopAllButOnePlusNPack3Bits:
      Pop(p, p->last);
      p->path[0] += static_cast<int>(bits->ReadBits(3)) + 1;
      break;
    case kPopAllButOnePlusNPack6Bits:
      Pop(p, p->last);
      p->path[0] += static_cast<int>(bits->ReadBits(6)) + 1;
      break;
    case kPopNPlusOne:
      Pop(p, static_cast<int>(bits->ReadUBitVarFieldPath()));
      p->path[p->last] += 1;
      break;
    case kPopNPlusN:
      Pop(p, static_cast<int>(bits->ReadUBitVarFieldPath()));
      p->path[p->last] += bits->ReadVarInt32();
      break;
    case kPopNAndNonTopographical:
      Pop(p, static_cast<int>(bits->ReadUBitVarFieldPath()));
      for (int i = 0; i <= p->last; ++i) {
        if (bits->ReadBit()) p->path[i] += bits->ReadVarInt32();
      }
      break;

    case kNonTopoComplex:
      for (int i = 0; i <= p->last; ++i) {
        if (bits->ReadBit()) p->path[i] += bits->ReadVarInt32();
      }
      break;
    case kNonTopoPenultimatePlusOne:
      if (p->last > 0) p->path[p->last - 1] += 1;
      break;
    case kNonTopoComplexPack4Bits:
      for (int i = 0; i <= p->last; ++i) {
        if (bits->ReadBit()) p->path[i] += static_cast<int>(bits->ReadBits(4)) - 7;
      }
      break;

    case kFieldPathEncodeFinish:
      p->done = true;
      break;
    default:
      p->done = true;
      break;
  }
}

}  // namespace

const char* FieldPathOpName(int op) {
  if (op < 0 || op >= kFieldPathOpCount) return "?";
  return kOpNames[op];
}

std::string FieldPath::ToString() const {
  std::string out;
  for (int i = 0; i <= last; ++i) {
    if (i != 0) out += '/';
    out += std::to_string(path[i]);
  }
  return out;
}

const FieldPathHuffman& FieldPathHuffman::Instance() {
  static const FieldPathHuffman instance;
  return instance;
}

// Ordinary Huffman construction over the fixed weights. The one part that is
// not ordinary is the tie-break: equal weights are ordered by node value
// descending, and internal nodes are numbered from the leaf count upwards, so
// later-created nodes lose ties. Get that wrong and the tree is subtly
// different from the encoder's.
FieldPathHuffman::FieldPathHuffman() {
  struct Item {
    int weight;
    int value;  // node number, used only for tie-breaking
    int node;
  };
  struct Order {
    bool operator()(const Item& a, const Item& b) const {
      if (a.weight != b.weight) return a.weight > b.weight;  // min-heap
      return a.value < b.value;  // equal weights: higher value first
    }
  };

  nodes_.reserve(kFieldPathOpCount * 2);
  std::priority_queue<Item, std::vector<Item>, Order> heap;
  for (int op = 0; op < kFieldPathOpCount; ++op) {
    Node leaf;
    leaf.op = op;
    nodes_.push_back(leaf);
    // A zero weight would swallow the tie-break ordering, so it counts as one.
    heap.push(Item{std::max(kFieldPathWeights[op], 1), op,
                   static_cast<int>(nodes_.size()) - 1});
  }

  int next_value = kFieldPathOpCount;
  while (heap.size() > 1) {
    const Item a = heap.top();
    heap.pop();
    const Item b = heap.top();
    heap.pop();

    Node parent;
    parent.left = a.node;
    parent.right = b.node;
    nodes_.push_back(parent);
    heap.push(Item{a.weight + b.weight, next_value++,
                   static_cast<int>(nodes_.size()) - 1});
  }
  root_ = heap.empty() ? -1 : heap.top().node;
}

int FieldPathHuffman::ReadOp(BitReader* bits) const {
  int node = root_;
  while (node >= 0 && nodes_[static_cast<std::size_t>(node)].op < 0) {
    if (!bits->ok()) return -1;
    node = bits->ReadBit() ? nodes_[static_cast<std::size_t>(node)].right
                           : nodes_[static_cast<std::size_t>(node)].left;
  }
  if (node < 0 || !bits->ok()) return -1;
  return nodes_[static_cast<std::size_t>(node)].op;
}

std::string FieldPathHuffman::CodeFor(int op) const {
  std::string code;
  // Depth-first search for the leaf, recording the turns taken.
  struct Frame {
    int node;
    std::string code;
  };
  std::vector<Frame> stack{{root_, ""}};
  while (!stack.empty()) {
    const Frame frame = stack.back();
    stack.pop_back();
    if (frame.node < 0) continue;
    const Node& node = nodes_[static_cast<std::size_t>(frame.node)];
    if (node.op >= 0) {
      if (node.op == op) return frame.code;
      continue;
    }
    stack.push_back({node.left, frame.code + "0"});
    stack.push_back({node.right, frame.code + "1"});
  }
  return code;
}

int FieldPathHuffman::LeafCount() const {
  int leaves = 0;
  for (const Node& node : nodes_) {
    if (node.op >= 0) ++leaves;
  }
  return leaves;
}

bool ReadFieldPaths(BitReader* bits, std::vector<FieldPath>* out) {
  const FieldPathHuffman& huffman = FieldPathHuffman::Instance();
  FieldPath path;
  // A sane cap: an update touching thousands of fields means the stream has
  // gone wrong, and without this a bad tree loops until the bits run out.
  for (int guard = 0; guard < 4096; ++guard) {
    const int op = huffman.ReadOp(bits);
    if (op < 0) return false;
    Apply(op, bits, &path);
    if (path.done) return true;
    if (path.last < 0 || path.last >= 7) return false;
    out->push_back(path);
  }
  return false;
}

}  // namespace cs2mv
