#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace swtest {
enum class Backend { scalar, simd, cuda };
enum class Model { overlap, local };
enum class Execution { single, batch };
struct Scoring { int match = 3, mismatch = -1, open = -4, extend = -3; };
struct Pair { std::string ref, query; };
using Batch = std::vector<Pair>;
struct Result { int offset; std::string cigar; int score; };
struct Options {
  Execution execution = Execution::batch;
  unsigned threads = 1;
  // Conservative sum across the batch, including matrix boundaries.
  std::uint64_t max_cells = 16'000'000;
};
struct Descriptor { const char* name; Model model; int score_bits; };
Descriptor describe(Backend backend);
// Empty string means available; other values explain unavailable build/device.
std::string unavailable_reason(Backend backend);

// Owns immutable host input. run() returns ordered, host-ready results and must
// complete all device work before returning. Calls on one object are sequential.
// Preparation is outside timing; allocation, transfers, dispatch, traceback and
// result construction inside run() are included. No hidden batch=1 mode switch.
class PreparedWork {
 public:
  virtual ~PreparedWork() = default;
  virtual std::vector<Result> run() = 0;
};
std::unique_ptr<PreparedWork> prepare(Backend, Batch, Scoring = {}, Options = {});
Batch generate(std::size_t count, std::size_t length, unsigned identity,
               std::uint32_t seed);
// Independent affine-gap DP; gap cost = open + (length - 1) * extend.
// overlap: zero boundaries, no zero clamp, best last row/column.
// local: zero clamp, best cell. This is not the legacy scalar fast path.
int reference_score(const Pair&, Scoring, Model);
std::uint64_t cells(const Batch&);
}  // namespace swtest
