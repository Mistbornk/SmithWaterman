#include "backend.hpp"
#include <Biovoltron/smithwaterman.hpp>
#include <Biovoltron/smithwaterman_simd.hpp>
#if SW_HAS_CUDA
#include <Biovoltron/smithwaterman_cuda.hpp>
#endif
#include <algorithm>
#include <future>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace swtest {
Descriptor describe(Backend b) {
  switch (b) {
    case Backend::scalar: return {"scalar", Model::overlap, 32};
    case Backend::simd: return {"simd", Model::local, 16};
    case Backend::cuda: return {"cuda", Model::overlap, 32};
  }
  throw std::invalid_argument("Unknown backend");
}
std::string unavailable_reason(Backend b) {
  if (b != Backend::cuda) return {};
#if SW_HAS_CUDA
  int count = 0;
  auto status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) return cudaGetErrorString(status);
  if (count == 0) return "No CUDA device";
  return {};
#else
  return "Built with SW_ENABLE_CUDA=OFF";
#endif
}
std::uint64_t cells(const Batch& batch) {
  std::uint64_t total = 0;
  for (const auto& pair : batch) {
    const auto n = static_cast<std::uint64_t>(pair.ref.size()) + 1;
    const auto m = static_cast<std::uint64_t>(pair.query.size()) + 1;
    if (n > (UINT64_MAX - total) / m) throw std::length_error("Cell count overflow");
    total += n * m;
  }
  return total;
}
namespace {
template<class T> Result convert(T result) {
  return {result.offset, std::string(result.cigar), result.score};
}
class LegacyWork final : public PreparedWork {
  Backend backend_;
  Batch batch_;
  Scoring scoring_;
  Options options_;
  std::vector<std::string> refs_, queries_;
 public:
  LegacyWork(Backend b, Batch batch, Scoring p, Options options)
      : backend_(b), batch_(std::move(batch)), scoring_(p), options_(options) {
    if (b == Backend::cuda && options.execution == Execution::batch) {
      for (const auto& pair : batch_) {
        refs_.push_back(pair.ref);
        queries_.push_back(pair.query);
      }
    }
  }
  std::vector<Result> run() override {
    const auto p = scoring_;
#if SW_HAS_CUDA
    if (backend_ == Backend::cuda) {
      std::vector<Result> out;
      out.reserve(batch_.size());
      const auto params = biovoltron::SmithWatermanCuda::Parameters{p.match,p.mismatch,p.open,p.extend};
      if (options_.execution == Execution::batch) {
        for (auto& result : biovoltron::SmithWatermanCuda::batch_align(refs_, queries_, params))
          out.push_back(convert(std::move(result)));
      } else {
        for (const auto& pair : batch_)
          out.push_back(convert(biovoltron::SmithWatermanCuda::align(pair.ref, pair.query, params)));
      }
      auto status = cudaDeviceSynchronize();
      if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
      if (out.size() != batch_.size()) throw std::runtime_error("Backend returned wrong result count");
      return out;
    }
#endif
    std::vector<Result> out(batch_.size());
    if (batch_.empty()) return out;
    const auto workers = std::min<std::size_t>(options_.threads, batch_.size());
    auto worker = [&](std::size_t id) {
      for (std::size_t i = id; i < batch_.size(); i += workers) {
        const auto& pair = batch_[i];
        if (backend_ == Backend::scalar)
          out[i] = convert(biovoltron::SmithWaterman::align(pair.ref, pair.query,
              {p.match,p.mismatch,p.open,p.extend}));
        else
          out[i] = convert(biovoltron::SmithWatermanSimd::align(pair.ref, pair.query,
              {p.match,p.mismatch,p.open,p.extend}));
      }
    };
    if (workers == 1) worker(0);
    else {
      std::vector<std::future<void>> tasks;
      for (std::size_t i = 0; i < workers; ++i)
        tasks.push_back(std::async(std::launch::async, worker, i));
      for (auto& task : tasks) task.get();
    }
    return out;
  }
};
}
std::unique_ptr<PreparedWork> prepare(Backend b, Batch batch, Scoring p, Options options) {
  const auto desc = describe(b);
  if (options.execution != Execution::single && options.execution != Execution::batch)
    throw std::invalid_argument("Unknown execution mode");
  if (!options.threads || options.threads > 256 || !options.max_cells)
    throw std::invalid_argument("threads must be 1..256 and max_cells must be positive");
  if ((b == Backend::cuda || options.execution == Execution::single) && options.threads != 1)
    throw std::invalid_argument("CUDA and single mode require threads=1");
  if (p.match <= 0 || p.mismatch > 0 || p.open >= 0 || p.extend >= 0)
    throw std::invalid_argument("Expected match>0, mismatch<=0, open<0, extend<0");
  // Conservative range bound also covers negative intermediate DP values.
  const std::int64_t magnitude = std::max({std::int64_t(p.match), -std::int64_t(p.mismatch),
                                         -std::int64_t(p.open), -std::int64_t(p.extend)});
  const auto limit = desc.score_bits == 16 ? INT16_MAX / 2 : INT32_MAX / 4;
  std::uint64_t ref_total = 0, query_total = 0;
  for (const auto& pair : batch) {
    if (pair.ref.empty() || pair.query.empty()) throw std::invalid_argument("Empty sequences are unsupported");
    for (const auto* seq : {&pair.ref, &pair.query}) {
      if (seq->find_first_not_of("ACGT") != std::string::npos)
        throw std::invalid_argument("Only uppercase A/C/G/T is supported");
      if (seq->size() > static_cast<std::size_t>(INT32_MAX - 1))
        throw std::length_error("Sequence exceeds backend index range");
    }
    if (pair.ref.size() + pair.query.size() > static_cast<std::uint64_t>(limit / magnitude))
      throw std::length_error("Scoring may overflow backend arithmetic");
    ref_total += pair.ref.size(); query_total += pair.query.size();
    if (ref_total > INT32_MAX || query_total > INT32_MAX)
      throw std::length_error("Packed input exceeds backend index range");
  }
  if (batch.size() > INT32_MAX || cells(batch) > options.max_cells)
    throw std::length_error("Batch exceeds configured cell budget");
  const auto reason = unavailable_reason(b);
  if (!reason.empty()) throw std::runtime_error(reason);
  return std::make_unique<LegacyWork>(b, std::move(batch), p, options);
}
Batch generate(std::size_t count, std::size_t length, unsigned identity, std::uint32_t seed) {
  if (!length || identity > 100) throw std::invalid_argument("Invalid dataset dimensions/identity");
  std::mt19937 rng(seed);
  const std::string bases = "ACGT";
  Batch batch;
  batch.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    Pair pair;
    for (std::size_t j = 0; j < length; ++j) {
      unsigned base = rng() % 4;
      pair.ref += bases[base];
      pair.query += bases[(rng() % 100 < identity) ? base : (base + 1 + rng() % 3) % 4];
    }
    batch.push_back(std::move(pair));
  }
  return batch;
}
int reference_score(const Pair& pair, Scoring p, Model model) {
  if (pair.ref.empty() || pair.query.empty()) return 0;
  const std::int64_t neg = std::numeric_limits<std::int64_t>::min() / 4;
  const auto m = pair.query.size();
  std::vector<std::int64_t> prev(m + 1), curr(m + 1), vertical(m + 1, neg);
  std::int64_t best = model == Model::local ? 0 : neg;
  for (std::size_t i = 1; i <= pair.ref.size(); ++i) {
    std::int64_t horizontal = neg;
    curr[0] = 0;
    for (std::size_t j = 1; j <= m; ++j) {
      vertical[j] = std::max(prev[j] + p.open, vertical[j] + p.extend);
      horizontal = std::max(curr[j-1] + p.open, horizontal + p.extend);
      curr[j] = std::max({prev[j-1] + (pair.ref[i-1] == pair.query[j-1] ? p.match : p.mismatch),
                          vertical[j], horizontal});
      if (model == Model::local) curr[j] = std::max<std::int64_t>(0, curr[j]);
      if (model == Model::local || i == pair.ref.size() || j == m)
        best = std::max(best, curr[j]);
    }
    prev.swap(curr);
  }
  if (best < INT32_MIN || best > INT32_MAX) throw std::overflow_error("Reference score overflow");
  return static_cast<int>(best);
}
}  // namespace swtest
