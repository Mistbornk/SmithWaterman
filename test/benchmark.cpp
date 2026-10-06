#include "support/backend.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
using namespace swtest;
namespace {
std::uint64_t number(const std::string& text) {
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("Expected unsigned integer: " + text);
  std::size_t used = 0;
  const auto n = std::stoull(text, &used);
  if (used != text.size()) throw std::invalid_argument("Invalid number");
  return n;
}
}
int main(int argc, char** argv) {
  try {
    Backend backend = Backend::scalar;
    Options options;
    std::uint64_t count = 16, length = 128, identity = 70, seed = 42;
    std::uint64_t warmups = 2, repetitions = 7;
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (key == "--help") {
        std::cout << "sw-benchmark [--backend scalar|simd|cuda] [--mode single|batch]\n"
                     "  [--batch 16] [--length 128] [--identity 70] [--seed 42]\n"
                     "  [--threads 1] [--warmup 2] [--repetitions 7] [--max-cells 16000000]\n"
                     "CSV reports synchronous end-to-end host latency, not kernel time.\n";
        return 0;
      }
      if (++i == argc) throw std::invalid_argument("Missing value for " + key);
      const std::string value = argv[i];
      if (key == "--backend") {
        if (value == "scalar") backend = Backend::scalar;
        else if (value == "simd") backend = Backend::simd;
        else if (value == "cuda") backend = Backend::cuda;
        else throw std::invalid_argument("Unknown backend: " + value);
      } else if (key == "--mode") {
        if (value == "single") options.execution = Execution::single;
        else if (value == "batch") options.execution = Execution::batch;
        else throw std::invalid_argument("Unknown mode: " + value);
      } else {
        const auto n = number(value);
        if (key == "--batch") count = n;
        else if (key == "--length") length = n;
        else if (key == "--identity") identity = n;
        else if (key == "--seed") seed = n;
        else if (key == "--warmup") warmups = n;
        else if (key == "--repetitions") repetitions = n;
        else if (key == "--max-cells") options.max_cells = n;
        else if (key == "--threads") {
          if (n > std::numeric_limits<unsigned>::max()) throw std::invalid_argument("threads out of range");
          options.threads = static_cast<unsigned>(n);
        } else throw std::invalid_argument("Unknown option: " + key);
      }
    }
    if (!count || !length || !repetitions || identity > 100 || seed > UINT32_MAX ||
        length >= INT32_MAX || count > INT32_MAX || repetitions > 100000 || warmups > 100000)
      throw std::invalid_argument("Invalid benchmark dimensions, seed or iteration count");
    // Check before generation/allocation, using division to avoid overflow.
    if (length + 1 > options.max_cells / (length + 1) / count)
      throw std::length_error("Dataset exceeds --max-cells");
    std::cerr << "compiler=" << __VERSION__ << " hardware_threads=" << std::thread::hardware_concurrency()
#ifdef NDEBUG
              << " assertions=off"
#else
              << " assertions=on"
#endif
              << " cuda_build=" << SW_HAS_CUDA << " scoring=3,-1,-4,-3\n";
    const auto input = generate(count, length, identity, seed);
    auto work = prepare(backend, input, {}, options);
    std::uint64_t checksum = 0;
    auto consume = [&](const std::vector<Result>& results) {
      if (results.size() != input.size()) throw std::runtime_error("Wrong result count");
      for (const auto& result : results) {
        checksum = checksum * 131 + static_cast<std::uint64_t>(result.score);
        checksum = checksum * 131 + static_cast<std::uint64_t>(result.offset);
        for (unsigned char c : result.cigar) checksum = checksum * 131 + c;
      }
    };
    for (std::uint64_t i = 0; i < warmups; ++i) consume(work->run());
    std::vector<double> samples;
    for (std::uint64_t i = 0; i < repetitions; ++i) {
      const auto start = std::chrono::steady_clock::now();
      const auto results = work->run();
      const auto end = std::chrono::steady_clock::now();
      samples.push_back(std::chrono::duration<double, std::micro>(end-start).count());
      consume(results); // Outside timed region; all results remain observable.
    }
    std::sort(samples.begin(), samples.end());
    const auto mid = samples.size() / 2;
    const double median = samples.size() % 2 ? samples[mid] : (samples[mid-1] + samples[mid]) / 2;
    const double p95 = samples[static_cast<std::size_t>(std::ceil(samples.size() * .95)) - 1];
    if (median <= 0) throw std::runtime_error("Clock resolution insufficient; increase workload");
    const auto desc = describe(backend);
    std::cout << "backend,model,mode,threads_requested,threads_effective,batch,length,identity,seed,warmups,repetitions,scope,min_us,median_us,p95_us,pairs_per_s,nominal_cells_per_s,checksum\n";
    std::cout << desc.name << ',' << (desc.model == Model::local ? "local" : "overlap") << ','
              << (options.execution == Execution::single ? "single" : "batch") << ','
              << options.threads << ',' << std::min<std::uint64_t>(options.threads, count) << ',' << count << ',' << length << ',' << identity << ',' << seed << ','
              << warmups << ',' << repetitions << ",end_to_end," << samples.front() << ',' << median << ','
              << p95 << ',' << count * 1e6 / median << ','
              << static_cast<double>(count) * length * length * 1e6 / median << ',' << checksum << '\n';
    return 0;
  } catch (const std::exception& e) { std::cerr << "benchmark: " << e.what() << '\n'; return 1; }
}
