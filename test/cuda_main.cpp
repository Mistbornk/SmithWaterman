#include "support/backend.hpp"
#include <iostream>
#include <stdexcept>
using namespace swtest;
int main() {
  const auto reason = unavailable_reason(Backend::cuda);
  if (!reason.empty()) { std::cout << "SKIP: " << reason << '\n'; return 77; }
  try {
    // Warp boundaries, heterogeneous lengths, batch=1 and multiple scoring schemes.
    for (const auto scoring : {Scoring{}, Scoring{10,-15,-30,-5}}) {
      Batch batch;
      for (auto length : {1u, 7u, 31u, 32u, 33u, 63u, 64u, 65u}) {
        auto pair = generate(1, length, 60, 919 + length).front();
        pair.query += "ACG"; // Force the kernel, even for single-pair execution.
        batch.push_back(std::move(pair));
      }
      for (auto count : {std::size_t(1), batch.size()}) {
        Batch input(batch.begin(), batch.begin() + count);
        const auto expected = prepare(Backend::scalar, input, scoring)->run();
        for (auto mode : {Execution::single, Execution::batch}) {
          Options options; options.execution = mode;
          auto work = prepare(Backend::cuda, input, scoring, options);
          for (int repeat = 0; repeat < 2; ++repeat) {
            const auto results = work->run();
            if (results.size() != input.size()) throw std::runtime_error("Wrong result count");
            for (std::size_t i = 0; i < input.size(); ++i) {
              if (results[i].score != reference_score(input[i], scoring, Model::overlap) ||
                  results[i].offset != expected[i].offset || results[i].cigar != expected[i].cigar) {
                std::cerr << "Mismatch pair=" << i << " length=" << input[i].ref.size()
                          << " mode=" << (mode == Execution::single ? "single" : "batch")
                          << " expected=" << expected[i].score << '/' << expected[i].offset << '/' << expected[i].cigar
                          << " actual=" << results[i].score << '/' << results[i].offset << '/' << results[i].cigar << '\n';
                return 1;
              }
            }
          }
        }
      }
    }
    std::cout << "CUDA correctness checks passed\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
