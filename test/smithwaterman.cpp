#include "catch.hpp"
#include "support/backend.hpp"
#include <stdexcept>
using namespace swtest;
namespace {
// Replay the public overlap CIGAR against its input, independently of traceback.
void check_overlap_trace(const Pair& pair, const Result& result, Scoring p) {
  REQUIRE(result.offset >= 0);
  std::size_t ref = static_cast<std::size_t>(result.offset), query = 0;
  int score = 0;
  const auto& cigar = result.cigar;
  REQUIRE_FALSE(cigar.empty());
  for (std::size_t pos = 0; pos < cigar.size();) {
    std::size_t count = 0;
    REQUIRE(cigar[pos] >= '0'); REQUIRE(cigar[pos] <= '9');
    while (pos < cigar.size() && cigar[pos] >= '0' && cigar[pos] <= '9')
      count = count * 10 + static_cast<unsigned>(cigar[pos++] - '0');
    REQUIRE(count > 0); REQUIRE(pos < cigar.size());
    const char op = cigar[pos++];
    REQUIRE((op == 'M' || op == 'I' || op == 'D' || op == 'S'));
    if (op == 'M' || op == 'D') REQUIRE(ref + count <= pair.ref.size());
    if (op == 'M' || op == 'I' || op == 'S') REQUIRE(query + count <= pair.query.size());
    if (op == 'M') {
      for (std::size_t j = 0; j < count; ++j)
        score += pair.ref[ref++] == pair.query[query++] ? p.match : p.mismatch;
    } else if (op == 'I' || op == 'D') {
      score += p.open + static_cast<int>(count - 1) * p.extend;
      if (op == 'I') query += count; else ref += count;
    } else {
      REQUIRE((query == 0 || pos == cigar.size()));
      query += count;
    }
  }
  CHECK(query == pair.query.size());
  CHECK(score == result.score);
}
}  // namespace

TEST_CASE("Reference recurrence has explicit alignment semantics", "[cpu][oracle]") {
  CHECK(reference_score({"ACGT", "ACGT"}, {}, Model::local) == 12);
  CHECK(reference_score({"AAAA", "TTTT"}, {}, Model::local) == 0);
  CHECK(reference_score({"AAAA", "TTTT"}, {}, Model::overlap) == -1);
  CHECK(reference_score({"ACGTTGCA", "ACGTGCA"}, {}, Model::local) == 17);
  CHECK(reference_score({"TTACGTAA", "GGACGTCC"}, {}, Model::local) == 12);
  CHECK(reference_score({"TTACGTAA", "GGACGTCC"}, {}, Model::overlap) == 8);
}
TEST_CASE("Adapter rejects ambiguous and unsafe inputs", "[cpu][contract]") {
  CHECK_THROWS_AS(prepare(Backend::scalar, {{"", "A"}}), std::invalid_argument);
  CHECK_THROWS_AS(prepare(Backend::simd, {{"AN", "AA"}}), std::invalid_argument);
  CHECK_THROWS_AS(prepare(Backend::scalar, {{"ac", "AC"}}), std::invalid_argument);
  CHECK_THROWS_AS(prepare(Backend::scalar, {{"AC", "AC"}}, {0,-1,-4,-3}), std::invalid_argument);
  Options options;
  options.threads = 0;
  CHECK_THROWS_AS(prepare(Backend::scalar, {{"AC", "AC"}}, {}, options), std::invalid_argument);
  options.threads = 2; options.execution = Execution::single;
  CHECK_THROWS_AS(prepare(Backend::scalar, {{"AC", "AC"}}, {}, options), std::invalid_argument);
  options = {}; options.max_cells = 8;
  CHECK_THROWS_AS(prepare(Backend::scalar, {{"AC", "AC"}}, {}, options), std::length_error);
  CHECK_THROWS_AS(prepare(Backend::simd, {{std::string(200,'A'),std::string(200,'A')}},
                         {200,-150,-260,-11}), std::length_error);
  CHECK(prepare(Backend::scalar, {})->run().empty());
  CHECK(prepare(Backend::simd, {})->run().empty());
}
TEST_CASE("Seeded datasets are reproducible and varied", "[cpu][data]") {
  const auto a = generate(4, 64, 70, 42);
  const auto b = generate(4, 64, 70, 42);
  const auto c = generate(4, 64, 70, 43);
  REQUIRE(a.size() == 4);
  for (std::size_t i = 0; i < a.size(); ++i) {
    CHECK(a[i].ref == b[i].ref); CHECK(a[i].query == b[i].query);
    CHECK(a[i].ref != c[i].ref); CHECK(a[i].ref != a[i].query);
  }
  CHECK(a[0].ref != a[1].ref);
  CHECK(cells(a) == 4 * 65 * 65);
  const auto identical = generate(2, 32, 100, 9);
  CHECK(identical[0].ref == identical[0].query);
  const auto different = generate(1, 32, 0, 9);
  for (std::size_t i = 0; i < 32; ++i) CHECK(different[0].ref[i] != different[0].query[i]);
  CHECK_THROWS_AS(generate(1, 0, 70, 42), std::invalid_argument);
  CHECK_THROWS_AS(generate(1, 32, 101, 42), std::invalid_argument);
}
TEST_CASE("Exact match result includes offset CIGAR and score", "[cpu][alignment]") {
  for (const auto backend : {Backend::scalar, Backend::simd}) {
    INFO(describe(backend).name);
    auto results = prepare(backend, {{"ACGTACGT", "ACGTACGT"}})->run();
    REQUIRE(results.size() == 1);
    CHECK(results[0].offset == 0);
    CHECK(results[0].cigar == "8M");
    CHECK(results[0].score == 24);
  }
}
TEST_CASE("Scalar DP agrees exactly with independent overlap oracle", "[cpu][alignment]") {
  for (const auto scoring : {Scoring{}, Scoring{10,-15,-30,-5}}) {
    auto batch = generate(32, 37, 65, 179);
    // Unequal lengths exercise DP instead of the legacy <=2-mismatch shortcut.
    for (std::size_t i = 0; i < batch.size(); ++i) batch[i].query.resize(25 + i % 12);
    const auto results = prepare(Backend::scalar, batch, scoring)->run();
    REQUIRE(results.size() == batch.size());
    for (std::size_t i = 0; i < batch.size(); ++i) {
      INFO("pair=" << i);
      CHECK(results[i].score == reference_score(batch[i], scoring, Model::overlap));
      CHECK(results[i].offset >= 0);
      CHECK(results[i].offset < static_cast<int>(batch[i].ref.size()));
      CHECK_FALSE(results[i].cigar.empty());
      check_overlap_trace(batch[i], results[i], scoring);
    }
  }
}
TEST_CASE("CPU batch preserves every result and order across thread counts", "[cpu][batch]") {
  auto batch = generate(9, 48, 75, 381);
  for (auto backend : {Backend::scalar, Backend::simd}) {
    const auto expected = prepare(backend, batch)->run();
    for (auto threads : {2u, 4u, 16u}) {
      Options options; options.threads = threads;
      auto work = prepare(backend, batch, {}, options);
      for (int repetition = 0; repetition < 2; ++repetition) {
        const auto actual = work->run();
        REQUIRE(actual.size() == expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i) {
          CHECK(actual[i].score == expected[i].score);
          CHECK(actual[i].offset == expected[i].offset);
          CHECK(actual[i].cigar == expected[i].cigar);
        }
      }
    }
  }
}
// Explicit opt-in regressions: failures report existing algorithm defects.
// Never mark these WILL_FAIL: a repaired implementation should pass normally.
TEST_CASE("Scalar fast path must charge mismatches", "[.known-defect][known-defect]") {
  const Pair pair{"ACGTACGT", "ACGAACGT"};
  const auto result = prepare(Backend::scalar, {pair})->run().front();
  CHECK(result.score == reference_score(pair, {}, Model::overlap));
}
TEST_CASE("SIMD score agrees with independent local oracle", "[cpu][alignment]") {
  auto batch = generate(64, 37, 55, 1024);
  for (std::size_t i = 0; i < batch.size(); ++i) batch[i].query.resize(17 + i % 20);
  const auto results = prepare(Backend::simd, batch)->run();
  for (std::size_t i = 0; i < batch.size(); ++i) {
    INFO("seed=1024 pair=" << i);
    CHECK(results[i].score == reference_score(batch[i], {}, Model::local));
  }
}
