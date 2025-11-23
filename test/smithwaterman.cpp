#include <Biovoltron/smithwaterman.hpp>
#include <Biovoltron/smithwaterman_simd.hpp>
#include <Biovoltron/smithwaterman_cuda.hpp>
#include "catch.hpp"
#include <chrono>
#include <iostream>
#include <fstream>
#include <thread>

using namespace biovoltron;
// ====================== Batch helpers ===========================
namespace {

struct BatchRunResult {
  long long duration_us;
  std::vector<SmithWatermanSimd::SWResult> results;
};

// baseline batch（多執行緒，內部用 scalar SmithWaterman::align）
BatchRunResult
run_baseline_batch_thread(const std::vector<std::string>& refs,
                          const std::vector<std::string>& alts,
                          SmithWatermanSimd::Parameters params) {
  const std::size_t n = std::min(refs.size(), alts.size());
  BatchRunResult out{};
  out.results.resize(n);

  if (n == 0) {
    out.duration_us = 0;
    return out;
  }

  // 轉成 scalar 版的參數
  SmithWaterman::Parameters scalar_params{
      params.w_match,
      params.w_mismatch,
      params.w_open,
      params.w_extend};

  // 決定 thread 數
  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 1;
  if (num_threads > n) num_threads = static_cast<unsigned int>(n);

  auto worker = [&](unsigned int tid) {
    const std::size_t chunk_size = (n + num_threads - 1) / num_threads;
    const std::size_t begin = tid * chunk_size;
    const std::size_t end   = std::min(n, begin + chunk_size);

    for (std::size_t i = begin; i < end; ++i) {
      auto [offset, cigar, score] =
          SmithWaterman::align(refs[i], alts[i], scalar_params);
      out.results[i] = SmithWatermanSimd::SWResult{offset, cigar, score};
    }
  };

  auto start = std::chrono::high_resolution_clock::now();

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (unsigned int t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker, t);
  }
  for (auto& th : threads) th.join();

  auto end = std::chrono::high_resolution_clock::now();
  out.duration_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  return out;
}

// XSIMD batch：直接呼叫 SmithWatermanSimd::batch_align（內部已多執行緒）
BatchRunResult
run_simd_batch(const std::vector<std::string>& refs,
               const std::vector<std::string>& alts,
               SmithWatermanSimd::Parameters params) {
  BatchRunResult out{};
  const std::size_t n = std::min(refs.size(), alts.size());
  if (n == 0) {
    out.duration_us = 0;
    return out;
  }

  auto start = std::chrono::high_resolution_clock::now();
  out.results = SmithWatermanSimd::batch_align(refs, alts, params);
  auto end = std::chrono::high_resolution_clock::now();

  out.duration_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  return out;
}

}  // namespace
// ==================== End Batch helpers =========================


TEST_CASE("SmithWaterman::align - Performs Smith-Waterman alignment", "[SmithWaterman]") 
{
  // define in cmake
  std::string data_path = DATA_PATH;

  // read ref.fasta
  std::ifstream fref(data_path + "/ref.fasta");
  REQUIRE(fref.good());

  std::string line, ref;
  while (std::getline(fref, line)) {
      if (!line.empty() && line[0] != '>') ref += line;
  }

  // read alt.fasta
  std::ifstream falt(data_path + "/alt.fasta");
  REQUIRE(falt.good());

  std::string alt, line_alt;
  while (std::getline(falt, line_alt)) {
      if (!line_alt.empty() && line_alt[0] != '>') alt += line_alt;
  }
  REQUIRE(ref.size() == alt.size());

  SECTION("Performance: Baseline vs. CUDA") 
  {
      int truth_offset;
      std::string truth_cigar;
      int truth_score;

      long long duration_baseline = 0;
      long long duration_cuda = 0;

      //
      // ===== CPU Smith-Waterman =====
      //
      {
          auto start  = std::chrono::high_resolution_clock::now();
          auto [offset1, cigar1, score1] =
              biovoltron::SmithWaterman::align(ref, alt, biovoltron::SmithWaterman::ORIGINAL_DEFAULT);
          auto end    = std::chrono::high_resolution_clock::now();

          duration_baseline =
              std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

          std::cout << "SmithWaterman CPU time = "
                    << duration_baseline << " us\n";

          std::cout << "CPU Offset = " << offset1 << "\n";
          std::cout << "CPU CIGAR  = " << cigar1 << "\n";
          std::cout << "CPU SCORE  = " << score1 << "\n";

          truth_offset = offset1;
          truth_cigar  = cigar1;
          truth_score  = score1;
      }

      //
      // ===== CUDA Smith-Waterman =====
      //
      {
          auto start  = std::chrono::high_resolution_clock::now();
          auto [offset2, cigar2, score2] =
              biovoltron::SmithWatermanCuda::align(ref, alt, biovoltron::SmithWatermanCuda::ORIGINAL_DEFAULT);
          auto end    = std::chrono::high_resolution_clock::now();

          duration_cuda =
              std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

          std::cout << "SmithWaterman CUDA time = "
                    << duration_cuda << " us\n";

          // ====== correctness checks ======
          CHECK(offset2 == truth_offset);
          CHECK(score2 == truth_score);
          CHECK(std::string(cigar2) == truth_cigar);

          // ====== speedup ======
          double speedup = static_cast<double>(duration_baseline) /
                          static_cast<double>(duration_cuda);

          std::cout << "Speedup (CPU / CUDA) = "
                    << speedup << "x\n";
      }

    //{
    //    // 設定 Batch 大小 (例如 1000 對，或是更多以測試極限吞吐量)
    //    const int BATCH_SIZE = 500; 
    //    std::cout << "\n[Batch Benchmark] Running with " << BATCH_SIZE << " pairs...\n";

    //    // 準備 Batch 資料 (複製同樣的資料多次)
    //    std::vector<std::string> batch_refs(BATCH_SIZE, ref);
    //    std::vector<std::string> batch_alts(BATCH_SIZE, alt);

    //    auto start_batch = std::chrono::high_resolution_clock::now();
        
    //    // 呼叫 batch_align
    //    auto results = biovoltron::SmithWatermanCuda::batch_align(
    //        batch_refs, 
    //        batch_alts, 
    //        biovoltron::SmithWatermanCuda::ORIGINAL_DEFAULT
    //    );

    //    auto end_batch = std::chrono::high_resolution_clock::now();
        
    //    long long duration_batch = 
    //        std::chrono::duration_cast<std::chrono::microseconds>(end_batch - start_batch).count();

    //    std::cout << "SmithWaterman CUDA Batch time = " << duration_batch << " us\n";
    //    std::cout << "Average time per pair         = " << (double)duration_batch / BATCH_SIZE << " us\n";

    //    // ====== Correctness Checks ======
    //    // 隨機抽查幾個結果 (例如第一個和最後一個) 確保正確性
    //    REQUIRE(results.size() == BATCH_SIZE);
        
    //    // Check first
    //    CHECK(results[0].offset == truth_offset);
    //    CHECK(results[0].score == truth_score);
    //    CHECK(std::string(results[0].cigar) == truth_cigar);

    //    // Check last
    //    CHECK(results.back().offset == truth_offset);
    //    CHECK(results.back().score == truth_score);
    //    CHECK(std::string(results.back().cigar) == truth_cigar);

    //    // ====== Speedup Calculation (vs CPU Serial) ======
    //    // 假設 CPU 是序列執行的，總時間 = 單次時間 * 數量
    //    double projected_cpu_time = static_cast<double>(duration_baseline) * BATCH_SIZE/32;
    //    double speedup_batch = projected_cpu_time / static_cast<double>(duration_batch);

    //    std::cout << "Projected Speedup (CPU Serial / CUDA Batch) = " << speedup_batch << "x\n";
        
    //    // 計算 GCUPS (Giga Cell Updates Per Second)
    //    // 這是衡量 SW 效能的標準指標
    //    long long total_cells = (long long)ref.size() * alt.size() * BATCH_SIZE;
    //    double gcups = (double)total_cells / (duration_batch * 1000.0); // us to s -> 1e6, but result is Giga (1e9) -> factor 1e3
    //    std::cout << "Performance = " << gcups << " GCUPS\n";
    //}
  }
   SECTION("Performance: Baseline vs. SIMD") 
  {
      int truth_offset;
      std::string truth_cigar;
      int truth_score;

      long long duration_baseline = 0;
      long long duration_simd     = 0;

      //
      // ===== CPU Smith-Waterman (baseline) =====
      //
      {
          auto start  = std::chrono::high_resolution_clock::now();
          auto [offset1, cigar1, score1] =
              biovoltron::SmithWaterman::align(
                  ref,
                  alt,
                  biovoltron::SmithWaterman::ORIGINAL_DEFAULT);
          auto end    = std::chrono::high_resolution_clock::now();

          duration_baseline =
              std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

          std::cout << "SmithWaterman CPU time (baseline) = "
                    << duration_baseline << " us\n";

          std::cout << "CPU Offset = " << offset1 << "\n";
          std::cout << "CPU CIGAR  = " << cigar1 << "\n";
          std::cout << "CPU SCORE  = " << score1 << "\n";

          // 把 CPU 結果當作 truth
          truth_offset = offset1;
          truth_cigar  = std::string(cigar1);
          truth_score  = score1;
      }

      //
      // ===== SIMD Smith-Waterman =====
      //
      {
          auto start  = std::chrono::high_resolution_clock::now();
          auto [offset2, cigar2, score2] =
              biovoltron::SmithWatermanSimd::align(
                  ref,
                  alt,
                  biovoltron::SmithWatermanSimd::ORIGINAL_DEFAULT);
          auto end    = std::chrono::high_resolution_clock::now();

          duration_simd =
              std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

          std::cout << "SmithWaterman SIMD time = "
                    << duration_simd << " us\n";

          // ====== correctness checks ======
          CHECK(offset2 == truth_offset);
          CHECK(score2  == truth_score);
          CHECK(std::string(cigar2) == truth_cigar);

          // ====== speedup ======
          double speedup =
              static_cast<double>(duration_baseline) /
              static_cast<double>(duration_simd);

          std::cout << "Speedup (CPU scalar / CPU SIMD) = "
                    << speedup << "x\n";
      }
  }
    SECTION("Performance: Batch baseline(thread) vs. SIMD batch") 
  {
    // ----- 準備 Batch 資料 -----
    const int BATCH_SIZE = 5;  // 我在自己筆電WSL上跑而已，太大會爆記憶體
    std::cout << "\n[Batch] Running " << BATCH_SIZE << " pairs...\n";

    std::vector<std::string> batch_refs(BATCH_SIZE, ref);
    std::vector<std::string> batch_alts(BATCH_SIZE, alt);

    // 統一用同一組 scoring 參數，這裡用 ORIGINAL_DEFAULT
    SmithWatermanSimd::Parameters params{
        SmithWatermanSimd::ORIGINAL_DEFAULT.w_match,
        SmithWatermanSimd::ORIGINAL_DEFAULT.w_mismatch,
        SmithWatermanSimd::ORIGINAL_DEFAULT.w_open,
        SmithWatermanSimd::ORIGINAL_DEFAULT.w_extend};

    // ----- baseline batch (scalar + 多執行緒) -----
    auto baseline_batch = run_baseline_batch_thread(batch_refs, batch_alts, params);

    // ----- SIMD batch (XSIMD + 多執行緒) -----
    auto simd_batch = run_simd_batch(batch_refs, batch_alts, params);

    // ----- 時間與 per-pair 統計 -----
    std::cout << "[Batch] Baseline (thread, scalar) time = "
              << baseline_batch.duration_us << " us\n";
    std::cout << "[Batch] SIMD batch time                = "
              << simd_batch.duration_us << " us\n";

    std::cout << "[Batch] Avg per pair (baseline)        = "
              << static_cast<double>(baseline_batch.duration_us) / BATCH_SIZE
              << " us\n";
    std::cout << "[Batch] Avg per pair (SIMD)            = "
              << static_cast<double>(simd_batch.duration_us) / BATCH_SIZE
              << " us\n";

    // ----- 正確性檢查 -----
    REQUIRE(simd_batch.results.size() == baseline_batch.results.size());
    REQUIRE(simd_batch.results.size() == static_cast<std::size_t>(BATCH_SIZE));

    // 抽查第一個
    CHECK(simd_batch.results[0].offset == baseline_batch.results[0].offset);
    CHECK(simd_batch.results[0].score  == baseline_batch.results[0].score);
    CHECK(std::string(simd_batch.results[0].cigar)
          == std::string(baseline_batch.results[0].cigar));

    // 抽查最後一個
    CHECK(simd_batch.results.back().offset == baseline_batch.results.back().offset);
    CHECK(simd_batch.results.back().score  == baseline_batch.results.back().score);
    CHECK(std::string(simd_batch.results.back().cigar)
          == std::string(baseline_batch.results.back().cigar));

    // ----- Speedup & GCUPS -----
    double speedup_batch =
        static_cast<double>(baseline_batch.duration_us) /
        static_cast<double>(simd_batch.duration_us);

    std::cout << "[Batch] Speedup (baseline(thread) / SIMD batch) = "
              << speedup_batch << "x\n";

    long long total_cells =
        static_cast<long long>(ref.size()) *
        static_cast<long long>(alt.size()) *
        static_cast<long long>(BATCH_SIZE);

    double gcups =
        static_cast<double>(total_cells) / (simd_batch.duration_us * 1000.0); 
    std::cout << "[Batch] SIMD batch performance = "
              << gcups << " GCUPS\n";
  }


}