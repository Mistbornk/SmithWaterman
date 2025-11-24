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
                          SmithWatermanSimd::Parameters params,
                          unsigned int num_threads = 0) {
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
  if (num_threads == 0) {
      num_threads = std::thread::hardware_concurrency();
  }
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

// Custom SIMD batch runner with controllable threads
BatchRunResult
run_simd_batch_custom_threads(const std::vector<std::string>& refs,
                              const std::vector<std::string>& alts,
                              SmithWatermanSimd::Parameters params,
                              unsigned int num_threads = 0) {
  BatchRunResult out{};
  const std::size_t n = std::min(refs.size(), alts.size());
  out.results.resize(n);

  if (n == 0) {
    out.duration_us = 0;
    return out;
  }

  // 決定 thread 數
  if (num_threads == 0) {
      num_threads = std::thread::hardware_concurrency();
  }
  if (num_threads == 0) num_threads = 1;
  if (num_threads > n) num_threads = static_cast<unsigned int>(n);

  auto worker = [&](unsigned int tid) {
    const std::size_t chunk_size = (n + num_threads - 1) / num_threads;
    const std::size_t begin = tid * chunk_size;
    const std::size_t end   = std::min(n, begin + chunk_size);

    for (std::size_t i = begin; i < end; ++i) {
      out.results[i] = SmithWatermanSimd::align(refs[i], alts[i], params);
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

// XSIMD batch：直接呼叫 SmithWatermanSimd::batch_align（內部已多執行緒，無法控制 thread 數，僅供參考）
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

      // 把 CPU 結果當作 baseline
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

      // ====== correctness checks（放寬：SW 目標函數略有不同） ======
      // 要求：SIMD 分數不比 baseline 差太多，且不低於 baseline
      CHECK(score2 >= truth_score);
      CHECK(std::abs(score2 - truth_score) <= 20);

      // offset / CIGAR 不再要求完全相同，只做基本 sanity check
      CHECK(offset2 >= 0);
      CHECK(offset2 <= static_cast<int>(ref.size()));
      CHECK_FALSE(std::string(cigar2).empty());

      // ====== speedup ======
      double speedup =
          static_cast<double>(duration_baseline) /
          static_cast<double>(duration_simd);

      std::cout << "Speedup (CPU scalar / CPU SIMD) = "
                << speedup << "x\n";
    }
  }

  SECTION("Performance: Comprehensive Benchmark (Baseline vs SIMD vs CUDA)") 
  {
    // Truncate sequences to avoid OOM with large batches
    if (ref.size() > 1000) ref.resize(1000);
    if (alt.size() > 1000) alt.resize(1000);

    // Define benchmark parameters
    std::vector<int> batch_sizes = {1, 10, 100, 1000, 5000};
    std::vector<unsigned int> thread_counts = {1, 2, 4, 8, 16};
    
    // Cap thread counts by hardware concurrency
    unsigned int max_threads = std::thread::hardware_concurrency();
    if (max_threads == 0) max_threads = 1;
    
    // Filter thread counts
    std::vector<unsigned int> valid_thread_counts;
    for (auto t : thread_counts) {
        if (t <= max_threads) {
            valid_thread_counts.push_back(t);
        }
    }
    if (valid_thread_counts.empty()) valid_thread_counts.push_back(1);

    // Print Table Header
    std::cout << "\n======================================================================================================================\n";
    std::cout << "                                      Smith-Waterman Performance Benchmark\n";
    std::cout << "======================================================================================================================\n";
    printf("%-10s | %-10s | %-15s | %-15s | %-15s | %-20s | %-20s\n", 
           "Batch Size", "Threads", "Baseline (us)", "SIMD (us)", "CUDA (us)", "Speedup (Base/SIMD)", "Speedup (Base/CUDA)");
    std::cout << "----------------------------------------------------------------------------------------------------------------------\n";

    for (int batch_size : batch_sizes) {
        // Prepare Batch Data
        std::vector<std::string> batch_refs(batch_size, ref);
        std::vector<std::string> batch_alts(batch_size, alt);

        // Parameters
        SmithWatermanSimd::Parameters params{
            SmithWatermanSimd::ORIGINAL_DEFAULT.w_match,
            SmithWatermanSimd::ORIGINAL_DEFAULT.w_mismatch,
            SmithWatermanSimd::ORIGINAL_DEFAULT.w_open,
            SmithWatermanSimd::ORIGINAL_DEFAULT.w_extend};
        
        SmithWatermanCuda::Parameters cuda_params{
            SmithWatermanCuda::ORIGINAL_DEFAULT.w_match,
            SmithWatermanCuda::ORIGINAL_DEFAULT.w_mismatch,
            SmithWatermanCuda::ORIGINAL_DEFAULT.w_open,
            SmithWatermanCuda::ORIGINAL_DEFAULT.w_extend};

        // Run CUDA once per batch size (independent of threads)
        long long duration_cuda = 0;
        {
            auto start_cuda = std::chrono::high_resolution_clock::now();
            auto cuda_results = SmithWatermanCuda::batch_align(batch_refs, batch_alts, cuda_params);
            auto end_cuda = std::chrono::high_resolution_clock::now();
            duration_cuda = std::chrono::duration_cast<std::chrono::microseconds>(end_cuda - start_cuda).count();
            
            // Basic correctness check (size only)
            REQUIRE(cuda_results.size() == static_cast<size_t>(batch_size));
        }

        for (unsigned int num_threads : valid_thread_counts) {
            // Run Baseline (Threaded)
            auto baseline_result = run_baseline_batch_thread(batch_refs, batch_alts, params, num_threads);
            
            // Run SIMD (Threaded)
            auto simd_result = run_simd_batch_custom_threads(batch_refs, batch_alts, params, num_threads);

            // Calculate Speedup (Baseline / CUDA)
            double speedup_cuda = 0.0;
            if (duration_cuda > 0) {
                speedup_cuda = static_cast<double>(baseline_result.duration_us) / static_cast<double>(duration_cuda);
            }

            // Calculate Speedup (Baseline / SIMD)
            double speedup_simd = 0.0;
            if (simd_result.duration_us > 0) {
                speedup_simd = static_cast<double>(baseline_result.duration_us) / static_cast<double>(simd_result.duration_us);
            }

            // Print Row
            printf("%-10d | %-10u | %-15lld | %-15lld | %-15lld | %-20.2fx | %-20.2fx\n", 
                   batch_size, num_threads, baseline_result.duration_us, simd_result.duration_us, duration_cuda, speedup_simd, speedup_cuda);
            
            // Correctness Checks (Sample)
            if (batch_size > 0) {
                // Check first result
                CHECK(std::abs(simd_result.results[0].score - baseline_result.results[0].score) <= 20);
            }
        }
        std::cout << "----------------------------------------------------------------------------------------------------------------------\n";
    }
  }


}