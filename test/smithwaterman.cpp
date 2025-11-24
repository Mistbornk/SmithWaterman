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


BatchRunResult
run_cuda_batch(const std::vector<std::string>& refs,
               const std::vector<std::string>& alts,
               SmithWatermanSimd::Parameters params) {
  BatchRunResult out{};
  
  // 1. 檢查輸入大小
  if (alts.empty()) {
      out.duration_us = 0;
      return out;
  }

  // 2. 轉換參數 (SIMD Params -> CUDA Params)
  biovoltron::SmithWatermanCuda::Parameters cuda_params{
      params.w_match,
      params.w_mismatch,
      params.w_open,
      params.w_extend
  };

  std::vector<biovoltron::SmithWatermanCuda::SWResult> cuda_results;
  auto start = std::chrono::high_resolution_clock::now();


  cuda_results = biovoltron::SmithWatermanCuda::batch_align(refs, alts, cuda_params);

  auto end = std::chrono::high_resolution_clock::now();
  out.duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  out.results.resize(cuda_results.size());
  for (size_t i = 0; i < cuda_results.size(); ++i) {
      out.results[i].offset = cuda_results[i].offset;
      out.results[i].score  = cuda_results[i].score;
      out.results[i].cigar  = cuda_results[i].cigar;
  }

  return out;
}
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

  SECTION("Performance: Batch baseline(thread) vs. SIMD batch vs. CUDA batch") 
  {
    const int BATCH_SIZE = 500;
    std::cout << "\n[Batch] Running " << BATCH_SIZE << " pairs...\n";

    std::vector<std::string> batch_refs_expanded(1, ref);
    std::vector<std::string> batch_alts(BATCH_SIZE, alt);

    SmithWatermanSimd::Parameters params{
        SmithWatermanSimd::ORIGINAL_DEFAULT.w_match,
        SmithWatermanSimd::ORIGINAL_DEFAULT.w_mismatch,
        SmithWatermanSimd::ORIGINAL_DEFAULT.w_open,
        SmithWatermanSimd::ORIGINAL_DEFAULT.w_extend};

    auto baseline_batch = run_baseline_batch_thread(batch_refs_expanded, batch_alts, params);

    auto simd_batch = run_simd_batch(batch_refs_expanded, batch_alts, params);

    auto cuda_batch = run_cuda_batch(batch_refs_expanded, batch_alts, params);

    // ----- 報告 -----
    std::cout << "----------------------------------------------------------\n";
    std::cout << "[Batch] Baseline (Thread) time = " << baseline_batch.duration_us << " us\n";
    std::cout << "[Batch] SIMD Batch time        = " << simd_batch.duration_us << " us\n";
    std::cout << "[Batch] CUDA Batch time        = " << cuda_batch.duration_us << " us\n";
    std::cout << "----------------------------------------------------------\n";

    // ----- Speedup -----
    double speedup_simd = static_cast<double>(baseline_batch.duration_us) / simd_batch.duration_us;
    double speedup_cuda = static_cast<double>(baseline_batch.duration_us) / cuda_batch.duration_us;

    std::cout << "Speedup (Baseline / SIMD): " << speedup_simd << "x\n";
    std::cout << "Speedup (Baseline / CUDA): " << speedup_cuda << "x\n";

    // ----- GCUPS -----
    long long total_cells = static_cast<long long>(ref.size()) * alt.size() * BATCH_SIZE;
    double gcups_cuda = static_cast<double>(total_cells) / (cuda_batch.duration_us * 1000.0); 
    
    std::cout << "CUDA Performance: " << gcups_cuda << " GCUPS\n";
    std::cout << "----------------------------------------------------------\n";

    // ----- Correctness Check (Spot check) -----
    // 檢查最後一個結果是否一致
    if (!baseline_batch.results.empty() && !cuda_batch.results.empty()) {
        auto& cpu_res = baseline_batch.results.back();
        auto& gpu_res = cuda_batch.results.back();
        CHECK(gpu_res.score == cpu_res.score);
        CHECK(gpu_res.offset == cpu_res.offset);
        // CIGAR 可能因路徑選擇略有不同，可視情況 check
        // CHECK(gpu_res.cigar == cpu_res.cigar); 
    }
  }


}