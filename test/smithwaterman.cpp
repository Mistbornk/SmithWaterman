#include <Biovoltron/smithwaterman.hpp>
#include <Biovoltron/smithwaterman_cuda.hpp>
#include "catch.hpp"
#include <chrono>
#include <iostream>
#include <fstream>

using namespace biovoltron;

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
}