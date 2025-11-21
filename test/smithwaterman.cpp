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
  }
}