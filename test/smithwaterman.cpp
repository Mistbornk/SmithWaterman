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
    std::string truth_s;
    int truth_score;

    {
      auto start = std::chrono::high_resolution_clock::now(); 
      auto [offset, cigar, score] = biovoltron::SmithWaterman::align(ref, alt, biovoltron::SmithWaterman::ORIGINAL_DEFAULT);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      std::cout << "SmithWaterman time: " << duration << " us" << std::endl;

      std::string s(cigar);
      std::cout << "Offset = " << offset << "\n";
      std::cout << "CIGAR = " << s << "\n";
      std::cout << "SCORE = " << score << "\n";

      truth_offset = offset;
      truth_s = s;
      truth_score = score;
    }

    {
      auto start = std::chrono::high_resolution_clock::now(); 
      auto [offset, cigar, score] = biovoltron::SmithWatermanCuda::align(ref, alt, biovoltron::SmithWatermanCuda::ORIGINAL_DEFAULT);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      std::cout << "SmithWaterman Cuda time: " << duration << " us" << std::endl;

      //std::string s(cigar);
      //REQUIRE(offset == truth_offset);                // Should align from the beginning
      //REQUIRE(std::string(cigar) == truth_s); // Expect perfect match over all bases
      //REQUIRE(score == truth_score);
    }

  }
}