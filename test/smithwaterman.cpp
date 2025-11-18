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

  SECTION("Same reads") 
  {
    auto start = std::chrono::high_resolution_clock::now();  // Start timer
    // Test case: identical sequences
    const auto [offset, cigar] = SmithWaterman::align(ref, alt);

    REQUIRE(offset == 0);                // Should align from the beginning
    REQUIRE(std::string(cigar) == "162M"); // Expect perfect match over all 162 bases

    // End timer and print total execution time
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  
    std::cout << "SmithWaterman_same time: " << duration << " us" << std::endl;
  }

  SECTION("Substitutions") 
  {
    //auto start = std::chrono::high_resolution_clock::now();  // Start timer
    // Introduce mismatches between positions 70 and 79
    for (auto idx = 70; idx < 80; idx++) {
      ref[idx] = 'A';  // Change reference to A
      alt[idx] = 'T';  // Change alt to T
    }

    const auto [offset, cigar] = SmithWaterman::align(ref, alt);

    REQUIRE(offset == 0);                       // Still expected to align at position 0
    REQUIRE(std::string(cigar) == "69M10D1M10I82M"); // Complex alignment with substitutions turned into 10D/10I

    // End timer and print total execution time
    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  
    //std::cout << "SmithWaterman_substi time: " << duration << " us" << std::endl;
  }

  SECTION("Deletion") 
  {
    //auto start = std::chrono::high_resolution_clock::now();  // Start timer
    // Simulate a deletion in the alt (query) sequence at position 70
    alt.erase(alt.begin() + 70);

    const auto [offset, cigar] = SmithWaterman::align(ref, alt);

    REQUIRE(offset == 0);
    REQUIRE(std::string(cigar) == "70M1D91M");  // One deletion at position 70

    // End timer and print total execution time
    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  
    //std::cout << "SmithWaterman_del time: " << duration << " us" << std::endl;
  }

  SECTION("Insertion") 
  {
    //auto start = std::chrono::high_resolution_clock::now();  // Start timer
    // Simulate an insertion of 'T' in the alt (query) sequence at position 70
    alt.insert(alt.begin() + 70, 'T');

    const auto [offset, cigar] = SmithWaterman::align(ref, alt);

    REQUIRE(offset == 0);
    REQUIRE(std::string(cigar) == "70M1I92M");  // One insertion at position 70

    // End timer and print total execution time
    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  
    //std::cout << "SmithWaterman_ins time: " << duration << " us" << std::endl;
  }

  SECTION("Mix") 
  {
    //auto start = std::chrono::high_resolution_clock::now();  // Start timer
    // Introduce mismatches at multiple regions
    for (auto idx = 0; idx < ref.size(); idx++) {
      if ((idx > 10 && idx < 20) || (idx > 70 && idx < 80)
          || (idx > 120 && idx < 130)) {
        ref[idx] = 'A'; // Mutate reference
        alt[idx] = 'T'; // Mutate alt
      }
    }

    // Simulate one deletion and one insertion
    alt.erase(alt.begin() + 60);         // Delete at position 60
    alt.insert(alt.begin() + 90, 'T');   // Insert at position 90

    const auto [offset, cigar] = SmithWaterman::align(ref, alt);

    REQUIRE(offset == 0);
    REQUIRE(std::string(cigar) == "11M9D9I40M1D10M9D9I11M1I28M9D2M9I32M");
    // Explanation of CIGAR:
    // - 20S: soft clipped (low-scoring region at beginning)
    // - Mix of matches (M), deletions (D), and insertions (I) that reflect edits above

    // End timer and print total execution time
    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  
    //std::cout << "SmithWaterman_mix time: " << duration << " us" << std::endl;
  }
}

TEST_CASE("SmithWatermanCuda::align - Performs Smith-Waterman alignment", "[SmithWaterman]") 
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

  SECTION("Same reads") 
  {
    auto start = std::chrono::high_resolution_clock::now();  // Start timer
    // Test case: identical sequences
    const auto [offset, cigar] = SmithWatermanCuda::align(ref, alt);

    REQUIRE(offset == 0);                // Should align from the beginning
    REQUIRE(std::string(cigar) == std::to_string(ref.size()) + 'M'); // Expect perfect match over all bases

    // End timer and print total execution time
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  
    std::cout << "SmithWatermanCuda_same time: " << duration << " us" << std::endl;
  }
}