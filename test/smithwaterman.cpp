#include <Biovoltron/smithwaterman.hpp>
#include <Biovoltron/smithwaterman_cuda.hpp>
#include "catch.hpp"
#include <chrono>
#include <iostream>
#include <fstream>
using namespace biovoltron;

namespace {
std::string read_fasta_sequence(const std::string& path) {
  std::ifstream f(path);
  REQUIRE(f.good());
  std::string seq, line;
  while (std::getline(f, line)) {
    if (!line.empty() && line[0] != '>')
      seq += line;
  }
  return seq;
}
} // namespace

TEST_CASE("SmithWaterman::align / SmithWatermanCuda::align 基本行為", "[SmithWaterman]") {
  const std::string base = "ACGTACGTACGTACGTACGTACGTACGTACGT"; // 32 bp

  SECTION("同長度且差異 <=2 時走 quick path，CIGAR 應為全 M") {
    std::string ref = base;
    std::string alt = base;
    // 引入單一 mismatch 仍應走 quick path
    alt[5] = (alt[5] == 'A') ? 'C' : 'A';

    const auto cpu_start = std::chrono::high_resolution_clock::now();
    const auto [offset_cpu, cigar_cpu] = SmithWaterman::align(ref, alt);
    const auto cpu_end = std::chrono::high_resolution_clock::now();

    const auto gpu_start = std::chrono::high_resolution_clock::now();
    const auto [offset_gpu, cigar_gpu] = SmithWatermanCuda::align(ref, alt);
    const auto gpu_end = std::chrono::high_resolution_clock::now();

    const std::string expected = std::to_string(ref.size()) + 'M';
    REQUIRE(offset_cpu == 0);
    REQUIRE(offset_gpu == 0);
    REQUIRE(std::string(cigar_cpu) == expected);
    REQUIRE(std::string(cigar_gpu) == expected);

    const auto cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    const auto gpu_us = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count();
    const double speedup = gpu_us > 0 ? static_cast<double>(cpu_us) / static_cast<double>(gpu_us) : 0.0;
    std::cout << "[Quick path] CPU: " << cpu_us << " us, CUDA: " << gpu_us
              << " us, speedup: " << speedup << "x" << std::endl;
  }

  SECTION("插入案例，檢查 CPU/GPU 結果一致且包含 I") {
    std::string ref = base;
    std::string alt = base;
    alt.insert(10, "T"); // 在第 10 個位置插入

    const auto cpu_start = std::chrono::high_resolution_clock::now();
    const auto [offset_cpu, cigar_cpu] = SmithWaterman::align(ref, alt);
    const auto cpu_end = std::chrono::high_resolution_clock::now();

    const auto gpu_start = std::chrono::high_resolution_clock::now();
    const auto [offset_gpu, cigar_gpu] = SmithWatermanCuda::align(ref, alt);
    const auto gpu_end = std::chrono::high_resolution_clock::now();

    REQUIRE(offset_cpu == offset_gpu);
    REQUIRE(std::string(cigar_cpu) == std::string(cigar_gpu));
    REQUIRE(std::string(cigar_cpu).find('I') != std::string::npos);

    const auto cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    const auto gpu_us = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count();
    const double speedup = gpu_us > 0 ? static_cast<double>(cpu_us) / static_cast<double>(gpu_us) : 0.0;
    std::cout << "[Insertion] CPU: " << cpu_us << " us, CUDA: " << gpu_us
              << " us, speedup: " << speedup << "x" << std::endl;
  }

  SECTION("缺失案例，檢查 CPU/GPU 結果一致且包含 D") {
    std::string ref = base;
    std::string alt = base;
    alt.erase(10, 1); // 刪除一個字元

    const auto cpu_start = std::chrono::high_resolution_clock::now();
    const auto [offset_cpu, cigar_cpu] = SmithWaterman::align(ref, alt);
    const auto cpu_end = std::chrono::high_resolution_clock::now();

    const auto gpu_start = std::chrono::high_resolution_clock::now();
    const auto [offset_gpu, cigar_gpu] = SmithWatermanCuda::align(ref, alt);
    const auto gpu_end = std::chrono::high_resolution_clock::now();

    REQUIRE(offset_cpu == offset_gpu);
    REQUIRE(std::string(cigar_cpu) == std::string(cigar_gpu));
    REQUIRE(std::string(cigar_cpu).find('D') != std::string::npos);

    const auto cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    const auto gpu_us = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count();
    const double speedup = gpu_us > 0 ? static_cast<double>(cpu_us) / static_cast<double>(gpu_us) : 0.0;
    std::cout << "[Deletion] CPU: " << cpu_us << " us, CUDA: " << gpu_us
              << " us, speedup: " << speedup << "x" << std::endl;
  }
}

TEST_CASE("使用檔案序列時 CPU/GPU 結果需一致", "[SmithWaterman]") {
  const std::string data_path = DATA_PATH;
  const std::string ref = read_fasta_sequence(data_path + "/ref.fasta");
  const std::string alt = read_fasta_sequence(data_path + "/alt.fasta");
  REQUIRE(!ref.empty());
  REQUIRE(!alt.empty());

  const auto cpu_start = std::chrono::high_resolution_clock::now();
  const auto [offset_cpu, cigar_cpu] = SmithWaterman::align(ref, alt);
  const auto cpu_end = std::chrono::high_resolution_clock::now();

  const auto gpu_start = std::chrono::high_resolution_clock::now();
  const auto [offset_gpu, cigar_gpu] = SmithWatermanCuda::align(ref, alt);
  const auto gpu_end = std::chrono::high_resolution_clock::now();

  REQUIRE(offset_cpu == offset_gpu);
  REQUIRE(std::string(cigar_cpu) == std::string(cigar_gpu));

  const auto cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
  const auto gpu_us = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count();
  const double speedup = gpu_us > 0 ? static_cast<double>(cpu_us) / static_cast<double>(gpu_us) : 0.0;
  std::cout << "[File input] CPU: " << cpu_us << " us, CUDA: " << gpu_us
            << " us, speedup: " << speedup << "x" << std::endl;
}
