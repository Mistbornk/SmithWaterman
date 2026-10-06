#pragma once

#include <Biovoltron/cigar.hpp>
#include <cassert>
#include <limits>

#include <string_view>
#include <string>
#include <vector>

namespace biovoltron {

/**
 * @ingroup align
 *
 * @brief Implements the CPU SIMD Smith–Waterman local alignment algorithm
 *        with affine gap penalties (using xsimd under the hood).
 *
 * 介面盡量與 SmithWaterman / SmithWatermanCuda 保持一致：
 *  - 使用 SWResult 回傳 offset / CIGAR / score
 *  - 使用 Parameters 設定 scoring
 *  - 提供單一 pair 的 align()，以及可選的 batch_align()
 */
struct SmithWatermanSimd {

  // ---------------------------------------------------------------------------
  // Public types
  // ---------------------------------------------------------------------------

  struct SWResult {
    int   offset;  ///< Alignment start position on the reference
    Cigar cigar;   ///< CIGAR string representing the alignment
    int   score;   ///< Alignment score
  };

  /**
   * @brief Parameters for scoring alignment.
   *
   * Includes match reward, mismatch penalty, gap opening penalty,
   * and gap extension penalty.
   */
  struct Parameters {
    int w_match;    ///< Score for a match
    int w_mismatch; ///< Penalty for a mismatch
    int w_open;     ///< Penalty for opening a gap
    int w_extend;   ///< Penalty for extending a gap
  };

  // ---------------------------------------------------------------------------
  // Common scoring presets (kept consistent with scalar / CUDA versions)
  // ---------------------------------------------------------------------------

  /// Original BWA-style parameters: match = +3, mismatch = -1, gap = -1 - k
  static constexpr auto ORIGINAL_DEFAULT = Parameters{3, -1, -4, -3};

  /// Standard NGS alignment scoring scheme
  static constexpr auto STANDARD_NGS = Parameters{25, -50, -110, -6};

  /// Custom scoring used in Biovoltron's newer SW implementation
  static constexpr auto NEW_SW_PARAMETERS = Parameters{200, -150, -260, -11};

  /// Parameters used for alignment to the best haplotype
  static constexpr auto ALIGNMENT_TO_BEST_HAPLOTYPE_SW_PARAMETERS
    = Parameters{10, -15, -30, -5};

  /// Threshold for how many mismatches are allowed to be considered a "good match"
  static constexpr auto MAX_MISMATCHES = 2;

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * @brief Aligns two sequences using SIMD-accelerated Smith–Waterman.
   *
   * 若 ref / alt 長度相同且 mismatch 數量 <= MAX_MISMATCHES，
   * 會直接走 fast path 回傳一條簡單的 "lenM" CIGAR 以節省計算。
   *
   * @param ref    Reference sequence.
   * @param alt    Alternate (query) sequence.
   * @param params Alignment parameters (optional).
   *
   * @return SWResult with offset, CIGAR and score.
   */
  static auto
  align(std::string_view ref,
        std::string_view alt,
        Parameters params = NEW_SW_PARAMETERS) -> SWResult;

  /**
   * @brief Batch version of SIMD Smith–Waterman.
   *
   * 先求 correctness 為主：一開始可以在 .cpp 裡面
   * 直接 for-loop 呼叫單一版 align() 實作，
   * 之後若還有時間再做真正的 batch SIMD 最佳化。
   *
   * @param refs   List of reference sequences (size N).
   * @param alts   List of query sequences (size N).
   * @param params Alignment parameters (optional).
   *
   * @return Vector of SWResult, one per (ref, alt) pair.
   */
  static auto
  batch_align(const std::vector<std::string>& refs,
              const std::vector<std::string>& alts,
              Parameters params = NEW_SW_PARAMETERS)
      -> std::vector<SWResult>;

private:
  // -------------------------------------------------------------------------
  // Fast path helper：判斷是否可以直接回傳 "lenM"
  // -------------------------------------------------------------------------

  /**
   * @brief Check if ref and alt are "almost identical" so we can skip DP.
   *
   * 條件：
   *  - 兩邊長度相同
   *  - mismatch 次數 <= MAX_MISMATCHES
   */
  static auto
  well_match(std::string_view ref,
             std::string_view alt) -> bool;
};

}  // namespace biovoltron
