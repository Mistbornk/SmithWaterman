#pragma once


#include <Biovoltron/cigar.hpp>
#include <cassert>
#include <limits>
#include <stdexcept>

namespace biovoltron {

/**
 * @ingroup align
 *
 * @brief Implements the CUDA Smith-Waterman local alignment algorithm with affine gap penalties .
 * 
 */
struct SmithWatermanCuda {

  struct SWResult {
      int offset; 
      Cigar cigar;
      int score;
  };

  /**
   * @brief Parameters for scoring alignment.
   *
   * Includes match reward, mismatch penalty, gap opening penalty, and gap extension penalty.
   */
  struct Parameters {
    int w_match;   ///< Score for a match
    int w_mismatch;///< Penalty for a mismatch
    int w_open;    ///< Penalty for opening a gap
    int w_extend;  ///< Penalty for extending a gap
  };

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

  /**
   * @brief Aligns two sequences using Smith-Waterman and returns the alignment offset and CIGAR string.
   *
   * If the sequences are of equal length and differ by at most 2 mismatches,
   * the function will directly return a simple match CIGAR string (e.g., "150M")
   * without performing full Smith-Waterman alignment.
   *
   * @param ref Reference sequence.
   * @param alt Alternate (query) sequence.
   * @param params Alignment parameters (optional).
   * @return Pair of alignment offset and CIGAR string.
   */
  static auto
  align(std::string_view ref, std::string_view alt,
        Parameters params = NEW_SW_PARAMETERS)
    -> SWResult; 
  
};

} // namespace biovoltron