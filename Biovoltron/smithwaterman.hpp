#pragma once

#include <Biovoltron/cigar.hpp>
#include <cassert>
#include <limits>
#include <vector>
#include <string>
#include <string_view> 
#include <utility> 
#include <algorithm>

namespace biovoltron {

/**
 * @ingroup align
 *
 * @brief Implements the Smith-Waterman local alignment algorithm with affine gap penalties.
 *
 * This structure supports local sequence alignment of two DNA strings (reference and alternative),
 * generating a CIGAR string representing the optimal alignment path.
 */
struct SmithWaterman {

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
   * @brief Quickly checks whether two strings differ by at most MAX_MISMATCHES.
   *
   * @param ref Reference string
   * @param alt Alternate string
   * @return true if they differ by <= MAX_MISMATCHES, false otherwise
   */
  static bool well_match(std::string_view ref, std::string_view alt) {
    int mismatch = 0;
    for (std::size_t i = 0; mismatch <= MAX_MISMATCHES && i < ref.size(); i++)
      if (alt[i] != ref[i])
        mismatch++;
    return mismatch <= MAX_MISMATCHES;
  }

 private:
  /**
   * @brief Populates scoring and traceback matrices using the Smith-Waterman algorithm.
   *
   * @param ref Reference sequence
   * @param alt Alternate sequence
   * @param score Matrix to hold alignment scores
   * @param trace Matrix to hold traceback directions
   * @param params Alignment scoring parameters
   */
  static void
  calculate_matrix(std::string_view ref, std::string_view alt,
                   std::vector<std::vector<int>>& score,
                   std::vector<std::vector<int>>& trace, Parameters params) {
    const std::size_t row_size = score.size();
    const std::size_t col_size = score[0].size();

    // Gap tracking vectors for affine penalties
    std::vector<int> gap_size_down(col_size + 1, 0);
    std::vector<int> best_gap_down(col_size + 1, std::numeric_limits<int>::min() / 2);
    std::vector<int> gap_size_right(row_size + 1, 0);
    std::vector<int> best_gap_right(row_size + 1, std::numeric_limits<int>::min() / 2);

    const int w_match    = params.w_match;
    const int w_mismatch = params.w_mismatch;
    const int w_open     = params.w_open;
    const int w_extend   = params.w_extend;

    for (std::size_t i = 1; i < row_size; i++) {
      for (std::size_t j = 1; j < col_size; j++) {
        // Diagonal (match or mismatch)
        const int step_diag =
          score[i - 1][j - 1]
          + ((ref[i - 1] == alt[j - 1]) ? w_match : w_mismatch);

        // Gap in ref (down)
        const int gap_open_down = score[i - 1][j] + w_open;
        best_gap_down[j] += w_extend;
        if (gap_open_down > best_gap_down[j]) {
          best_gap_down[j] = gap_open_down;
          gap_size_down[j] = 1;
        } else {
          gap_size_down[j]++;
        }
        const int step_down = best_gap_down[j];
        const int step_down_size = gap_size_down[j];

        // Gap in alt (right)
        const int gap_open_right = score[i][j - 1] + w_open;
        best_gap_right[i] += w_extend;
        if (gap_open_right > best_gap_right[i]) {
          best_gap_right[i] = gap_open_right;
          gap_size_right[i] = 1;
        } else {
          gap_size_right[i]++;
        }
        const int step_right = best_gap_right[i];
        const int step_right_size = gap_size_right[i];

        // Select the best move. The priority is diagonal > right > down.
        if (step_diag >= step_down && step_diag >= step_right) {
          score[i][j] = step_diag;
          trace[i][j] = 0;  // Diagonal move
        } else if (step_right >= step_down) {
          score[i][j] = step_right;
          trace[i][j] = -step_right_size;  // Insertion
        } else {
          score[i][j] = step_down;
          trace[i][j] = step_down_size;  // Deletion
        }
      }
    }
  }

  /**
   * @brief Performs traceback from the optimal cell to construct a CIGAR string.
   *
   * @param score Filled scoring matrix
   * @param trace Filled trace matrix
   * @return A pair containing the alignment offset and resulting CIGAR object
   */
  static SWResult
  calculate_cigar(std::vector<std::vector<int>>& score,
                  std::vector<std::vector<int>>& trace) {
    const int ref_size = static_cast<int>(score.size()) - 1;
    const int alt_size = static_cast<int>(score.front().size()) - 1;

    int max_score = std::numeric_limits<int>::min();
    int segment_len = 0;

    // look for the largest score on the rightmost column. we use >= combined
    // with the traversal direction to ensure that if two scores are equal, the
    // one closer to diagonal gets picked
    int pos_i = 0;
    for (int i = 1; i <= ref_size; i++) {
      const int cur_score = score[i][alt_size];
      if (cur_score >= max_score) {
        max_score = cur_score;
        pos_i = i;
      }
    }

    // now look for a larger score on the bottom-most row
    int pos_j = alt_size;
    auto diff = [](int x, int y) { return x > y ? x - y : y - x; };
    for (int j = 1; j <= alt_size; j++) {
      const int cur_score = score[ref_size][j];
      if (cur_score > max_score
          || (cur_score == max_score
              && diff(ref_size, j) < diff(pos_i, pos_j))) {
        max_score = cur_score;
        pos_i = ref_size;
        pos_j = j;
        // end of alternate is overhanging; we will just record it as 'M' segment
        segment_len = alt_size - j;  // Soft clip overhangs
      }
    }

    Cigar cigar{};
    if (segment_len > 0) {
      cigar.emplace_back(static_cast<unsigned>(segment_len), 'S');  // soft-clip at the end
      segment_len = 0;
    }

    char state = 'M';  // initial assumed state
    do {
      const int cur_trace = trace[pos_i][pos_j];

      char new_state;
      int step_size;
      if (cur_trace > 0) {
        new_state = 'D';
        step_size = cur_trace;
      } else if (cur_trace < 0) {
        new_state = 'I';
        step_size = -cur_trace;
      } else {
        new_state = 'M';
        step_size = 1;
      }

      // Update position based on move
      switch (new_state) {
        case 'M':
          pos_i--;
          pos_j--;
          break;
        case 'I':
          pos_j -= step_size;
          break;
        case 'D':
          pos_i -= step_size;
          break;
        default:
          break;
      }

      // Build CIGAR operation
      if (new_state == state) {
        segment_len += step_size;
      } else {
        cigar.emplace_back(static_cast<unsigned>(segment_len), state);
        segment_len = step_size;
        state = new_state;
      }
    } while (pos_i > 0 && pos_j > 0);

    cigar.emplace_back(static_cast<unsigned>(segment_len), state);
    const int align_offset = pos_i;

    if (pos_j > 0)
      cigar.emplace_back(static_cast<unsigned>(pos_j), 'S');  // soft-clip at the beginning

    cigar.reverse();
    return SWResult{align_offset, cigar, max_score};
  }

 public:
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
  static SWResult
  align(std::string_view ref, std::string_view alt,
        Parameters params = NEW_SW_PARAMETERS) {
    assert(!ref.empty() && !alt.empty());

    if (alt.size() == ref.size() && well_match(ref, alt)) {
      const std::string cigar_str = std::to_string(ref.size()) + std::string("M");
      return SWResult{0, Cigar(cigar_str), (int)params.w_match * (int)ref.size()};
    }

    std::vector<std::vector<int>> score(ref.size() + 1, std::vector<int>(alt.size() + 1, 0));
    std::vector<std::vector<int>> trace(ref.size() + 1, std::vector<int>(alt.size() + 1, 0));
    calculate_matrix(ref, alt, score, trace, params);
    return calculate_cigar(score, trace);
  }
};

}  // namespace biovoltron
