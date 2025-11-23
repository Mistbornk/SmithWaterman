#include <Biovoltron/smithwaterman_simd.hpp>
#include <Biovoltron/cigar.hpp>

#include <xsimd/xsimd.hpp>

#include <cassert>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include <thread>

namespace biovoltron {

// =========================== internal helpers ===============================

namespace {

using batch_i32 = xsimd::batch<int32_t>;

/**
 * SIMD 加速版本的 DP matrix 建立：
 *  - 先對每一行 ref[i-1]，用 SIMD 一次算出 alt 上所有位置的
 *    match / mismatch 分數 (diag_scores[j])。
 *  - 再用與 scalar 版幾乎相同的 affine gap DP 更新 score / trace。
 */
void calculate_matrix_simd(std::string_view ref,
                           std::string_view alt,
                           std::vector<std::vector<int>>& score,
                           std::vector<std::vector<int>>& trace,
                           SmithWatermanSimd::Parameters params) {
  const std::size_t row_size = score.size();        // = ref.size() + 1
  const std::size_t col_size = score.front().size(); // = alt.size() + 1
  const std::size_t alt_len  = alt.size();

  // Gap tracking vectors for affine penalties
  std::vector<int> gap_size_down(col_size + 1, 0);
  std::vector<int> best_gap_down(col_size + 1,
                                 std::numeric_limits<int>::min() / 2);
  std::vector<int> gap_size_right(row_size + 1, 0);
  std::vector<int> best_gap_right(row_size + 1,
                                  std::numeric_limits<int>::min() / 2);

  const int w_match    = params.w_match;
  const int w_mismatch = params.w_mismatch;
  const int w_open     = params.w_open;
  const int w_extend   = params.w_extend;

  // 把 alt 的字元先轉成 32-bit 整數，方便用 xsimd 做等號比對
  std::vector<int32_t> alt_codes(alt_len);
  for (std::size_t j = 0; j < alt_len; ++j) {
    alt_codes[j] = static_cast<unsigned char>(alt[j]);
  }

  // diag_scores[j] = ref[i-1] 和 alt[j-1] 的 match/mismatch 分數
  // 注意：diag_scores[0] 不使用，從 1..alt_len 對應到 j。
  std::vector<int> diag_scores(col_size, 0);

  const std::size_t vec_size = batch_i32::size;

  for (std::size_t i = 1; i < row_size; ++i) {
    // --------------------  用 SIMD 算一整列的 s(i, j)  --------------------
    const int32_t ref_code = static_cast<unsigned char>(ref[i - 1]);
    const batch_i32 v_ref(ref_code);
    const batch_i32 v_match(w_match);
    const batch_i32 v_mismatch(w_mismatch);

    std::size_t j = 0;
    for (; j + vec_size <= alt_len; j += vec_size) {
      // 載入一段 alt 的編碼
      batch_i32 v_alt = xsimd::load_unaligned(&alt_codes[j]);
      auto mask       = (v_alt == v_ref);  // batch_bool<int32_t>

      // 相等的地方給 match 分數，不相等給 mismatch
      batch_i32 v_scores = xsimd::select(mask, v_match, v_mismatch);

      // 存到 diag_scores，索引 +1 對齊 j (1..alt_len)
      xsimd::store_unaligned(&diag_scores[j + 1], v_scores);
    }
    // 處理尾巴不足一個 batch 的部分 (scalar)
    for (; j < alt_len; ++j) {
      diag_scores[j + 1] =
          (alt[j] == ref[i - 1]) ? w_match : w_mismatch;
    }

    // --------------------  scalar DP (跟原本 calculate_matrix 幾乎一樣) ----
    for (std::size_t col = 1; col < col_size; ++col) {
      // diagonal: match / mismatch
      const int step_diag = score[i - 1][col - 1] + diag_scores[col];

      // Gap in ref (down)
      const int gap_open_down = score[i - 1][col] + w_open;
      best_gap_down[col] += w_extend;
      if (gap_open_down > best_gap_down[col]) {
        best_gap_down[col] = gap_open_down;
        gap_size_down[col] = 1;
      } else {
        gap_size_down[col]++;
      }
      const int step_down      = best_gap_down[col];
      const int step_down_size = gap_size_down[col];

      // Gap in alt (right)
      const int gap_open_right = score[i][col - 1] + w_open;
      best_gap_right[i] += w_extend;
      if (gap_open_right > best_gap_right[i]) {
        best_gap_right[i] = gap_open_right;
        gap_size_right[i] = 1;
      } else {
        gap_size_right[i]++;
      }
      const int step_right      = best_gap_right[i];
      const int step_right_size = gap_size_right[i];

      // Select the best move. Priority: diagonal > right > down.
      if (step_diag >= step_down && step_diag >= step_right) {
        score[i][col] = step_diag;
        trace[i][col] = 0;                   // diagonal (M)
      } else if (step_right >= step_down) {
        score[i][col] = step_right;
        trace[i][col] = -step_right_size;    // insertion (I)
      } else {
        score[i][col] = step_down;
        trace[i][col] = step_down_size;      // deletion (D)
      }

      // Local alignment：不允許分數小於 0
      if (score[i][col] < 0) {
        score[i][col] = 0;
        trace[i][col] = 0;
      }
    }
  }
}

/**
 * Traceback：根據 score / trace 回推，產生 CIGAR 和 offset。
 * 這一段基本上就是把 smithwaterman.hpp 裡的 calculate_cigar 拿過來用，
 * 只是回傳型別改成 SmithWatermanSimd::SWResult。
 */
auto calculate_cigar_simd(std::vector<std::vector<int>>& score,
                          std::vector<std::vector<int>>& trace)
    -> SmithWatermanSimd::SWResult {
  const int ref_size = static_cast<int>(score.size()) - 1;
  const int alt_size = static_cast<int>(score.front().size()) - 1;

  int max_score   = std::numeric_limits<int>::min();
  int segment_len = 0;

  // 先在最右邊一整欄找最大值（= ref 的內部某位置結束）
  int pos_i = 0;
  for (int i = 1; i <= ref_size; ++i) {
    const int cur_score = score[i][alt_size];
    if (cur_score >= max_score) {
      max_score = cur_score;
      pos_i     = i;
    }
  }

  // 再看最底下一整列是不是有更大的分數，或相同但更靠近對角線
  int pos_j = alt_size;
  auto diff = [](int x, int y) { return x > y ? x - y : y - x; };
  for (int j = 1; j <= alt_size; ++j) {
    const int cur_score = score[ref_size][j];
    if (cur_score > max_score ||
        (cur_score == max_score &&
         diff(ref_size, j) < diff(pos_i, pos_j))) {
      max_score = cur_score;
      pos_i     = ref_size;
      pos_j     = j;
      // alt 在尾端多出來的部分當作 soft-clip
      segment_len = alt_size - j;
    }
  }

  Cigar cigar{};
  if (segment_len > 0) {
    cigar.emplace_back(static_cast<unsigned>(segment_len), 'S');  // tail soft-clip
    segment_len = 0;
  }

  char state = 'M';  // 初始假設在 M 狀態
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

    // 根據狀態更新位置
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

    // 組 CIGAR：連續同一個 op 就累加長度
    if (new_state == state) {
      segment_len += step_size;
    } else {
      cigar.emplace_back(static_cast<unsigned>(segment_len), state);
      segment_len = step_size;
      state       = new_state;
    }
  } while (pos_i > 0 && pos_j > 0);

  cigar.emplace_back(static_cast<unsigned>(segment_len), state);
  const int align_offset = pos_i;

  if (pos_j > 0) {
    cigar.emplace_back(static_cast<unsigned>(pos_j), 'S');  // 頭部 soft-clip
  }

  cigar.reverse();
  return SmithWatermanSimd::SWResult{align_offset, cigar, max_score};
}

}  // namespace

// =========================== SmithWatermanSimd ===============================

// private helper: well_match（和 scalar 版邏輯一致）
auto SmithWatermanSimd::well_match(std::string_view ref,
                                   std::string_view alt) -> bool {
  if (ref.size() != alt.size()) {
    return false;
  }

  int mismatch = 0;
  for (std::size_t i = 0;
       i < ref.size() && mismatch <= MAX_MISMATCHES;
       ++i) {
    if (ref[i] != alt[i]) {
      ++mismatch;
    }
  }
  return mismatch <= MAX_MISMATCHES;
}

// 單一 pair 的 SIMD SW
auto SmithWatermanSimd::align(std::string_view ref,
                              std::string_view alt,
                              Parameters params) -> SWResult {
  assert(!ref.empty() && !alt.empty());

  // Fast path：長度一樣且 mismatch 很少，直接視為全 M
  if (ref.size() == alt.size() && well_match(ref, alt)) {
    const int len = static_cast<int>(ref.size());
    const std::string cigar_str = std::to_string(len) + 'M';

    return SWResult{
        /*offset=*/0,
        /*cigar=*/Cigar(cigar_str),
        /*score=*/params.w_match * len};
  }

  // 建立 DP 表
  std::vector<std::vector<int>> score(ref.size() + 1,
                                      std::vector<int>(alt.size() + 1, 0));
  std::vector<std::vector<int>> trace(ref.size() + 1,
                                      std::vector<int>(alt.size() + 1, 0));

  // 用 SIMD 建 matrix
  calculate_matrix_simd(ref, alt, score, trace, params);

  // Traceback 產生 CIGAR + offset + score
  return calculate_cigar_simd(score, trace);
}

auto SmithWatermanSimd::batch_align(const std::vector<std::string>& refs,
                                    const std::vector<std::string>& alts,
                                    Parameters params)
    -> std::vector<SWResult> {
  const std::size_t n = std::min(refs.size(), alts.size());
  std::vector<SWResult> results(n);

  if (n == 0) {
    return results;
  }

  // 決定要開幾個執行緒
  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 1;
  }
  if (num_threads > n) {
    num_threads = static_cast<unsigned int>(n);
  }

  // 每個 thread 負責一個連續區間 [begin, end)
  auto worker = [&](unsigned int tid) {
    const std::size_t chunk_size = (n + num_threads - 1) / num_threads;
    const std::size_t begin      = tid * chunk_size;
    const std::size_t end        = std::min(n, begin + chunk_size);

    for (std::size_t i = begin; i < end; ++i) {
      results[i] = align(refs[i], alts[i], params);
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (unsigned int t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker, t);
  }
  for (auto& th : threads) {
    th.join();
  }

  return results;
}

}  // namespace biovoltron
