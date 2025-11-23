#include <Biovoltron/smithwaterman_simd.hpp>
#include <Biovoltron/cigar.hpp>

#include <xsimd/xsimd.hpp>

#include <array>
#include <vector>
#include <string>
#include <string_view>
#include <cstdint>
#include <algorithm>
#include <thread>

namespace biovoltron {

namespace {

using value_type = std::int16_t;
constexpr std::uint8_t kBases = 4;

// 4x4 substitution matrix: [ref_base][alt_base]
using sub_t = std::array<std::array<value_type, kBases>, kBases>;

// aligned vector，給 xsimd 用
using vector_aligned =
    std::vector<value_type, xsimd::default_allocator<value_type>>;

// 建 substitution matrix：對角 = match，其他 = -mismatch_abs
sub_t make_substitution_matrix(value_type match, value_type mismatch_abs) {
  sub_t m{};
  for (std::uint8_t i = 0; i < kBases; ++i) {
    for (std::uint8_t j = 0; j < kBases; ++j) {
      m[i][j] = (i == j) ? match : static_cast<value_type>(-mismatch_abs);
    }
  }
  return m;
}

// A/C/G/T -> 0..3
inline std::uint8_t encode_base(char c) {
  switch (c) {
    case 'A': return 0;
    case 'C': return 1;
    case 'G': return 2;
    case 'T': return 3;
    default:  return 0;  // fallback，非 ACGT 都當 A 看
  }
}

// 從兩條 alignment 字串壓成 CIGAR（M / I / D）
inline Cigar build_cigar_from_alignment(const std::string& align_ref,
                                        const std::string& align_alt) {
  Cigar cigar;
  if (align_ref.empty()) {
    return cigar;
  }

  char cur_op = 0;
  unsigned cur_len = 0;

  auto flush = [&]() {
    if (cur_len == 0) return;
    cigar.emplace_back(cur_len, cur_op);
    cur_len = 0;
  };

  const std::size_t n = align_ref.size();
  for (std::size_t i = 0; i < n; ++i) {
    const char a = align_ref[i];
    const char b = align_alt[i];
    char op = 0;

    if (a != '-' && b != '-') {
      op = 'M';  // match / mismatch 都算 M
    } else if (a != '-' && b == '-') {
      op = 'D';
    } else if (a == '-' && b != '-') {
      op = 'I';
    } else {
      // 理論上不會兩個都是 '-'
      continue;
    }

    if (op == cur_op) {
      ++cur_len;
    } else {
      flush();
      cur_op = op;
      cur_len = 1;
    }
  }
  flush();
  return cigar;
}

}  // namespace


auto SmithWatermanSimd::align(std::string_view ref,
                              std::string_view alt,
                              Parameters params) -> SWResult {
  // 空字串處理
  if (ref.empty() || alt.empty()) {
    return SWResult{0, Cigar{}, 0};
  }

  // ===== 參數轉換：Biovoltron =====
  //
  // Biovoltron:
  //   w_match > 0
  //   w_mismatch < 0
  //   w_open < 0
  //   w_extend < 0
  //
  // 學長的 striped 版：
  //   match > 0
  //   mismatch_abs > 0（實際套用時用 -mismatch_abs）
  //   gap_open > 0, gap_extend > 0（H/E/F 用減法）
  //
  const value_type match        = static_cast<value_type>(params.w_match);
  const value_type mismatch_abs = static_cast<value_type>(-params.w_mismatch);
  const value_type gap_open     = static_cast<value_type>(-params.w_open);
  const value_type gap_extend   = static_cast<value_type>(-params.w_extend);

  using batch_t = xsimd::batch<value_type>;
  constexpr std::size_t batch_size = batch_t::size;

  const std::size_t len_ref = ref.size();
  const std::size_t len_alt = alt.size();

  // striped 參數
  const std::size_t seg_len     = (len_alt + batch_size) / batch_size;
  const std::size_t profile_len = seg_len * batch_size;

  // 建 substitution matrix & profile :contentReference[oaicite:1]{index=1}
  const auto sub_mat = make_substitution_matrix(match, mismatch_abs);

  std::array<vector_aligned, kBases> profile;
  for (std::uint8_t base = 0; base < kBases; ++base) {
    profile[base].assign(profile_len, 0);
    for (std::size_t j = 0; j < seg_len; ++j) {
      for (std::size_t k = (j == 0) ? 1u : 0u; k < batch_size; ++k) {
        const std::size_t idx = k * seg_len + j - 1;
        if (idx < len_alt) {
          const std::uint8_t code = encode_base(alt[idx]);
          profile[base][j * batch_size + k] = sub_mat[base][code];
        }
      }
    }
  }

  // H / E / F 三個 DP 矩陣（每列一條 profile_len 長度的向量）
  std::vector<vector_aligned> H(len_ref + 1, vector_aligned(profile_len, 0));
  std::vector<vector_aligned> E(len_ref + 1, vector_aligned(profile_len, 0));
  std::vector<vector_aligned> F(len_ref + 1, vector_aligned(profile_len, 0));

  const batch_t GAP_O(gap_open);
  const batch_t GAP_E(gap_extend);
  const batch_t ZERO(static_cast<value_type>(0));

  std::size_t max_i = 0, max_j = 0, max_k = 0;
  value_type max_score = 0;

  // ================== 主 DP 迴圈（striped + SIMD） ==================
  for (std::size_t i = 1; i <= len_ref; ++i) {
    // vH[0], vH[1] 交錯當 buffer
    std::array<batch_t, 2> vH{
        xsimd::slide_left<sizeof(value_type)>(
            xsimd::load_aligned(&H[i - 1][profile_len - batch_size])),
        ZERO};

    batch_t vF = ZERO;
    const std::uint8_t base = encode_base(ref[i - 1]);

    for (std::size_t j = 0; j < seg_len; ++j) {
      const bool even = ((j & 1u) == 0u);
      auto& vH_prev = vH[even];     // i-1, j-1 / lazy buffer
      auto& vH_curr = vH[!even];    // i,   j

      // vertical gap F：從上方延伸
      vF = xsimd::max(vF - GAP_E, vH_prev - GAP_O);

      // 讀 H[i-1][j]，更新 E
      vH_prev = xsimd::load_aligned(&H[i - 1][j * batch_size]);

      batch_t vE = xsimd::max(
          xsimd::load_aligned(&E[i - 1][j * batch_size]) - GAP_E,
          vH_prev - GAP_O);

      // match/mismatch + 取 max(H, E, F, 0)
      vH_curr = xsimd::max(
          xsimd::max(
              vH_curr + xsimd::load_aligned(&profile[base][j * batch_size]),
              ZERO),
          xsimd::max(vE, vF));

      // 更新 max_score
      const value_type new_score = xsimd::reduce_max(vH_curr);
      if (new_score > max_score) {
        max_score = new_score;
        max_i = i;
        max_j = j;
      }

      // 寫回 H / E / F
      xsimd::store_aligned(&H[i][j * batch_size], vH_curr);
      xsimd::store_aligned(&E[i][j * batch_size], vE);
      xsimd::store_aligned(&F[i][j * batch_size], vF);
    }

    // ========== Lazy F loop（修正 F 沿著 row 的 propagate）==========
    const std::size_t last_idx = (~seg_len) & 1u;  // segLen 偶/奇 決定用哪個 buffer
    vF = xsimd::slide_left<sizeof(value_type)>(
        xsimd::max(vF - GAP_E, vH[last_idx] - GAP_O));

    std::size_t j = 0;
    std::size_t pass = 0;
    while (true) {
      batch_t vh = xsimd::load_aligned(&H[i][j * batch_size]);
      if (xsimd::count(vF > (vh - GAP_O)) == 0) {
        break;
      }
      xsimd::store_aligned(&H[i][j * batch_size], xsimd::max(vh, vF));
      xsimd::store_aligned(
          &F[i][j * batch_size],
          xsimd::max(xsimd::load_aligned(&F[i][j * batch_size]), vF));

      if (j + 1 == seg_len) {
        vF = xsimd::slide_left<sizeof(value_type)>(vF - GAP_E);
        j = 0;
        if (++pass > 2) break;
      } else {
        vF -= GAP_E;
        ++j;
      }
    }
  }

  // ================== Traceback：從 max_score 開始 ==================
  if (max_score <= 0) {
    // 全部都 <=0，代表沒有局部對齊
    return SWResult{0, Cigar{}, 0};
  }

  // 找到在該 stripe 中真正等於 max_score 的 lane k
  for (std::size_t k = 0; k < batch_size; ++k) {
    if (H[max_i][max_j * batch_size + k] == max_score) {
      max_k = k;
      break;
    }
  }

  std::size_t i = max_i;
  std::size_t j = max_j;
  std::size_t k = max_k;

  auto cell_idx = [&](std::size_t ii, std::size_t jj, std::size_t kk) {
    (void)ii;  // row 已經在外層
    return jj * batch_size + kk;
  };

  // j,k 以 striped 座標往「左上」走
  auto move_left = [&]() {
    if (j == 0) {
      // 往前一個 stripe，同時 k--（從右邊 slide）
      if (k > 0) {
        --k;
      }
      j = seg_len - 1;
    } else {
      --j;
    }
  };

  std::string align_ref;
  std::string align_alt;

  while (i > 0) {
    const std::size_t idx = cell_idx(i, j, k);
    const value_type h = H[i][idx];
    if (h <= 0) break;

    const std::size_t q_pos = k * seg_len + j - 1;  // alt 的 index

    const value_type e = E[i][idx];
    const value_type f = F[i][idx];

    // 優先順序：E（水平 gap in alt）> F（垂直 gap in ref）> diag
    if (h == e) {
      // gap in alt：ref 有字元，alt 缺字
      align_ref += ref[i - 1];
      align_alt += '-';

      // 連續往上延伸 E
      while (i > 1) {
        const std::size_t idx_up = cell_idx(i - 1, j, k);
        if (E[i][idx] != E[i - 1][idx_up] - gap_extend) break;
        --i;
        align_ref += ref[i - 1];
        align_alt += '-';
      }
      --i;
    } else if (h == f) {
      // gap in ref：ref 缺，alt 有
      align_ref += '-';
      align_alt += alt[q_pos];

      // 往左延伸 F（striped 左移）
      auto prev_idx = (j == 0) ? (profile_len - batch_size + k - 1)
                               : ((j - 1) * batch_size + k);
      while (F[i][idx] == F[i][prev_idx] - gap_extend) {
        move_left();
        prev_idx = (j == 0) ? (profile_len - batch_size + k - 1)
                            : ((j - 1) * batch_size + k);
        align_ref += '-';
        // q_pos 這個簡化版本，實務上可以再更精確追蹤
        align_alt += alt[q_pos];
      }
      move_left();
    } else {
      // diag：match or mismatch
      align_ref += ref[i - 1];
      align_alt += alt[q_pos];
      --i;
      move_left();
    }
  }

  // 反轉成正向
  std::reverse(align_ref.begin(), align_ref.end());
  std::reverse(align_alt.begin(), align_alt.end());

  // 用 alignment 壓成 CIGAR
  Cigar cigar = build_cigar_from_alignment(align_ref, align_alt);

  // offset：對齊在 ref 上的起始位置 = 剛剛 traceback 結束時的 i
  const int offset = static_cast<int>(i);
  const int score  = static_cast<int>(max_score);

  return SWResult{offset, std::move(cigar), score};
}
auto
SmithWatermanSimd::batch_align(const std::vector<std::string>& refs,
                               const std::vector<std::string>& alts,
                               Parameters params) -> std::vector<SWResult> {
  const std::size_t n = std::min(refs.size(), alts.size());
  std::vector<SWResult> results(n);

  if (n == 0) {
    return results;
  }

  // 決定要開幾個 thread
  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 1;
  }
  if (num_threads > n) {
    num_threads = static_cast<unsigned int>(n);
  }

  // 每個 thread 負責一段 [begin, end)
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
