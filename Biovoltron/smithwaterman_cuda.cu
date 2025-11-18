#include <Biovoltron/smithwaterman_cuda.hpp>
#include <cuda_runtime.h>
#include <stdexcept>

#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#include <climits>

namespace biovoltron {

// ----------------- CUDA error helper -----------------
static inline void check_cuda(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
  }
}

// --- CUDA kernel ---
// Batched 版：一個 block 負責一個 alignment，
// 目前仍由 thread 0 在 block 裡跑 CPU 版 double-loop（後續可再細化平行化）
__global__ void smith_waterman_kernel(const char* all_refs,
                                      const char* all_alts,
                                      int ref_len, int alt_len,
                                      int w_match, int w_mismatch,
                                      int w_open, int w_extend,
                                      int* all_scores, int* all_traces,
                                      int* all_gap_size_down,
                                      int* all_best_gap_down,
                                      int* all_gap_size_right,
                                      int* all_best_gap_right,
                                      int n_alignments)
{
  const int aln = blockIdx.x;
  if (aln >= n_alignments)
    return;

  // 暫時只讓 thread 0 做事；之後可以改成 block 內多 thread 分工
  if (threadIdx.x != 0)
    return;

  const int rows = ref_len + 1;
  const int cols = alt_len + 1;

  // 針對這個 alignment 的 slice
  const char* ref = all_refs + static_cast<std::size_t>(aln) * ref_len;
  const char* alt = all_alts + static_cast<std::size_t>(aln) * alt_len;

  int* score = all_scores + static_cast<std::size_t>(aln) * rows * cols;
  int* trace = all_traces + static_cast<std::size_t>(aln) * rows * cols;

  int* gap_size_down  = all_gap_size_down  + static_cast<std::size_t>(aln) * (cols + 1);
  int* best_gap_down  = all_best_gap_down  + static_cast<std::size_t>(aln) * (cols + 1);
  int* gap_size_right = all_gap_size_right + static_cast<std::size_t>(aln) * (rows + 1);
  int* best_gap_right = all_best_gap_right + static_cast<std::size_t>(aln) * (rows + 1);

  auto idx = [cols](int i, int j) {
    return i * cols + j;
  };

  // 初始化 score / trace
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      score[idx(i, j)] = 0;
      trace[idx(i, j)] = 0;
    }
  }

  // gap tracking arrays（和 CPU 版相同邏輯），由 host 預先配置，只在這裡初始化
  const int NEG_INF = INT_MIN / 2;

  for (int j = 0; j <= cols; ++j) {
    gap_size_down[j] = 0;
    best_gap_down[j] = NEG_INF;
  }
  for (int i = 0; i <= rows; ++i) {
    gap_size_right[i] = 0;
    best_gap_right[i] = NEG_INF;
  }

  for (int i = 1; i < rows; ++i) {
    for (int j = 1; j < cols; ++j) {
      const int diag_score = score[idx(i - 1, j - 1)];
      const int step_diag =
        diag_score + ((ref[i - 1] == alt[j - 1]) ? w_match : w_mismatch);

      // Gap in ref (down)
      const int gap_open_down = score[idx(i - 1, j)] + w_open;
      best_gap_down[j] += w_extend;
      if (gap_open_down > best_gap_down[j]) {
        best_gap_down[j] = gap_open_down;
        gap_size_down[j] = 1;
      } else {
        gap_size_down[j] += 1;
      }
      const int step_down      = best_gap_down[j];
      const int step_down_size = gap_size_down[j];

      // Gap in alt (right)
      const int gap_open_right = score[idx(i, j - 1)] + w_open;
      best_gap_right[i] += w_extend;
      if (gap_open_right > best_gap_right[i]) {
        best_gap_right[i] = gap_open_right;
        gap_size_right[i] = 1;
      } else {
        gap_size_right[i] += 1;
      }
      const int step_right      = best_gap_right[i];
      const int step_right_size = gap_size_right[i];

      // pick best: diag > right >= down
      if (step_diag >= step_down && step_diag >= step_right) {
        score[idx(i, j)] = step_diag;
        trace[idx(i, j)] = 0;                 // Diagonal
      } else if (step_right >= step_down) {
        score[idx(i, j)] = step_right;
        trace[idx(i, j)] = -step_right_size;  // Insertion (right)
      } else {
        score[idx(i, j)] = step_down;
        trace[idx(i, j)] = step_down_size;    // Deletion (down)
      }
    }
  }
}

// ----------------- Host-side traceback (flattened) -----------------

static std::pair<int, Cigar>
traceback_and_build_cigar(const int* score,
                          const int* trace,
                          int ref_len, int alt_len)
{
  const int rows = ref_len + 1;
  const int cols = alt_len + 1;

  auto idx = [cols](int i, int j) {
    return i * cols + j;
  };

  const int ref_size = ref_len;
  const int alt_size = alt_len;

  int max_score = std::numeric_limits<int>::min();
  int segment_len = 0;

  // 從最後一欄找最大 score
  int pos_i = 0;
  int pos_j = alt_size;
  for (int i = 1; i <= ref_size; ++i) {
    const int cur_score = score[idx(i, alt_size)];
    if (cur_score >= max_score) {
      max_score = cur_score;
      pos_i = i;
      pos_j = alt_size;
    }
  }

  // 再從最後一列找更佳分數（或離對角線較近）
  auto diff = [](int x, int y) { return x > y ? x - y : y - x; };
  for (int j = 1; j <= alt_size; ++j) {
    const int cur_score = score[idx(ref_size, j)];
    if (cur_score > max_score
        || (cur_score == max_score
            && diff(ref_size, j) < diff(pos_i, pos_j))) {
      max_score = cur_score;
      pos_i = ref_size;
      pos_j = j;
      // overhang at end of alt → 記成 soft clip
      segment_len = alt_size - j;
    }
  }

  Cigar cigar{};
  if (segment_len > 0) {
    cigar.emplace_back(static_cast<unsigned>(segment_len), 'S');
    segment_len = 0;
  }

  char state = 'M';

  do {
    const int cur_trace = trace[idx(pos_i, pos_j)];
    char new_state;
    int  step_size;

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

    if (new_state == state) {
      segment_len += step_size;
    } else {
      if (segment_len > 0)
        cigar.emplace_back(static_cast<unsigned>(segment_len), state);
      segment_len = step_size;
      state = new_state;
    }
  } while (pos_i > 0 && pos_j > 0);

  if (segment_len > 0)
    cigar.emplace_back(static_cast<unsigned>(segment_len), state);

  const int align_offset = pos_i;

  if (pos_j > 0)
    cigar.emplace_back(static_cast<unsigned>(pos_j), 'S');

  cigar.reverse();
  return std::make_pair(align_offset, cigar);
}

// ----------------- Batched interface -----------------

auto SmithWatermanCuda::align_batch(const std::vector<TaskView>& tasks,
                                    Parameters params)
  -> std::vector<std::pair<int, Cigar>>
{
  const int n = static_cast<int>(tasks.size());
  if (n == 0)
    return {};

  // 目前簡化假設：所有 ref 長度一樣、所有 alt 長度一樣
  const std::size_t ref_len = tasks.front().ref.size();
  const std::size_t alt_len = tasks.front().alt.size();

  for (const auto& t : tasks) {
    if (t.ref.size() != ref_len || t.alt.size() != alt_len) {
      throw std::runtime_error(
        "SmithWatermanCuda::align_batch currently requires all refs/alts "
        "to have the same length");
    }
  }

  const int rows = static_cast<int>(ref_len) + 1;
  const int cols = static_cast<int>(alt_len) + 1;
  const std::size_t mat_size = static_cast<std::size_t>(rows) * cols;

  // host flattened buffers
  std::vector<char> h_ref_all(n * ref_len);
  std::vector<char> h_alt_all(n * alt_len);

  for (int i = 0; i < n; ++i) {
    std::copy_n(tasks[i].ref.data(), ref_len,
                h_ref_all.data() + static_cast<std::size_t>(i) * ref_len);
    std::copy_n(tasks[i].alt.data(), alt_len,
                h_alt_all.data() + static_cast<std::size_t>(i) * alt_len);
  }

  std::vector<int> h_score_all(n * mat_size, 0);
  std::vector<int> h_trace_all(n * mat_size, 0);

  // device buffers
  char* d_ref_all = nullptr;
  char* d_alt_all = nullptr;
  int*  d_score_all = nullptr;
  int*  d_trace_all = nullptr;

  int*  d_gap_size_down_all  = nullptr;
  int*  d_best_gap_down_all  = nullptr;
  int*  d_gap_size_right_all = nullptr;
  int*  d_best_gap_right_all = nullptr;

  const std::size_t ref_bytes = h_ref_all.size() * sizeof(char);
  const std::size_t alt_bytes = h_alt_all.size() * sizeof(char);
  const std::size_t mat_bytes = h_score_all.size() * sizeof(int);

  check_cuda(cudaMalloc(&d_ref_all, ref_bytes), "cudaMalloc d_ref_all");
  check_cuda(cudaMalloc(&d_alt_all, alt_bytes), "cudaMalloc d_alt_all");
  check_cuda(cudaMalloc(&d_score_all, mat_bytes), "cudaMalloc d_score_all");
  check_cuda(cudaMalloc(&d_trace_all, mat_bytes), "cudaMalloc d_trace_all");

  // gap arrays：每個 alignment 各有 (cols+1) / (rows+1) 長度
  check_cuda(cudaMalloc(&d_gap_size_down_all,
                        static_cast<std::size_t>(n) * (cols + 1) * sizeof(int)),
             "cudaMalloc gap_size_down_all");
  check_cuda(cudaMalloc(&d_best_gap_down_all,
                        static_cast<std::size_t>(n) * (cols + 1) * sizeof(int)),
             "cudaMalloc best_gap_down_all");
  check_cuda(cudaMalloc(&d_gap_size_right_all,
                        static_cast<std::size_t>(n) * (rows + 1) * sizeof(int)),
             "cudaMalloc gap_size_right_all");
  check_cuda(cudaMalloc(&d_best_gap_right_all,
                        static_cast<std::size_t>(n) * (rows + 1) * sizeof(int)),
             "cudaMalloc best_gap_right_all");

  check_cuda(cudaMemcpy(d_ref_all, h_ref_all.data(), ref_bytes,
                        cudaMemcpyHostToDevice),
             "cudaMemcpy ref_all");
  check_cuda(cudaMemcpy(d_alt_all, h_alt_all.data(), alt_bytes,
                        cudaMemcpyHostToDevice),
             "cudaMemcpy alt_all");

  // 一個 block 處理一個 alignment，目前每個 block 只用單一 thread
  dim3 grid(n);
  dim3 block(1);
  smith_waterman_kernel<<<grid, block>>>(
      d_ref_all, d_alt_all,
      static_cast<int>(ref_len), static_cast<int>(alt_len),
      params.w_match, params.w_mismatch,
      params.w_open, params.w_extend,
      d_score_all, d_trace_all,
      d_gap_size_down_all, d_best_gap_down_all,
      d_gap_size_right_all, d_best_gap_right_all,
      n);

  check_cuda(cudaGetLastError(), "kernel launch");
  check_cuda(cudaDeviceSynchronize(), "kernel sync");

  check_cuda(cudaMemcpy(h_score_all.data(), d_score_all, mat_bytes,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy score_all back");
  check_cuda(cudaMemcpy(h_trace_all.data(), d_trace_all, mat_bytes,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy trace_all back");

  cudaFree(d_ref_all);
  cudaFree(d_alt_all);
  cudaFree(d_score_all);
  cudaFree(d_trace_all);
  cudaFree(d_gap_size_down_all);
  cudaFree(d_best_gap_down_all);
  cudaFree(d_gap_size_right_all);
  cudaFree(d_best_gap_right_all);

  std::vector<std::pair<int, Cigar>> results;
  results.reserve(n);

  for (int i = 0; i < n; ++i) {
    const int* score_ptr = h_score_all.data() + static_cast<std::size_t>(i) * mat_size;
    const int* trace_ptr = h_trace_all.data() + static_cast<std::size_t>(i) * mat_size;

    results.push_back(
      traceback_and_build_cigar(score_ptr, trace_ptr,
                                static_cast<int>(ref_len),
                                static_cast<int>(alt_len)));
  }

  return results;
}

// ----------------- Single-pair interface -----------------

auto SmithWatermanCuda::align(std::string_view ref, std::string_view alt,
                              Parameters params)
  -> std::pair<int, Cigar>
{
  assert(!ref.empty() && !alt.empty());

  // 和 CPU 版一樣的 quick path
  if (alt.size() == ref.size() && well_match(ref, alt)) {
    const std::string cigar_str = std::to_string(ref.size()) + "M";
    return {0, Cigar(cigar_str)};
  }

  // 一般情況用 batched 實作處理（batch size = 1）
  std::vector<TaskView> tasks;
  tasks.push_back(TaskView{ref, alt});

  auto results = align_batch(tasks, params);
  return results.front();
}

} // namespace biovoltron
