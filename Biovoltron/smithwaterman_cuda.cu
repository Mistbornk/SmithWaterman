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
// 簡單版：一個 alignment，一個 thread，直接在 device 上跑 CPU 那個 double loop
__global__ void smith_waterman_kernel(const char* ref, int ref_len,
                                      const char* alt, int alt_len,
                                      int w_match, int w_mismatch,
                                      int w_open, int w_extend,
                                      int* score, int* trace,
                                      int* gap_size_down,
                                      int* best_gap_down,
                                      int* gap_size_right,
                                      int* best_gap_right)
{
  // 暫時只讓 thread 0 做事
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  const int rows = ref_len + 1;
  const int cols = alt_len + 1;

  auto idx = [cols](int i, int j) {
    return i * cols + j;
  };

  // 初始化 score / trace
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      score[idx(i,j)] = 0;
      trace[idx(i,j)] = 0;
    }
  }

  // gap tracking arrays（和 CPU 版相同邏輯），由 host 配好記憶體，這裡只初始化

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
      const int diag_score = score[idx(i-1, j-1)];
      const int step_diag =
        diag_score + ((ref[i-1] == alt[j-1]) ? w_match : w_mismatch);

      // Gap in ref (down)
      const int gap_open_down = score[idx(i-1, j)] + w_open;
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
      const int gap_open_right = score[idx(i, j-1)] + w_open;
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
        score[idx(i,j)] = step_diag;
        trace[idx(i,j)] = 0;             // Diagonal
      } else if (step_right >= step_down) {
        score[idx(i,j)] = step_right;
        trace[idx(i,j)] = -step_right_size; // Insertion (right)
      } else {
        score[idx(i,j)] = step_down;
        trace[idx(i,j)] = step_down_size;   // Deletion (down)
      }
    }
  }
}

// ----------------- Host-side traceback (flattened) -----------------

static std::pair<int, Cigar>
traceback_and_build_cigar(const std::vector<int>& score,
                          const std::vector<int>& trace,
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

// Interface
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

  const int ref_len = static_cast<int>(ref.size());
  const int alt_len = static_cast<int>(alt.size());
  const int rows = ref_len + 1;
  const int cols = alt_len + 1;
  const size_t mat_size = static_cast<size_t>(rows) * cols;

  // host buffer（flattened）
  std::vector<int> h_score(mat_size, 0);
  std::vector<int> h_trace(mat_size, 0);

  // device buffer
  char* d_ref = nullptr;
  char* d_alt = nullptr;
  int*  d_score = nullptr;
  int*  d_trace = nullptr;

  int*  d_gap_size_down  = nullptr;
  int*  d_best_gap_down  = nullptr;
  int*  d_gap_size_right = nullptr;
  int*  d_best_gap_right = nullptr;

  check_cuda(cudaMalloc(&d_ref, ref_len * sizeof(char)), "cudaMalloc d_ref");
  check_cuda(cudaMalloc(&d_alt, alt_len * sizeof(char)), "cudaMalloc d_alt");
  check_cuda(cudaMalloc(&d_score, mat_size * sizeof(int)), "cudaMalloc d_score");
  check_cuda(cudaMalloc(&d_trace, mat_size * sizeof(int)), "cudaMalloc d_trace");

  // gap arrays 長度：down 按 j (col)、right 按 i (row)
  check_cuda(cudaMalloc(&d_gap_size_down,  (cols + 1) * sizeof(int)), "cudaMalloc gap_size_down");
  check_cuda(cudaMalloc(&d_best_gap_down,  (cols + 1) * sizeof(int)), "cudaMalloc best_gap_down");
  check_cuda(cudaMalloc(&d_gap_size_right, (rows + 1) * sizeof(int)), "cudaMalloc gap_size_right");
  check_cuda(cudaMalloc(&d_best_gap_right, (rows + 1) * sizeof(int)), "cudaMalloc best_gap_right");

  check_cuda(cudaMemcpy(d_ref, ref.data(), ref_len * sizeof(char),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy ref");
  check_cuda(cudaMemcpy(d_alt, alt.data(), alt_len * sizeof(char),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy alt");

  // 目前先啟一個 block、一個 thread
  dim3 grid(1);
  dim3 block(1);
  smith_waterman_kernel<<<grid, block>>>(
      d_ref, ref_len,
      d_alt, alt_len,
      params.w_match, params.w_mismatch,
      params.w_open, params.w_extend,
      d_score, d_trace,
      d_gap_size_down, d_best_gap_down,
      d_gap_size_right, d_best_gap_right);

  check_cuda(cudaGetLastError(), "kernel launch");
  check_cuda(cudaDeviceSynchronize(), "kernel sync");

  check_cuda(cudaMemcpy(h_score.data(), d_score,
                        mat_size * sizeof(int),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy score back");
  check_cuda(cudaMemcpy(h_trace.data(), d_trace,
                        mat_size * sizeof(int),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy trace back");

  cudaFree(d_ref);
  cudaFree(d_alt);
  cudaFree(d_score);
  cudaFree(d_trace);
  cudaFree(d_gap_size_down);
  cudaFree(d_best_gap_down);
  cudaFree(d_gap_size_right);
  cudaFree(d_best_gap_right);

  return traceback_and_build_cigar(h_score, h_trace, ref_len, alt_len);
}

} // namespace biovoltron