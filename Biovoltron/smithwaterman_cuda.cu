#include <Biovoltron/smithwaterman_cuda.hpp>
#include <cuda_runtime.h>
#include <stdexcept>

#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#include <climits>
#include <memory>

namespace biovoltron {

// ----------------- CUDA error helper -----------------
static inline void check_cuda(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
  }
}

struct CudaDeleter {
  void operator()(void* ptr) const noexcept {
    if (ptr) cudaFree(ptr);
  }
};

struct CudaHostDeleter {
  void operator()(void* ptr) const noexcept {
    if (ptr) cudaFreeHost(ptr);
  }
};

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, CudaDeleter>;

template <typename T>
using cuda_host_unique_ptr = std::unique_ptr<T, CudaHostDeleter>;

template <typename T>
static cuda_unique_ptr<T> make_device_buffer(std::size_t count, const char* msg) {
  T* ptr = nullptr;
  check_cuda(cudaMalloc(&ptr, count * sizeof(T)), msg);
  return cuda_unique_ptr<T>(ptr);
}

template <typename T>
static cuda_host_unique_ptr<T> make_pinned_buffer(std::size_t count, const char* msg) {
  T* ptr = nullptr;
  check_cuda(cudaMallocHost(reinterpret_cast<void**>(&ptr), count * sizeof(T)), msg);
  return cuda_host_unique_ptr<T>(ptr);
}

// --- CUDA kernel ---
// Batched 版：一個 block 負責一個 alignment，
// 目前仍由 thread 0 在 block 裡跑 DP + traceback（之後可以再做 finer-grained 平行化）
__global__ void smith_waterman_kernel(const char* __restrict__ all_refs,
                                      const char* __restrict__ all_alts,
                                      int ref_len, int alt_len,
                                      int w_match, int w_mismatch,
                                      int w_open, int w_extend,
                                      int* __restrict__ all_scores, int* __restrict__ all_traces,
                                      int* __restrict__ all_gap_size_down,
                                      int* __restrict__ all_best_gap_down,
                                      int* __restrict__ all_gap_size_right,
                                      int* __restrict__ all_best_gap_right,
                                      // CIGAR 輸出相關
                                      unsigned* __restrict__ all_cigar_lens,
                                      char*     __restrict__ all_cigar_ops,
                                      int*      __restrict__ cigar_start,
                                      int*      __restrict__ cigar_count,
                                      int*      __restrict__ align_offsets,
                                      int       max_cigar_ops,
                                      int       n_alignments)
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

  // 這個 alignment 對應的 CIGAR 區段
  const int cigar_base = aln * max_cigar_ops;

  auto idx = [cols](int i, int j) {
    return i * cols + j;
  };

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

  // --- DP 填表 ---
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

  // --- 在 GPU 上做 traceback + CIGAR 產生 ---

  const int ref_size = ref_len;
  const int alt_size = alt_len;

  // ⬅ 這裡原本用 std::numeric_limits<int>::min()，改成 INT_MIN 避免 NVCC 錯誤
  int max_score = INT_MIN;
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

  // 我們會把 CIGAR 元素寫到 global array 的尾端往前寫：
  //   先寫出「反向」CIGAR（像 CPU 版的 vector），
  //   再利用從尾端往前寫的方式，讓 host 讀出時自動變成正向。
  int write_pos = max_cigar_ops - 1;
  int elem_count = 0;

  auto emit_cigar = [&](unsigned len, char op) {
    if (len == 0) return;
    if (write_pos < 0) return;  // 避免 out-of-bounds；實務上 max_cigar_ops 要設夠大
    all_cigar_lens[cigar_base + write_pos] = len;
    all_cigar_ops[cigar_base + write_pos] = op;
    --write_pos;
    ++elem_count;
  };

  // 若尾端有 overhang → 先寫一個 'S'
  if (segment_len > 0) {
    emit_cigar(static_cast<unsigned>(segment_len), 'S');
    segment_len = 0;
  }

  char state = 'M';

  // 主 traceback loop
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
      emit_cigar(static_cast<unsigned>(segment_len), state);
      segment_len = step_size;
      state = new_state;
    }
  } while (pos_i > 0 && pos_j > 0);

  // 最後累積的 segment
  emit_cigar(static_cast<unsigned>(segment_len), state);

  const int align_offset = pos_i;

  // 如果 alt 還有剩 → leading soft-clip
  if (pos_j > 0) {
    emit_cigar(static_cast<unsigned>(pos_j), 'S');
  }

  // 此時 emit 出來的序列順序是「反向」，但我們是從陣列尾端往前寫，
  // 例如 L0, L1, L2 寫到 positions: ..., 9,8,7，
  // host 會從 start = 7, len = 3，依序讀 7,8,9 => L2, L1, L0，
  // 正好就是正向的 CIGAR。
  cigar_start[aln]   = write_pos + 1;
  cigar_count[aln]   = elem_count;
  align_offsets[aln] = align_offset;
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

  // 為了安全，CIGAR 最多可能是 ref_len + alt_len + 2 個 element
  // （全 I + 全 D + 兩個 S 等極端情形）
  const int max_cigar_ops = static_cast<int>(ref_len + alt_len + 2);

  // host flattened buffers（pinned，較快的 H2D 拷貝）
  auto h_ref_all = make_pinned_buffer<char>(static_cast<std::size_t>(n) * ref_len,
                                            "cudaMallocHost h_ref_all");
  auto h_alt_all = make_pinned_buffer<char>(static_cast<std::size_t>(n) * alt_len,
                                            "cudaMallocHost h_alt_all");

  for (int i = 0; i < n; ++i) {
    std::copy_n(tasks[i].ref.data(), ref_len,
                h_ref_all.get() + static_cast<std::size_t>(i) * ref_len);
    std::copy_n(tasks[i].alt.data(), alt_len,
                h_alt_all.get() + static_cast<std::size_t>(i) * alt_len);
  }

  const std::size_t ref_bytes = static_cast<std::size_t>(n) * ref_len * sizeof(char);
  const std::size_t alt_bytes = static_cast<std::size_t>(n) * alt_len * sizeof(char);

  // score/trace matrix（只在 device 使用，不會拷回 host）
  const std::size_t mat_bytes_all = static_cast<std::size_t>(n) * mat_size * sizeof(int);

  // CIGAR 緩衝區總大小
  const std::size_t cigar_total_ops = static_cast<std::size_t>(n) * max_cigar_ops;
  const std::size_t cigar_lens_bytes = cigar_total_ops * sizeof(unsigned);
  const std::size_t cigar_ops_bytes  = cigar_total_ops * sizeof(char);

  // device buffers
  auto d_ref_all           = make_device_buffer<char>(static_cast<std::size_t>(n) * ref_len,
                                                      "cudaMalloc d_ref_all");
  auto d_alt_all           = make_device_buffer<char>(static_cast<std::size_t>(n) * alt_len,
                                                      "cudaMalloc d_alt_all");
  auto d_score_all         = make_device_buffer<int>(static_cast<std::size_t>(n) * mat_size,
                                                     "cudaMalloc d_score_all");
  auto d_trace_all         = make_device_buffer<int>(static_cast<std::size_t>(n) * mat_size,
                                                     "cudaMalloc d_trace_all");
  auto d_gap_size_down_all = make_device_buffer<int>(static_cast<std::size_t>(n) * (cols + 1),
                                                     "cudaMalloc gap_size_down_all");
  auto d_best_gap_down_all = make_device_buffer<int>(static_cast<std::size_t>(n) * (cols + 1),
                                                     "cudaMalloc best_gap_down_all");
  auto d_gap_size_right_all= make_device_buffer<int>(static_cast<std::size_t>(n) * (rows + 1),
                                                     "cudaMalloc gap_size_right_all");
  auto d_best_gap_right_all= make_device_buffer<int>(static_cast<std::size_t>(n) * (rows + 1),
                                                     "cudaMalloc best_gap_right_all");
  auto d_cigar_lens        = make_device_buffer<unsigned>(cigar_total_ops, "cudaMalloc d_cigar_lens");
  auto d_cigar_ops         = make_device_buffer<char>(cigar_total_ops, "cudaMalloc d_cigar_ops");
  auto d_cigar_start       = make_device_buffer<int>(n, "cudaMalloc d_cigar_start");
  auto d_cigar_count       = make_device_buffer<int>(n, "cudaMalloc d_cigar_count");
  auto d_align_offsets     = make_device_buffer<int>(n, "cudaMalloc d_align_offsets");

  check_cuda(cudaMemcpy(d_ref_all.get(), h_ref_all.get(), ref_bytes,
                        cudaMemcpyHostToDevice),
             "cudaMemcpy ref_all");
  check_cuda(cudaMemcpy(d_alt_all.get(), h_alt_all.get(), alt_bytes,
                        cudaMemcpyHostToDevice),
             "cudaMemcpy alt_all");

  // score/trace and gap-size arrays are zeroed up-front to avoid single-thread clearing inside the kernel
  check_cuda(cudaMemset(d_score_all.get(), 0, mat_bytes_all), "cudaMemset score_all");
  check_cuda(cudaMemset(d_trace_all.get(), 0, mat_bytes_all), "cudaMemset trace_all");
  check_cuda(cudaMemset(d_gap_size_down_all.get(), 0,
                        static_cast<std::size_t>(n) * (cols + 1) * sizeof(int)),
             "cudaMemset gap_size_down_all");
  check_cuda(cudaMemset(d_gap_size_right_all.get(), 0,
                        static_cast<std::size_t>(n) * (rows + 1) * sizeof(int)),
             "cudaMemset gap_size_right_all");

  // 一個 block 處理一個 alignment，目前每個 block 只用單一 thread
  dim3 grid(n);
  dim3 block(1);
  smith_waterman_kernel<<<grid, block>>>(
      d_ref_all.get(), d_alt_all.get(),
      static_cast<int>(ref_len), static_cast<int>(alt_len),
      params.w_match, params.w_mismatch,
      params.w_open, params.w_extend,
      d_score_all.get(), d_trace_all.get(),
      d_gap_size_down_all.get(), d_best_gap_down_all.get(),
      d_gap_size_right_all.get(), d_best_gap_right_all.get(),
      d_cigar_lens.get(), d_cigar_ops.get(),
      d_cigar_start.get(), d_cigar_count.get(),
      d_align_offsets.get(),
      max_cigar_ops,
      n);

  check_cuda(cudaGetLastError(), "kernel launch");
  check_cuda(cudaDeviceSynchronize(), "kernel sync");

  // 將 CIGAR 結果帶回 host（比整個 score/trace matrix 小很多）
  auto h_cigar_lens    = make_pinned_buffer<unsigned>(cigar_total_ops, "cudaMallocHost cigar_lens");
  auto h_cigar_ops     = make_pinned_buffer<char>(cigar_total_ops, "cudaMallocHost cigar_ops");
  auto h_cigar_start   = make_pinned_buffer<int>(n, "cudaMallocHost cigar_start");
  auto h_cigar_count   = make_pinned_buffer<int>(n, "cudaMallocHost cigar_count");
  auto h_align_offsets = make_pinned_buffer<int>(n, "cudaMallocHost align_offsets");

  check_cuda(cudaMemcpy(h_cigar_lens.get(), d_cigar_lens.get(), cigar_lens_bytes,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy cigar_lens");
  check_cuda(cudaMemcpy(h_cigar_ops.get(), d_cigar_ops.get(), cigar_ops_bytes,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy cigar_ops");
  check_cuda(cudaMemcpy(h_cigar_start.get(), d_cigar_start.get(), n * sizeof(int),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy cigar_start");
  check_cuda(cudaMemcpy(h_cigar_count.get(), d_cigar_count.get(), n * sizeof(int),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy cigar_count");
  check_cuda(cudaMemcpy(h_align_offsets.get(), d_align_offsets.get(), n * sizeof(int),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy align_offsets");

  // 在 host 上組回 Cigar 物件
  std::vector<std::pair<int, Cigar>> results;
  results.reserve(n);

  for (int aln = 0; aln < n; ++aln) {
    const int base = aln * max_cigar_ops;
    const int start = h_cigar_start.get()[aln];
    const int cnt   = h_cigar_count.get()[aln];

    Cigar cigar{};
    for (int k = 0; k < cnt; ++k) {
      const int idx = start + k;
      const unsigned len = h_cigar_lens.get()[base + idx];
      const char op      = h_cigar_ops.get()[base + idx];
      cigar.emplace_back(len, op);
    }

    results.emplace_back(h_align_offsets.get()[aln], std::move(cigar));
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
