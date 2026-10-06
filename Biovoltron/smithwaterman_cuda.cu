#include <Biovoltron/smithwaterman_cuda.hpp>

#include <cuda_runtime.h>

#include <cassert>
#include <climits>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <iostream>

#define CUDACHECKASYNC                                       \
{ cudaError_t err = cudaPeekAtLastError();                   \
  if(err != cudaSuccess) {                                  \
    printf("CUDA error %s at %s:%d\n",                       \
           cudaGetErrorString(err), __FILE__, __LINE__);     \
    exit(static_cast<int>(err));                             \
  }                                                          \
}

// 同步檢查 CUDA API 呼叫（malloc/memcpy/getDevice 等）
#define CUDA_CHECK(call)                                                    \
do {                                                                        \
  cudaError_t _e = (call);                                                  \
  if (_e != cudaSuccess) {                                                  \
    fprintf(stderr, "CUDA error %s at %s:%d\n",                             \
            cudaGetErrorString(_e), __FILE__, __LINE__);                    \
    exit(static_cast<int>(_e));                                             \
  }                                                                         \
} while (0)

// ============================================================================
// Internal CUDA utilities (warp helpers, wavefront SW core)
// ============================================================================

namespace {

constexpr int WARP_SIZE       = 32;
// 一個 block 有多少 warp，同時處理多少 pair。
// 4070 Ti Super 的 sweet spot 大約 4~8，這裡用 8 (=256 threads/block)。
constexpr int WARPS_PER_BLOCK = 8;

__device__ __forceinline__ int warp_tid()
{
    return threadIdx.x & (WARP_SIZE - 1);
}

template <typename T>
__device__ __forceinline__ T shfl_up(T v, int delta)
{
    return __shfl_up_sync(0xffffffffu, v, delta);
}

template <typename T>
__device__ __forceinline__ T shfl_down(T v, int delta)
{
    return __shfl_down_sync(0xffffffffu, v, delta);
}

__device__ __forceinline__ int dmax(int a, int b)
{
    return a > b ? a : b;
}

__device__ __forceinline__ int dmin(int a, int b)
{
    return a < b ? a : b;
}

__device__ __forceinline__ int dmax3(int a, int b, int c)
{
    return dmax(a, dmax(b, c));
}

__device__ __forceinline__ int dmin3(int a, int b, int c)
{
    return dmin(a, dmin(b, c));
}

struct AlignmentResult
{
    int   score;
    int   j;   // row   (in reference)
    int   i;   // column (in query)

    __device__ __forceinline__
    AlignmentResult() : score(INT_MIN / 2), j(0), i(0) {}

    __device__ __forceinline__
    AlignmentResult(int s, int jj, int ii) : score(s), j(jj), i(ii) {}
};

/**
 * Wavefront Smith-Waterman for a single alignment, executed by exactly one warp.
 *
 * - ref: length N
 * - alt: length M
 * - Each thread in the warp corresponds to one column within a 32-wide stripe.
 */
__device__ __forceinline__
void sw_warp_device(const char* __restrict__ ref,
                    const char* __restrict__ alt,
                    int N,                   // ref length
                    int M,                   // alt length
                    int match_score,
                    int mismatch_score,
                    int gap_open,            // vertical gaps
                    int gap_extend,          // horizontal gaps
                    int* __restrict__ best_score_out,
                    int2* __restrict__ sink_out,
                    int* __restrict__ temp_h,
                    int* __restrict__ temp_f,
                    int* __restrict__ temp_len_f,
                    int8_t* __restrict__ trace,
                    int trace_stride)
{
    // -infinite
    const int NEG_INF = INT_MIN / 2;
    // thread id (1-indexed)
    const unsigned int lane = warp_tid();  // 0..31

    // 初始化 DP column buffers
    for (int j = lane; j <= N; j += WARP_SIZE) {
        temp_h[j]     = 0;
        temp_f[j]     = NEG_INF;
        temp_len_f[j] = 0;
    }

    __syncwarp();

    // H(i-1,j), H(i,j-1), H(i-1,j-1), current score
    int h_top  = 0, h_left = 0, h_diag = 0, h_val  = 0;
    // E(i-1,j), E(i,j)
    int e_top  = NEG_INF, e_val = NEG_INF;
    // F(i,j-1), F(i,j)
    int f_left = NEG_INF, f_val = NEG_INF;

    int len_e_top  = 0, len_e_val  = 0;
    int len_f_left = 0, len_f_val  = 0;

    AlignmentResult best;

    unsigned char r_j             = 0;
    int temp_h_cache              = 0;
    int temp_f_cache              = 0;
    int temp_len_f_cache          = 0;
    unsigned char reference_cache = 0;

    const unsigned int wi = lane + 1; // current warp stripe's column index (1..WARP_SIZE)

    // warp block
    /*
            [<------------------  M (str for alt)  -------------------->]
        |   [<----- 32 ----->][<----- 32 ----->][<----- 32 ----->][<...>]
        |   +-----------------+-----------------+-----------------+-----+
        N   |                 |                 |                 |     |
        |   |                 |                 |                 |     |
        r   |     Stripe 1    |      Stripe 2   |      Stripe 3   | ... |
        e   | warp_block=0-31 |warp_block=32-63 |warp_block=64-95 |     |
        f   |                 |                 |                 |     |
        |   +-----------------+-----------------+-----------------+-----+
    */
    // wi is local thread offset
    /*
                <----------------- Stripe 1 (Width 32) ----------------->
                    Col 1            Col 2        Col 3    ...   Col 32
                +---------------+-----------+-------------+---+---------+
            j=1 |               |           |             |...|         |
                +---------------+-----------+-------------+---+---------+
            j=2 |               |           |             |...|         |
                +---------------+-----------+-------------+---+---------+
                ...             |           |             |...|         |
                +---------------+-----------+-------------+---+---------+
                ^               ^               ^                       ^
                |               |               |                       |
        Thread  1               2               3       ...         Thread 32
        (wi =   1               2               3       ...          wi=32)
    */
    // Through query (alt) do  WARP_SIZE stripe
    for (int warp_block = 0; warp_block < M; warp_block += WARP_SIZE)
    {
        const unsigned int warp_block_width =
            (warp_block + WARP_SIZE >= M) ? (M - warp_block) : WARP_SIZE;

        const unsigned int i = wi + warp_block; // global column index (1..M)

        h_top  = 0;         // H(0, j)
        h_diag = 0;         // H(0, j-1)
        e_top  = NEG_INF;   // E(0, j)
        f_left = NEG_INF;   // F(i, 0)
        len_e_top = 0;
        len_f_left = 0;

        // The alt char for this thread
        const unsigned char s_i =
            (i <= static_cast<unsigned int>(M))
                ? static_cast<unsigned char>(alt[i - 1])
                : 0;

// For the stripe's anti-diagonals
        /*
        block_diag = 2 (example):

         i = 1   2   3...31  32
             +---+---+...+---+---+....
        j=1  |   |   |   |   | w |  <-- Thread 32 (wi=32), j = 5-4 = 1
             +---+---+...+---+---+....
        j=2  |   |   |   | w |   |  <-- Thread 31 (wi=31), j = 5-3 = 2
             +---+---+...+---+---+....
        ...  |                   |
             +---+---+...+---+---+....
        j=31 |   | w |   |   |   |  <-- Thread 2 (wi=2), j = 5-2 = 3
             +---+---+...+---+---+....
        j=32 | w |   |   |   |   |  <-- Thread 1 (wi=1), j = 5-1 = 4
             +---+---+...+---+---+....
        */
        for (unsigned int block_diag = 2;
             block_diag <= warp_block_width + static_cast<unsigned>(N);
             block_diag += WARP_SIZE)
        {
            // Every WARP_SIZE anti-diagonals reload cache
            const unsigned int cache_row = (block_diag - 2) + lane;
            if (cache_row < static_cast<unsigned>(N)) {
                temp_h_cache     = temp_h[cache_row];
                temp_f_cache     = temp_f[cache_row];
                temp_len_f_cache = temp_len_f[cache_row];
                reference_cache  = static_cast<unsigned char>(ref[cache_row]);
            } else {
                temp_h_cache     = 0;
                temp_f_cache     = NEG_INF;
                temp_len_f_cache = 0;
                reference_cache  = 0;
            }

            for (unsigned int diag = block_diag;
                 diag < block_diag + WARP_SIZE; ++diag)
            {
                const unsigned int diag_len =
                    dmin3(diag - 1,
                          static_cast<unsigned>(WARP_SIZE),
                          warp_block_width);

                const unsigned int j = diag - wi; // row index (1..N)

                if (wi <= diag_len && j <= static_cast<unsigned>(N))
                {
                    if (wi == 1)
                    {
                        // thread 1 read reference char and left cell value
                        // and for lane 0 (first warp) will read new reference char in his row and left value=0
                        r_j        = reference_cache;
                        h_left     = temp_h_cache;
                        f_left     = temp_f_cache;
                        len_f_left = temp_len_f_cache;
                    }

                    const int S_ij =
                        (r_j == s_i) ? match_score : mismatch_score;
                    int score_diag = h_diag + S_ij;

                    // E (Vertical)
                    int e_open   = h_top + gap_open;
                    int e_extend = e_top + gap_extend;
                    if (e_open > e_extend) {
                        e_val = e_open;
                        len_e_val = 1;
                    } else {
                        e_val = e_extend;
                        len_e_val = len_e_top + 1;
                    }

                    // F (Horizontal)
                    int f_open   = h_left + gap_open;
                    int f_extend = f_left + gap_extend;
                    if (f_open > f_extend) {
                        f_val = f_open;
                        len_f_val = 1;
                    } else {
                        f_val = f_extend;
                        len_f_val = len_f_left + 1;
                    }

                    h_val = dmax3(score_diag, e_val, f_val);

                    // Traceback encoding: 0=diag(M), >0=up(D,len), <0=left(I,len)
                    int8_t trace_val = 0;
                    if (h_val == score_diag) {
                        trace_val = 0;
                    } else if (h_val == f_val) {
                        int l = (len_f_val > 127) ? 127 : len_f_val;
                        trace_val = static_cast<int8_t>(-l);
                    } else {
                        int l = (len_e_val > 127) ? 127 : len_e_val;
                        trace_val = static_cast<int8_t>(l);
                    }

                    trace[size_t(j) * trace_stride + i] = trace_val;

                    if (wi == WARP_SIZE) {
                        temp_h[j - 1]     = h_val;
                        temp_f[j - 1]     = f_val;
                        temp_len_f[j - 1] = len_f_val;
                    }

                    bool is_last_col = (i == static_cast<unsigned>(M));
                    bool is_last_row = (j == static_cast<unsigned>(N));

                    if (is_last_col || is_last_row)
                    {
                        if (h_val > best.score) {
                            best = AlignmentResult(h_val, j, i);
                        }
                    }

                    h_diag    = h_left;
                    h_top     = h_val;
                    e_top     = e_val;
                    f_left    = f_val;
                    len_e_top = len_e_val;
                }

                // warp shifts
                r_j           = shfl_up(r_j, 1);
                h_left        = shfl_up(h_val, 1);
                f_left        = shfl_up(f_val, 1);
                len_f_left    = shfl_up(len_f_val, 1);

                temp_h_cache     = shfl_down(temp_h_cache, 1);
                temp_f_cache     = shfl_down(temp_f_cache, 1);
                temp_len_f_cache = shfl_down(temp_len_f_cache, 1);
                reference_cache  = shfl_down(reference_cache, 1);
            }
        }
    }

    // warp reduction to find best score and sink
    int best_score = best.score;
    int best_j     = best.j;
    int best_i     = best.i;

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        int other_score = __shfl_down_sync(0xffffffffu, best_score, offset);
        int other_j     = __shfl_down_sync(0xffffffffu, best_j,     offset);
        int other_i     = __shfl_down_sync(0xffffffffu, best_i,     offset);

        if (other_score > best_score ||
           (other_score == best_score && other_score > INT_MIN / 2))
        {
            best_score = other_score;
            best_j     = other_j;
            best_i     = other_i;
        }
    }

    if (warp_tid() == 0)
    {
        *best_score_out = best_score;
        sink_out->x     = best_j;  // row in ref
        sink_out->y     = best_i;  // column in alt
    }
}

// Simple single-pair wrapper kernel (one warp per block).
__global__
void sw_warp_kernel(const char* __restrict__ ref,
                    const char* __restrict__ alt,
                    int N,                   // ref length
                    int M,                   // alt length
                    int match_score,
                    int mismatch_score,
                    int gap_open,        // vertical gaps
                    int gap_extend,       // horizontal gaps
                    int* __restrict__ best_score_out,
                    int2* __restrict__ sink_out,
                    int* __restrict__ global_temp_h,
                    int* __restrict__ global_temp_f,
                    int* __restrict__ global_temp_len_f,
                    int8_t* __restrict__ trace)
{
    extern __shared__ int smem[];
    int* temp_h;
    int* temp_f;
    int* temp_len_f;

    if (global_temp_h == nullptr) {
        temp_h     = smem;
        temp_f     = smem + (N + 1);
        temp_len_f = smem + 2 * (N + 1);
    } else {
        temp_h     = global_temp_h;
        temp_f     = global_temp_f;
        temp_len_f = global_temp_len_f;
    }

    sw_warp_device(ref, alt, N, M,
                   match_score, mismatch_score,
                   gap_open, gap_extend,
                   best_score_out, sink_out,
                   temp_h, temp_f, temp_len_f,
                   trace, M + 1);
}

/**
 * 強化版 batch kernel：
 * - 每個 warp 負責一個 (ref, alt) pair
 * - 每個 block 有 WARPS_PER_BLOCK 個 warp
 */
__global__
void sw_batch_kernel(const char* __restrict__ all_refs,
                     const char* __restrict__ all_alts,
                     const int* __restrict__ ref_offsets,
                     const int* __restrict__ alt_offsets,
                     const int* __restrict__ ref_lengths,
                     const int* __restrict__ alt_lengths,
                     int num_pairs,
                     int match_score,
                     int mismatch_score,
                     int gap_open,
                     int gap_extend,
                     int* __restrict__ all_best_scores,
                     int2* __restrict__ all_sinks,
                     int8_t* __restrict__ all_traces,
                     const long long* __restrict__ trace_offsets,
                     int* __restrict__ global_temp_buffer, // Large buffer for temp arrays if needed
                     int max_ref_len)
{
    int warp_id_in_block = threadIdx.x / WARP_SIZE;    // 0..WARPS_PER_BLOCK-1
    if (warp_id_in_block >= WARPS_PER_BLOCK) return;

    int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    int pair_idx       = global_warp_id;
    if (pair_idx >= num_pairs) return;

    int N = ref_lengths[pair_idx];
    int M = alt_lengths[pair_idx];

    const char* ref = all_refs + ref_offsets[pair_idx];
    const char* alt = all_alts + alt_offsets[pair_idx];

    int*  best_score_out = all_best_scores + pair_idx;
    int2* sink_out       = all_sinks       + pair_idx;
    int8_t* trace        = all_traces      + trace_offsets[pair_idx];

    int* temp_h;
    int* temp_f;
    int* temp_len_f;

    if (global_temp_buffer != nullptr) {
        // 為每個 pair 分配一段 buffer（pair_idx 保證唯一）
        long long offset = (long long)pair_idx * 3 * (max_ref_len + 1);
        temp_h     = global_temp_buffer + offset;
        temp_f     = temp_h + (max_ref_len + 1);
        temp_len_f = temp_f + (max_ref_len + 1);
    } else {
        // 如果想用 shared memory，可以在這裡再依 warp 分段
        extern __shared__ int smem[];
        temp_h     = smem + warp_id_in_block * 3 * (N + 1);
        temp_f     = temp_h + (N + 1);
        temp_len_f = temp_f + (N + 1);
    }

    sw_warp_device(ref, alt, N, M,
                   match_score, mismatch_score,
                   gap_open, gap_extend,
                   best_score_out, sink_out,
                   temp_h, temp_f, temp_len_f,
                   trace, M + 1);
}

} // anonymous namespace

// ============================================================================
// Public API implementation
// ============================================================================

namespace biovoltron {

SmithWatermanCuda::SWResult
SmithWatermanCuda::cpu_traceback_int8(int N, int M,
                                      int2 sink,
                                      const std::vector<int8_t>& trace,
                                      int best_score)
{
    Cigar cigar;

    int i = sink.y; // column in alt
    int j = sink.x; // row in ref

    // tail soft-clip if sink 不在末端
    if (i < M) cigar.emplace_back(M - i, 'S');

    char state = 'M';
    int segment_len = 0;

    while (i > 0 && j > 0) {
        int8_t dir = trace[size_t(j) * (M + 1) + i];

        char new_state;
        int step_size;

        if (dir == 0) {
            new_state = 'M'; step_size = 1;
        } else if (dir > 0) {
            new_state = 'D'; step_size = dir;
        } else { 
            new_state = 'I'; step_size = -dir;
        }

        if (new_state == 'M') { i--; j--; }
        else if (new_state == 'D') { j -= step_size; }
        else { i -= step_size; }

        if (segment_len == 0) {
            state = new_state;
            segment_len = step_size;
        } else if (new_state == state) {
            segment_len += step_size;
        } else {
            cigar.emplace_back(segment_len, state);
            segment_len = step_size;
            state = new_state;
        }
    }

    if (segment_len > 0) cigar.emplace_back(segment_len, state);
    if (i > 0) cigar.emplace_back(i, 'S'); // head soft-clip

    cigar.reverse();

    return {j, cigar, best_score};
}

auto SmithWatermanCuda::align(std::string_view ref,
                              std::string_view alt,
                              Parameters params)
-> SWResult
{
    if (ref.empty() || alt.empty()) {
        return SWResult{};
    }

    const int N = static_cast<int>(ref.size());
    const int M = static_cast<int>(alt.size());

    // optional CPU fast-path: 等長且 mismatch 很少
    if (N == M) {
        int mism = 0;
        for (int k = 0; k < N && mism <= MAX_MISMATCHES; ++k) {
            if (ref[k] != alt[k]) ++mism;
        }
        if (mism <= MAX_MISMATCHES) {
            Cigar c;
            c.emplace_back(N, 'M');
            int score = (N - mism) * params.w_match +
                        mism * params.w_mismatch;
            return {0, c, score};
        }
    }

    // Device pointers
    char*  d_ref  = nullptr;
    char*  d_alt  = nullptr;
    int*   d_best = nullptr;
    int2*  d_sink = nullptr;
    int*   d_temp_h     = nullptr;
    int*   d_temp_f     = nullptr;
    int*   d_temp_len_f = nullptr;
    int8_t* d_trace     = nullptr;

    CUDA_CHECK(cudaMalloc(&d_ref,  N * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_alt,  M * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_best, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(int2)));
    CUDA_CHECK(cudaMalloc(&d_trace,
                          (size_t)(N + 1) * (size_t)(M + 1) * sizeof(int8_t)));

    // Decide whether to use shared memory for temp arrays
    int dev_id = 0;
    CUDA_CHECK(cudaGetDevice(&dev_id));
    int max_smem = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &max_smem, cudaDevAttrMaxSharedMemoryPerBlock, dev_id));
    size_t needed_bytes = 3 * (N + 1) * sizeof(int);

    bool   use_smem   = (needed_bytes <= static_cast<size_t>(max_smem));
    size_t kernel_smem = use_smem ? needed_bytes : 0;

    if (!use_smem) {
        CUDA_CHECK(cudaMalloc(&d_temp_h,     (N + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_temp_f,     (N + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_temp_len_f, (N + 1) * sizeof(int)));
    }

    CUDA_CHECK(cudaMemcpy(d_ref, ref.data(),
                          N * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_alt, alt.data(),
                          M * sizeof(char), cudaMemcpyHostToDevice));

    dim3 grid(1);
    dim3 block(WARP_SIZE);

    sw_warp_kernel<<<grid, block, kernel_smem>>>(
        d_ref, d_alt, N, M,
        params.w_match, params.w_mismatch,
        params.w_open,  params.w_extend,
        d_best, d_sink,
        d_temp_h, d_temp_f, d_temp_len_f,
        d_trace);

    CUDACHECKASYNC;

    int  best_score = 0;
    int2 sink       = make_int2(0, 0);
    std::vector<int8_t> h_trace((N + 1) * (M + 1));

    CUDA_CHECK(cudaMemcpy(&best_score, d_best,
                          sizeof(int),  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sink, d_sink,
                          sizeof(int2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_trace.data(), d_trace,
                          h_trace.size() * sizeof(int8_t),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_ref);
    cudaFree(d_alt);
    cudaFree(d_best);
    cudaFree(d_sink);
    cudaFree(d_trace);
    if (d_temp_h)     cudaFree(d_temp_h);
    if (d_temp_f)     cudaFree(d_temp_f);
    if (d_temp_len_f) cudaFree(d_temp_len_f);

    return cpu_traceback_int8(N, M, sink, h_trace, best_score);
}

auto SmithWatermanCuda::batch_align(const std::vector<std::string>& refs,
                                    const std::vector<std::string>& alts,
                                    Parameters params)
    -> std::vector<SWResult>
{
    size_t num_pairs = std::min(refs.size(), alts.size());
    if (num_pairs == 0) return {};

    std::vector<SWResult> results(num_pairs);

    // 1. Prepare host data
    std::vector<int>        h_ref_offsets(num_pairs);
    std::vector<int>        h_alt_offsets(num_pairs);
    std::vector<int>        h_ref_lengths(num_pairs);
    std::vector<int>        h_alt_lengths(num_pairs);
    std::vector<long long>  h_trace_offsets(num_pairs);

    std::string all_refs_concat;
    std::string all_alts_concat;

    size_t     total_ref_len    = 0;
    size_t     total_alt_len    = 0;
    long long  total_trace_size = 0;
    int        max_ref_len      = 0;

    for (size_t i = 0; i < num_pairs; ++i) {
        h_ref_offsets[i] = static_cast<int>(total_ref_len);
        h_alt_offsets[i] = static_cast<int>(total_alt_len);
        h_ref_lengths[i] = static_cast<int>(refs[i].size());
        h_alt_lengths[i] = static_cast<int>(alts[i].size());

        total_ref_len += refs[i].size();
        total_alt_len += alts[i].size();

        h_trace_offsets[i] = total_trace_size;
        total_trace_size +=
            static_cast<long long>(refs[i].size() + 1) *
            static_cast<long long>(alts[i].size() + 1);

        if (static_cast<int>(refs[i].size()) > max_ref_len) {
            max_ref_len = static_cast<int>(refs[i].size());
        }
    }

    all_refs_concat.reserve(total_ref_len);
    all_alts_concat.reserve(total_alt_len);
    for (size_t i = 0; i < num_pairs; ++i) {
        all_refs_concat += refs[i];
        all_alts_concat += alts[i];
    }

    // 2. Allocate device memory
    char *d_all_refs, *d_all_alts;
    int  *d_ref_offsets, *d_alt_offsets;
    int  *d_ref_lengths, *d_alt_lengths;
    int  *d_best_scores;
    int2 *d_sinks;
    int8_t    *d_all_traces;
    long long *d_trace_offsets;
    int       *d_global_temp = nullptr;

    CUDA_CHECK(cudaMalloc(&d_all_refs,   total_ref_len * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_all_alts,   total_alt_len * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_ref_offsets, num_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_alt_offsets, num_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ref_lengths, num_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_alt_lengths, num_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_scores, num_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sinks,       num_pairs * sizeof(int2)));
    CUDA_CHECK(cudaMalloc(&d_all_traces,
                          total_trace_size * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_trace_offsets,
                          num_pairs * sizeof(long long)));

    // Global temp buffer: 3*(max_ref_len+1) ints per pair
    size_t needed_smem = 3 * (max_ref_len + 1) * sizeof(int);
    size_t temp_buffer_size = num_pairs * needed_smem;
    CUDA_CHECK(cudaMalloc(&d_global_temp, temp_buffer_size));

    // 3. Copy data to device
    CUDA_CHECK(cudaMemcpy(d_all_refs, all_refs_concat.data(),
                          total_ref_len, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_all_alts, all_alts_concat.data(),
                          total_alt_len, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ref_offsets, h_ref_offsets.data(),
                          num_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_alt_offsets, h_alt_offsets.data(),
                          num_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ref_lengths, h_ref_lengths.data(),
                          num_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_alt_lengths, h_alt_lengths.data(),
                          num_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_trace_offsets, h_trace_offsets.data(),
                          num_pairs * sizeof(long long),
                          cudaMemcpyHostToDevice));

    // 4. Launch kernel: multi-warp-per-block
    int blocks = static_cast<int>(
        (num_pairs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    dim3 grid(blocks);
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);
    size_t kernel_smem = 0; // 使用 global_temp，不用 dynamic shared mem

    sw_batch_kernel<<<grid, block, kernel_smem>>>(
        d_all_refs, d_all_alts,
        d_ref_offsets, d_alt_offsets,
        d_ref_lengths, d_alt_lengths,
        static_cast<int>(num_pairs),
        params.w_match, params.w_mismatch,
        params.w_open,  params.w_extend,
        d_best_scores, d_sinks,
        d_all_traces, d_trace_offsets,
        d_global_temp, max_ref_len);

    CUDACHECKASYNC;

    // 5. Copy results back
    std::vector<int>    h_best_scores(num_pairs);
    std::vector<int2>   h_sinks(num_pairs);
    std::vector<int8_t> h_all_traces(total_trace_size);

    CUDA_CHECK(cudaMemcpy(h_best_scores.data(), d_best_scores,
                          num_pairs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sinks.data(), d_sinks,
                          num_pairs * sizeof(int2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_all_traces.data(), d_all_traces,
                          total_trace_size * sizeof(int8_t),
                          cudaMemcpyDeviceToHost));

    // 6. CPU traceback for each pair
    for (size_t i = 0; i < num_pairs; ++i) {
        int N = h_ref_lengths[i];
        int M = h_alt_lengths[i];
        int2 sink = h_sinks[i];
        int score = h_best_scores[i];
        long long start = h_trace_offsets[i];
        long long cells = (long long)(N + 1) * (M + 1);

        std::vector<int8_t> pair_trace(
            h_all_traces.begin() + start,
            h_all_traces.begin() + start + cells);

        results[i] = cpu_traceback_int8(N, M, sink, pair_trace, score);
    }

    // 7. Free device memory
    cudaFree(d_all_refs);   cudaFree(d_all_alts);
    cudaFree(d_ref_offsets); cudaFree(d_alt_offsets);
    cudaFree(d_ref_lengths); cudaFree(d_alt_lengths);
    cudaFree(d_best_scores); cudaFree(d_sinks);
    cudaFree(d_all_traces);  cudaFree(d_trace_offsets);
    if (d_global_temp) cudaFree(d_global_temp);

    return results;
}

} // namespace biovoltron
