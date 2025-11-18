#include <Biovoltron/smithwaterman_cuda.hpp>
#include <cuda_runtime.h>

#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#define CUDACHECKASYNC \
{ cudaError_t err = cudaPeekAtLastError(); \
  if(err != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(err); \
  }}

// -------------------------------------------------------------------------------------------------
// internal CUDA utilities (warp helpers)
// -------------------------------------------------------------------------------------------------
namespace {

constexpr int WARP_SIZE = 32;

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

// -------------------------------------------------------------------------------------------------
// Result of one alignment inside the warp
// (score + sink position). Minimal版，類似 nvbio::alignment_result<>.
// -------------------------------------------------------------------------------------------------
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

// -------------------------------------------------------------------------------------------------
// Warp-parallel Smith-Waterman scoring kernel (no traceback).
// 參考 nvbio/alignment/sw/sw_warp_inl.h 的寫法但簡化為：
//   * 只做 LOCAL alignment
//   * 線性 gap penalty（之後可以改 affine 狀態機）
//   * 一個 warp 處理一個 alignment
// ref : 垂直 (長度 N)
// alt : 水平 (長度 M)
// -------------------------------------------------------------------------------------------------
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
                    int2* __restrict__ sink_out)
{
    const unsigned int lane = warp_tid();
	const int NEG_INF = INT_MIN / 2;

    // 動態 shared memory: column[0..N]
    extern __shared__ int temp[];

    for (int j = lane; j <= N; j += WARP_SIZE)
        temp[j] = 0;

    __syncthreads();

    int h_top  = 0;  // H(i-1,j)
    int h_left = 0;  // H(i,j-1)
    int h_diag = 0;  // H(i-1,j-1)
    int h_val  = 0;

    int e_top  = 0;  // E(i-1,j)
    int e_val  = 0;  // E(i,j)

    int f_left = 0;  // F(i,j-1)
    int f_val  = 0;  // F(i,j)

    AlignmentResult best;

    unsigned char r_j = 0;
    int           temp_cache      = 0;
    unsigned char reference_cache = 0;

    const unsigned int wi = lane + 1; // DP 中當前 warp stripe 的 column index (1..WARP_SIZE)

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
        <------------ Stripe 1 (Width 32) ------------>
        Col 1        Col 2        Col 3     ...  Col 32
        +---+---+---+---+---+---+---+---+---+---+---+
    j=1 |   |   |   |   |   |   |   |   |   |   |   |
        +---+---+---+---+---+---+---+---+---+---+---+
    j=2 |   |   |   |   |   |   |   |   |   |   |   |
        +---+---+---+---+---+---+---+---+---+---+---+
        ...
        +---+---+---+---+---+---+---+---+---+---+---+
        ^   ^   ^                                   ^
        |   |   |                                   |
Thread  1   2   3       ...                 Thread 32
(wi =   1   2   3       ...                     wi=32)
    */
    // Through query (alt) do  WARP_SIZE stripe
    for (int warp_block = 0; warp_block < M; warp_block += WARP_SIZE)
    {

        const unsigned int warp_block_width = (warp_block + WARP_SIZE >= M) ? (M - warp_block) : WARP_SIZE;
        // global thread index in DP column index (0..M), eq: thread 1 with colume index 0, 32, 64 .. 
        const unsigned int i = wi + warp_block; 

        h_top  = 0;         // H(0, j)
        h_diag = 0;         // H(0, j-1)
        e_top  = 0;         // E(0, j) 
        f_left = 0;         // F(i, 0)

        // The alt char for this thread
        const unsigned char s_i = (i <= (unsigned)M) ? static_cast<unsigned char>(alt[i - 1]) : 0;

        // For the stripe's anti-diagonals
        /*
        block_diag = 34 (example):

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
        for (unsigned int block_diag = 2; block_diag <= warp_block_width + (unsigned)N; block_diag += WARP_SIZE)
        {
            // Every WARP_SIZE anti-diagonals reload cache
            const unsigned int cache_row = (block_diag - 2) + lane;
            temp_cache      = (cache_row < (unsigned)N) ? temp[cache_row]          					 : 0; // left value
            reference_cache = (cache_row < (unsigned)N) ? static_cast<unsigned char>(ref[cache_row]) : 0; // ref char

            for (unsigned int diag = block_diag; diag < block_diag + WARP_SIZE; ++diag)
            {
                const unsigned int diag_len = dmin3(diag - 1, (unsigned)WARP_SIZE, warp_block_width);
                const unsigned int j = diag - wi; // row index (1..N)

                if (wi <= diag_len && j <= (unsigned)N)
                {
                    if (wi == 1)
                    {
                        // thread 1 read reference char and left cell value
                        // and for lane 0 (first warp) will read new reference char in his row and left value=0
                        r_j   = reference_cache;
                        h_left = temp_cache;
                    }
                    // if ref[j] == alt[i]
                    const int S_ij = (r_j == s_i) ? match_score : mismatch_score;
					// --- affine E vertical gaps ---
                    e_val = dmax(h_top + gap_open, e_top + gap_extend);
                    // --- affine F horizontal gaps ---
                    f_val = dmax(h_left + gap_open, f_left + gap_extend);

                    // compute H(i,j)
					h_val = h_diag + S_ij;
					h_val = dmax3(h_val, e_val, f_val);

                    if (wi == WARP_SIZE)
                        temp[j - 1] = h_val;
					
					if (h_val >= best.score)
						best = AlignmentResult(h_val, j, i);

                    h_diag = h_left;
                    h_top  = h_val;
                    e_top  = e_val;
                    f_left = f_val;
                }

                // warp shift
                r_j           = shfl_up(r_j, 1);
                h_left        = shfl_up(h_val, 1);
                f_left        = shfl_up(f_val, 1);
                temp_cache      = shfl_down(temp_cache, 1);
                reference_cache = shfl_down(reference_cache, 1);
            }
        }
    }

    // warp reduction to find best score and sink(i, j)
    int best_score = best.score;
    int best_j     = best.j;
    int best_i     = best.i;

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        int other_score = __shfl_down_sync(0xffffffffu, best_score, offset);
        int other_j     = __shfl_down_sync(0xffffffffu, best_j,     offset);
        int other_i     = __shfl_down_sync(0xffffffffu, best_i,     offset);

        if (other_score > best_score)
        {
            best_score = other_score;
            best_j     = other_j;
            best_i     = other_i;
        }
    }

    if (lane == 0)
    {
        *best_score_out = best_score;
        sink_out->x     = best_j;  // row in ref
        sink_out->y     = best_i;  // column in alt
    }
}

} // anonymous namespace

namespace biovoltron {

// -------------------------------------------------------------------------------------------------
// Public API
// -------------------------------------------------------------------------------------------------

auto SmithWatermanCuda::align(std::string_view ref,
                              std::string_view alt,
                              Parameters params)
-> SWResult
{
    if (ref.empty() || alt.empty()) {
        return SWResult{};
    }
	cudaFree(0);

    const int N = static_cast<int>(ref.size());
    const int M = static_cast<int>(alt.size());

    // ---- device memory ----
    char* d_ref  = nullptr;
    char* d_alt  = nullptr;
    int*  d_best = nullptr;
    int2* d_sink = nullptr;

    cudaMalloc(&d_ref,  N * sizeof(char));
    cudaMalloc(&d_alt,  M * sizeof(char));
    cudaMalloc(&d_best, sizeof(int));
    cudaMalloc(&d_sink, sizeof(int2));

    cudaMemcpy(d_ref, ref.data(), N * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alt, alt.data(), M * sizeof(char), cudaMemcpyHostToDevice);

    // ---- launch kernel ----
    dim3 grid(1);
    dim3 block(WARP_SIZE);

    const size_t shared_bytes = (N + 1) * sizeof(int);
    sw_warp_kernel<<<grid, block, shared_bytes>>>(
        d_ref,
        d_alt,
        N,
        M,
        params.w_match,
        params.w_mismatch,
        params.w_open,
        params.w_extend,
        d_best,
        d_sink);

    CUDACHECKASYNC;

    int  best_score = 0;
    int2 sink       = make_int2(0, 0);

    cudaMemcpy(&best_score, d_best, sizeof(int),  cudaMemcpyDeviceToHost);
    cudaMemcpy(&sink,       d_sink, sizeof(int2), cudaMemcpyDeviceToHost);

    cudaFree(d_ref);
    cudaFree(d_alt);
    cudaFree(d_best);
    cudaFree(d_sink);


	printf("[SW CUDA DEBUG] best_score = %d\n", best_score);
	printf("[SW CUDA DEBUG] sink (j,i) = (%d, %d)\n", sink.x, sink.y);
	fflush(stdout);
    // TODO:
    //  0. Fixed aligmnet bug: 還沒找到為什麼有時候分數跟 baseline 不一樣
    //  1. 根據 sink (最佳 j,i) 做 traceback，計算真正的 offset 與 CIGAR, return SWResult, 把 best_score / offset / cigar 填進去
    //  2. 提升 GPU 利用率, 新增 batch align: 一次可以同時跑許多 align pair (ref vs. alt), 目前只用到 warp size 個 block 做一個 align pair
    //
    return SWResult{};
}

} // namespace biovoltron
