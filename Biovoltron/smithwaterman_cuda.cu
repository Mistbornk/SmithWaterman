#include <Biovoltron/smithwaterman_cuda.hpp>
#include <cuda_runtime.h>

#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <iostream>

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
    // -infinite
    const int NEG_INF = INT_MIN / 2;
    // thread id (1-indexed)
    const unsigned int lane = warp_tid();

    extern __shared__ int smem[];
    int* temp_h;
    int* temp_f;
    int* temp_len_f;

    if (global_temp_h == nullptr) {
        temp_h = smem;
        temp_f = smem + (N + 1);
        temp_len_f = smem + 2 * (N + 1);
    } else {
        temp_h = global_temp_h;
        temp_f = global_temp_f;
        temp_len_f = global_temp_len_f;
    }

    for (int j = lane; j <= N; j += WARP_SIZE) {
        temp_h[j]     = 0; 
        temp_f[j]     = NEG_INF; 
        temp_len_f[j] = 0;
    }

    __syncthreads();

    // H(i-1,j), H(i,j-1), H(i-1,j-1), current score
    int h_top  = 0, h_left = 0, h_diag = 0, h_val  = 0;  
    // E(i-1,j), E(i,j)
    int e_top  = NEG_INF, e_val = NEG_INF;
    // F(i,j-1), F(i,j)
    int f_left = NEG_INF, f_val = NEG_INF;  

    int len_e_top = 0, len_e_val = 0;
    int len_f_left = 0, len_f_val = 0;

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

        const unsigned int warp_block_width = (warp_block + WARP_SIZE >= M) ? (M - warp_block) : WARP_SIZE;
        // global thread index in DP column index (0..M), eq: thread 1 with colume index 0, 32, 64 .. 
        const unsigned int i = wi + warp_block; 

        h_top  = 0;         // H(0, j)
        h_diag = 0;         // H(0, j-1)
        e_top  = NEG_INF;   // E(0, j) 
        f_left = NEG_INF;   // F(i, 0)
        len_e_top = 0; len_f_left = 0;

        // The alt char for this thread
        const unsigned char s_i = (i <= (unsigned)M) ? static_cast<unsigned char>(alt[i - 1]) : 0;

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
        for (unsigned int block_diag = 2; block_diag <= warp_block_width + (unsigned)N; block_diag += WARP_SIZE)
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

            for (unsigned int diag = block_diag; diag < block_diag + WARP_SIZE; ++diag)
            {
                const unsigned int diag_len = dmin3(diag - 1, static_cast<unsigned>(WARP_SIZE), warp_block_width);
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

                    // if ref[j] == alt[i]
                    const int S_ij = (r_j == s_i) ? match_score : mismatch_score;
                    int score_diag = h_diag + S_ij;
                    
                    // E (Vertical)
                    int e_open = h_top + gap_open; int e_extend = e_top + gap_extend;
                    if (e_open > e_extend) { e_val = e_open; len_e_val = 1; } 
                    else { e_val = e_extend; len_e_val = len_e_top + 1; }

                    // F (Horizontal)
                    int f_open = h_left + gap_open; int f_extend = f_left + gap_extend;
                    if (f_open > f_extend) { f_val = f_open; len_f_val = 1; } 
                    else { f_val = f_extend; len_f_val = len_f_left + 1; }
                    
                    h_val = dmax3(score_diag, e_val, f_val);

                    // Traceback Logic (int8_t Length)
                    // Diag > Left (Insertion) > Up (Deletion)
                    int8_t trace_val = 0;
                    if (h_val == score_diag) {
                        trace_val = 0;
                    } else if (h_val == f_val) {
                        int l = (len_f_val > 127) ? 127 : len_f_val;
                        trace_val = (int8_t)(-l); // Negative for Left/Insertion
                    } else {
                        int l = (len_e_val > 127) ? 127 : len_e_val;
                        trace_val = (int8_t)(l);  // Positive for Up/Deletion
                    }

                    trace[size_t(j) * (M + 1) + i] = trace_val;

                    if (wi == WARP_SIZE) {
                        temp_h[j - 1] = h_val;
                        temp_f[j - 1] = f_val;
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

                    h_diag = h_left;
                    h_top  = h_val;
                    e_top  = e_val;
                    f_left = f_val;
                    len_e_top = len_e_val;
                }

                // warp shift
                r_j           = shfl_up(r_j, 1);
                h_left        = shfl_up(h_val, 1);
                f_left        = shfl_up(f_val, 1);
                len_f_left = shfl_up(len_f_val, 1);

                temp_h_cache    = shfl_down(temp_h_cache, 1);
                temp_f_cache    = shfl_down(temp_f_cache, 1);
                temp_len_f_cache = shfl_down(temp_len_f_cache, 1);
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

        if (other_score > best_score || (other_score == best_score && other_score > INT_MIN/2))
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
SmithWatermanCuda::SWResult cpu_traceback_int8(int N, int M, int2 sink, const std::vector<int8_t>& trace, int best_score) {
    Cigar cigar;
    int i = sink.y; 
    int j = sink.x; 
    
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

        if (cigar.size() == 0 && segment_len == 0) {
             state = new_state; segment_len = step_size;
        } else if (new_state == state) {
            segment_len += step_size;
        } else {
            cigar.emplace_back(segment_len, state);
            segment_len = step_size;
            state = new_state;
        }
    }
    
    if (segment_len > 0) cigar.emplace_back(segment_len, state);
    if (i > 0) cigar.emplace_back(i, 'S');
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

    // Device pointers
    char* d_ref = nullptr; char* d_alt = nullptr;
    int* d_best = nullptr; int2* d_sink = nullptr;
    int* d_temp_h = nullptr; int* d_temp_f = nullptr; int* d_temp_len_f = nullptr;
    int8_t* d_trace = nullptr; // int8 Trace

    // Malloc
    cudaMalloc(&d_ref,  N * sizeof(char));
    cudaMalloc(&d_alt,  M * sizeof(char));
    cudaMalloc(&d_best, sizeof(int));
    cudaMalloc(&d_sink, sizeof(int2));
    cudaMalloc(&d_trace, (size_t)(N + 1) * (size_t)(M + 1) * sizeof(int8_t));

    // See if it can use Shared Memory or not
    int dev_id = 0; cudaGetDevice(&dev_id);
    int max_smem = 0; cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock, dev_id);
    size_t needed_bytes = 3 * (N + 1) * sizeof(int);

    bool use_smem = (needed_bytes <= max_smem);
    size_t kernel_smem = use_smem ? needed_bytes : 0;
    int* k_h = nullptr; int* k_f = nullptr; int* k_lf = nullptr;
    
    if (!use_smem) {
        cudaMalloc(&d_temp_h, (N + 1) * sizeof(int));
        cudaMalloc(&d_temp_f, (N + 1) * sizeof(int));
        cudaMalloc(&d_temp_len_f, (N + 1) * sizeof(int));
        k_h = d_temp_h; k_f = d_temp_f; k_lf = d_temp_len_f;
        printf("Using Global Memory (Size: %zu bytes)\n", needed_bytes);
    }

    cudaMemcpy(d_ref, ref.data(), N * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alt, alt.data(), M * sizeof(char), cudaMemcpyHostToDevice);    

    // ---- launch kernel ----
    dim3 grid(1);
    dim3 block(WARP_SIZE);

    sw_warp_kernel<<<grid, block, kernel_smem>>>(
        d_ref, d_alt, N, M,
        params.w_match, params.w_mismatch, params.w_open, params.w_extend,
        d_best, d_sink, 
        k_h, k_f, k_lf,
        d_trace);

    CUDACHECKASYNC;

    int best_score = 0;
    int2 sink = make_int2(0,0);
    std::vector<int8_t> h_trace((N + 1) * (M + 1));

    cudaMemcpy(&best_score, d_best, sizeof(int),  cudaMemcpyDeviceToHost);
    cudaMemcpy(&sink,       d_sink, sizeof(int2), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_trace.data(), d_trace, h_trace.size() * sizeof(int8_t), cudaMemcpyDeviceToHost);

    // Free Memory
    cudaFree(d_ref); cudaFree(d_alt); cudaFree(d_best); cudaFree(d_sink); cudaFree(d_trace);
    if (d_temp_h) cudaFree(d_temp_h);
    if (d_temp_f) cudaFree(d_temp_f);
    if (d_temp_len_f) cudaFree(d_temp_len_f);

    return cpu_traceback_int8(N, M, sink, h_trace, best_score);
}

} // namespace biovoltron