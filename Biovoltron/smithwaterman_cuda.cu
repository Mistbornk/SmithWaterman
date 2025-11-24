#include <Biovoltron/smithwaterman_cuda.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>

#define CUDACHECKASYNC \
{ cudaError_t err = cudaPeekAtLastError(); \
  if(err != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(err); \
  }}

namespace {
    constexpr int WARP_SIZE = 32;

    __device__ __forceinline__ int warp_tid() { return threadIdx.x & (WARP_SIZE - 1); }
    
    template <typename T> __device__ __forceinline__ T shfl_up(T v, int delta) { return __shfl_up_sync(0xffffffffu, v, delta); }
    template <typename T> __device__ __forceinline__ T shfl_down(T v, int delta) { return __shfl_down_sync(0xffffffffu, v, delta); }
    
    __device__ __forceinline__ int dmax(int a, int b) { return a > b ? a : b; }
    __device__ __forceinline__ int dmax3(int a, int b, int c) { return dmax(a, dmax(b, c)); }
    __device__ __forceinline__ int dmin(int a, int b) { return a < b ? a : b; }
    __device__ __forceinline__ int dmin3(int a, int b, int c) { return dmin(a, dmin(b, c)); }

    struct AlignmentResult {
        int score, j, i;
        __device__ AlignmentResult() : score(INT_MIN / 2), j(0), i(0) {}
        __device__ AlignmentResult(int s, int jj, int ii) : score(s), j(jj), i(ii) {}
    };

__device__ void gpu_traceback_int8(int N, int M, int2 sink, const int8_t* __restrict__ trace, 
                                       uint32_t* cigar_out, int* cigar_len_out) 
    {
        int i = sink.y; int j = sink.x; int k = 0;
        
        // 1. Tail Soft Clipping
        if (i < M) cigar_out[k++] = ((M - i) << 4) | 4; // Op 4 = 'S'

        int segment_len = 0; int state = 0; // 0='M'

        while (i > 0 && j > 0) {
            int8_t val = trace[size_t(j) * (M + 1) + i];
            int current_op, step;

            if (val == 0) { current_op = 0; step = 1; } // M
            else if (val > 0) { current_op = 2; step = val; } // D (Up)
            else { current_op = 1; step = -val; } // I (Left)

            if (k == 0 && segment_len == 0) { // First op
                 state = current_op; segment_len = step;
            } else if (current_op == state) {
                segment_len += step;
            } else {
                cigar_out[k++] = (segment_len << 4) | state;
                segment_len = step;
                state = current_op;
            }

            if (current_op == 0) { i--; j--; }
            else if (current_op == 2) { j -= step; }
            else { i -= step; }
        }
        
        if (segment_len > 0) cigar_out[k++] = (segment_len << 4) | state;
        
        // 2. Head Soft Clipping
        if (i > 0) cigar_out[k++] = (i << 4) | 4; // S

        *cigar_len_out = k;
    }
}


__global__
void sw_batch_kernel(
    const char* __restrict__ all_refs, const size_t* __restrict__ ref_offsets,
    const char* __restrict__ all_alts, const size_t* __restrict__ alt_offsets,
    int num_pairs,
    int match_score, int mismatch_score, int gap_open, int gap_extend,
    // Outputs
    int* out_scores,
    int2* out_sinks,
    // Huge Global Buffers (Pre-allocated)
    int* global_temp_buffer,     // Stores H, F, LenF for all pairs
    size_t* temp_offsets,        // Offsets into global_temp_buffer for each pair
    int8_t* global_trace_buffer, // Stores Trace for all pairs
    size_t* trace_offsets,       // Offsets into global_trace_buffer
    uint32_t* global_cigar_buffer, // Output CIGARs
    size_t* cigar_offsets,       // Offsets into cigar buffer
    int* out_cigar_counts        // Output cigar lengths
)
{
    // 1. 確定 Pair ID
    int pair_id = blockIdx.x;
    if (pair_id >= num_pairs) return;

    const unsigned int lane = warp_tid();
    const int NEG_INF = INT_MIN / 2;

    // 2. 取得該 Pair 的資料位置與長度
    size_t r_start = ref_offsets[pair_id];
    size_t r_end   = ref_offsets[pair_id+1];
    int N = (int)(r_end - r_start);

    size_t a_start = alt_offsets[pair_id];
    size_t a_end   = alt_offsets[pair_id+1];
    int M = (int)(a_end - a_start);

    const char* ref = all_refs + r_start;
    const char* alt = all_alts + a_start;

    // 3. 設定 Global Memory 指標 (每個 Pair 都有自己的一塊區域)
    // Temp Buffer Layout: [H (N+1) | F (N+1) | LenF (N+1)]
    size_t my_temp_offset = temp_offsets[pair_id];
    int* temp_h     = global_temp_buffer + my_temp_offset;
    int* temp_f     = temp_h + (N + 1);
    int* temp_len_f = temp_f + (N + 1);

    // Trace Buffer
    int8_t* trace = global_trace_buffer + trace_offsets[pair_id];

    // 4. 初始化 Temp Buffer
    for (int j = lane; j <= N; j += WARP_SIZE) {
        temp_h[j] = 0; temp_f[j] = NEG_INF; temp_len_f[j] = 0;
    }
    // 即使是 Global Memory，為了防止編譯器重排導致的 Race，同步一下
    __syncwarp(); 

    // -----------------------------------------------------------
    // 以下邏輯與 Single Kernel 完全相同
    // -----------------------------------------------------------
    int h_top = 0, h_left = 0, h_diag = 0, h_val = 0;
    int e_top = NEG_INF, e_val = NEG_INF;
    int f_left = NEG_INF, f_val = NEG_INF;
    int len_e_top = 0, len_e_val = 0;
    int len_f_left = 0, len_f_val = 0;
    AlignmentResult best;

    unsigned char r_j = 0;
    int temp_h_cache = 0, temp_f_cache = 0, temp_len_f_cache = 0;
    unsigned char reference_cache = 0;

    const unsigned int wi = lane + 1;

    for (int warp_block = 0; warp_block < M; warp_block += WARP_SIZE) {
        const unsigned int warp_block_width = (warp_block + WARP_SIZE >= M) ? (M - warp_block) : WARP_SIZE;
        const unsigned int i = wi + warp_block;

        h_top = 0; h_diag = 0; e_top = NEG_INF; f_left = NEG_INF;
        len_e_top = 0; len_f_left = 0;

        const unsigned char s_i = (i <= (unsigned)M) ? static_cast<unsigned char>(alt[i - 1]) : 0;

        for (unsigned int block_diag = 2; block_diag <= warp_block_width + (unsigned)N; block_diag += WARP_SIZE) {
            const unsigned int cache_row = (block_diag - 2) + lane;
            if (cache_row < (unsigned)N) {
                temp_h_cache = temp_h[cache_row];
                temp_f_cache = temp_f[cache_row];
                temp_len_f_cache = temp_len_f[cache_row];
                reference_cache = static_cast<unsigned char>(ref[cache_row]);
            } else {
                temp_h_cache = 0; temp_f_cache = NEG_INF; temp_len_f_cache = 0; reference_cache = 0;
            }

            for (unsigned int diag = block_diag; diag < block_diag + WARP_SIZE; ++diag) {
                const unsigned int diag_len = dmin3(diag - 1, (unsigned)WARP_SIZE, warp_block_width);
                const unsigned int j = diag - wi;

                if (wi <= diag_len && j <= (unsigned)N) {
                    if (wi == 1) {
                        r_j = reference_cache; h_left = temp_h_cache; f_left = temp_f_cache; len_f_left = temp_len_f_cache;
                    }
                    const int S_ij = (r_j == s_i) ? match_score : mismatch_score;
                    int score_diag = h_diag + S_ij;

                    int e_open = h_top + gap_open; int e_extend = e_top + gap_extend;
                    if (e_open > e_extend) { e_val = e_open; len_e_val = 1; } else { e_val = e_extend; len_e_val = len_e_top + 1; }

                    int f_open = h_left + gap_open; int f_extend = f_left + gap_extend;
                    if (f_open > f_extend) { f_val = f_open; len_f_val = 1; } else { f_val = f_extend; len_f_val = len_f_left + 1; }

                    h_val = dmax3(score_diag, e_val, f_val);

                    int8_t trace_val = 0;
                    if (h_val == score_diag) trace_val = 0;
                    else if (h_val == f_val) { int l = (len_f_val > 127) ? 127 : len_f_val; trace_val = (int8_t)(-l); }
                    else { int l = (len_e_val > 127) ? 127 : len_e_val; trace_val = (int8_t)(l); }

                    trace[size_t(j) * (M + 1) + i] = trace_val;

                    if (wi == WARP_SIZE) { temp_h[j - 1] = h_val; temp_f[j - 1] = f_val; temp_len_f[j - 1] = len_f_val; }

                    bool is_last_col = (i == (unsigned)M); bool is_last_row = (j == (unsigned)N);
                    if (is_last_col || is_last_row) { if (h_val >= best.score) best = AlignmentResult(h_val, j, i); }

                    h_diag = h_left; h_top = h_val; e_top = e_val; f_left = f_val; len_e_top = len_e_val;
                }
                r_j = shfl_up(r_j, 1); h_left = shfl_up(h_val, 1); f_left = shfl_up(f_val, 1); len_f_left = shfl_up(len_f_val, 1);
                temp_h_cache = shfl_down(temp_h_cache, 1); temp_f_cache = shfl_down(temp_f_cache, 1); temp_len_f_cache = shfl_down(temp_len_f_cache, 1); reference_cache = shfl_down(reference_cache, 1);
            }
        }
    }

    // Reduction
    int best_score = best.score; int best_j = best.j; int best_i = best.i;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        int other_score = __shfl_down_sync(0xffffffffu, best_score, offset);
        int other_j = __shfl_down_sync(0xffffffffu, best_j, offset);
        int other_i = __shfl_down_sync(0xffffffffu, best_i, offset);
        if (other_score > best_score || (other_score == best_score && other_score > INT_MIN/2)) {
            best_score = other_score; best_j = other_j; best_i = other_i;
        }
    }

    // Output Results & Traceback (Lane 0 only)
    if (lane == 0) {
        out_scores[pair_id] = best_score;
        
        // Traceback: 寫入專屬於這個 Pair 的 CIGAR 區域
        // 注意: 這裡直接用 GPU 做 traceback，避免回傳整個 trace matrix
        size_t my_cigar_offset = cigar_offsets[pair_id];
        int cigar_len = 0;
        
        // 起點 (Start Pos) 稍後由 cpu decode 時算出，這裡只負責寫出 CIGAR ops
        // 但是我們需要把 best_j (Sink Row) 傳回去，這樣才能推算 Offset
        out_sinks[pair_id] = make_int2(best_j, best_i); 

        gpu_traceback_int8(N, M, make_int2(best_j, best_i), trace, 
                           global_cigar_buffer + my_cigar_offset, 
                           &cigar_len);
                           
        out_cigar_counts[pair_id] = cigar_len;
    }
}

namespace biovoltron {

// Batch Alignment Implementation
auto
SmithWatermanCuda::batch_align(const std::vector<std::string>& refs,
                               const std::vector<std::string>& alts,
                               SmithWatermanCuda::Parameters params) 
-> std::vector<SmithWatermanCuda::SWResult>
{
    size_t num_pairs = refs.size();
    if (num_pairs == 0) return {};

    // 1. 準備 Host 端資料 (Flattening)
    std::vector<char> h_refs_flat;
    std::vector<char> h_alts_flat;
    std::vector<size_t> h_ref_offsets(num_pairs + 1, 0);
    std::vector<size_t> h_alt_offsets(num_pairs + 1, 0);
    
    // 用來計算記憶體需求
    std::vector<size_t> h_temp_offsets(num_pairs, 0);
    std::vector<size_t> h_trace_offsets(num_pairs, 0);
    std::vector<size_t> h_cigar_offsets(num_pairs, 0);
    
    size_t total_temp_ints = 0;
    size_t total_trace_bytes = 0;
    size_t total_cigar_uints = 0;

    for (size_t i = 0; i < num_pairs; ++i) {
        const auto& r = refs[i];
        const auto& a = alts[i];
        
        h_refs_flat.insert(h_refs_flat.end(), r.begin(), r.end());
        h_alts_flat.insert(h_alts_flat.end(), a.begin(), a.end());
        
        h_ref_offsets[i+1] = h_refs_flat.size();
        h_alt_offsets[i+1] = h_alts_flat.size();

        int N = r.size();
        int M = a.size();

        h_temp_offsets[i]  = total_temp_ints;
        h_trace_offsets[i] = total_trace_bytes;
        h_cigar_offsets[i] = total_cigar_uints;

        // Accumulate sizes
        // Temp: 3 arrays of size N+1 (H, F, LenF)
        total_temp_ints += 3 * (N + 1);
        // Trace: (N+1)*(M+1) int8
        total_trace_bytes += (size_t)(N + 1) * (M + 1);
        // Cigar: max size approx N+M
        total_cigar_uints += (size_t)(N + M + 2);
    }

    // 2. 分配 Device Memory
    char* d_refs = nullptr; cudaMalloc(&d_refs, h_refs_flat.size());
    char* d_alts = nullptr; cudaMalloc(&d_alts, h_alts_flat.size());
    size_t* d_ref_offsets = nullptr; cudaMalloc(&d_ref_offsets, h_ref_offsets.size() * sizeof(size_t));
    size_t* d_alt_offsets = nullptr; cudaMalloc(&d_alt_offsets, h_alt_offsets.size() * sizeof(size_t));

    int* d_scores = nullptr; cudaMalloc(&d_scores, num_pairs * sizeof(int));
    int2* d_sinks = nullptr; cudaMalloc(&d_sinks, num_pairs * sizeof(int2));
    int* d_cigar_cnts = nullptr; cudaMalloc(&d_cigar_cnts, num_pairs * sizeof(int));

    // Huge Buffers
    int* d_global_temp = nullptr; cudaMalloc(&d_global_temp, total_temp_ints * sizeof(int));
    int8_t* d_global_trace = nullptr; cudaMalloc(&d_global_trace, total_trace_bytes * sizeof(int8_t));
    uint32_t* d_global_cigar = nullptr; cudaMalloc(&d_global_cigar, total_cigar_uints * sizeof(uint32_t));

    // Offsets for Buffers
    size_t* d_temp_offsets = nullptr; cudaMalloc(&d_temp_offsets, num_pairs * sizeof(size_t));
    size_t* d_trace_offsets = nullptr; cudaMalloc(&d_trace_offsets, num_pairs * sizeof(size_t));
    size_t* d_cigar_offsets = nullptr; cudaMalloc(&d_cigar_offsets, num_pairs * sizeof(size_t));

    // 3. 拷貝資料 (一次性拷貝比多次小拷貝快很多)
    cudaMemcpy(d_refs, h_refs_flat.data(), h_refs_flat.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alts, h_alts_flat.data(), h_alts_flat.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_offsets, h_ref_offsets.data(), h_ref_offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alt_offsets, h_alt_offsets.data(), h_alt_offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_temp_offsets, h_temp_offsets.data(), h_temp_offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_trace_offsets, h_trace_offsets.data(), h_trace_offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cigar_offsets, h_cigar_offsets.data(), h_cigar_offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    // 4. 啟動 Batch Kernel
    dim3 grid(num_pairs); // 每個 Block 處理一個 Pair
    dim3 block(WARP_SIZE); // 每個 Block 一個 Warp (32 threads)

    sw_batch_kernel<<<grid, block>>>(
        d_refs, d_ref_offsets,
        d_alts, d_alt_offsets,
        num_pairs,
        params.w_match, params.w_mismatch, params.w_open, params.w_extend,
        d_scores, d_sinks,
        d_global_temp, d_temp_offsets,
        d_global_trace, d_trace_offsets,
        d_global_cigar, d_cigar_offsets, d_cigar_cnts
    );
    CUDACHECKASYNC;

    // 5. 拷貝結果回 CPU
    std::vector<int> h_scores(num_pairs);
    std::vector<int2> h_sinks(num_pairs);
    std::vector<int> h_cigar_cnts(num_pairs);
    std::vector<uint32_t> h_all_cigars(total_cigar_uints);

    cudaMemcpy(h_scores.data(), d_scores, num_pairs * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sinks.data(), d_sinks, num_pairs * sizeof(int2), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cigar_cnts.data(), d_cigar_cnts, num_pairs * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_all_cigars.data(), d_global_cigar, total_cigar_uints * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // 6. 組裝 SWResult (CPU 解碼 CIGAR)
    std::vector<SWResult> results(num_pairs);
    const char op_map[] = {'M', 'I', 'D', 'N', 'S'};

    for (size_t i = 0; i < num_pairs; ++i) {
        Cigar cigar;
        size_t start_idx = h_cigar_offsets[i];
        int count = h_cigar_cnts[i];
        
        // 模擬回溯過程來計算 Offset (Start Position)
        // GPU 雖然產生了 CIGAR，但沒有算出最終的 start j。
        // 我們可以透過簡單地遍歷 CIGAR 動作來推算。
        int current_j = h_sinks[i].x; // End position from Kernel
        
        for (int k = 0; k < count; ++k) {
            uint32_t val = h_all_cigars[start_idx + k];
            int len = val >> 4;
            int op_idx = val & 0xF;
            char op = op_map[op_idx];
            cigar.emplace_back(len, op);
            
            // 反向推算 Start Position
            if (op == 'M' || op == 'D') {
                current_j -= len;
            }
        }
        cigar.reverse();
        
        results[i] = {current_j, cigar, h_scores[i]}; 
    }

    // 7. 釋放記憶體 (實際應用中建議用 RAII wrapper 或 Pool)
    cudaFree(d_refs); cudaFree(d_alts); cudaFree(d_ref_offsets); cudaFree(d_alt_offsets);
    cudaFree(d_scores); cudaFree(d_sinks); cudaFree(d_cigar_cnts);
    cudaFree(d_global_temp); cudaFree(d_global_trace); cudaFree(d_global_cigar);
    cudaFree(d_temp_offsets); cudaFree(d_trace_offsets); cudaFree(d_cigar_offsets);

    return results;
}

} // namespace biovoltron