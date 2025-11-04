#include <Biovoltron/smithwaterman_cuda.hpp>
#include <cuda_runtime.h>
#include <stdexcept>

namespace biovoltron {

// --- CUDA kernal ---
__global__ void smith_waterman_kernel() {

}

// Interface
auto SmithWatermanCuda::align(std::string_view ref, std::string_view alt,
                              Parameters params)
  -> std::pair<int, Cigar> 
{
    
  return std::pair{0, Cigar(std::to_string(ref.size()) + 'M')};
}

} // namespace biovoltron