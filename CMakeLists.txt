cmake_minimum_required(VERSION 3.18)

project(sw_cuda LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

add_library(biovoltron STATIC
    Biovoltron/smithwaterman_cuda.cu
)

target_link_libraries(biovoltron
    PUBLIC CUDA::cudart
)

target_include_directories(biovoltron PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(test)
