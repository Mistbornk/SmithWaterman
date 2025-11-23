cmake_minimum_required(VERSION 3.18)

project(biovoltron-test CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lboost_serialization")

file(GLOB_RECURSE SOURCE_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../Biovoltron/*.cpp"
)

add_executable(biovoltron-test ${SOURCE_FILES})

target_include_directories(biovoltron-test PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/.."
  "${CMAKE_CURRENT_LIST_DIR}/../xsimd"
)

target_link_libraries(biovoltron-test biovoltron)

target_compile_options(biovoltron-test PRIVATE -Wno-ignored-attributes)

target_compile_definitions(biovoltron-test
  PRIVATE DATA_PATH="${CMAKE_CURRENT_LIST_DIR}/data"
)

add_test(NAME biovoltron-run COMMAND biovoltron-test)
