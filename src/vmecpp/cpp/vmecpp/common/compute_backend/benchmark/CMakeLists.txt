# Benchmark executable for compute backends

add_executable(backend_benchmark
  ${CMAKE_CURRENT_SOURCE_DIR}/backend_benchmark.cc
)

target_link_libraries(backend_benchmark PRIVATE vmecpp_core)

if(VMECPP_ENABLE_CUDA)
  target_link_libraries(backend_benchmark PRIVATE vmecpp_cuda CUDA::cudart)
endif()
