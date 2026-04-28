set(_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/dft_data.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ideal_mhd_model.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ideal_mhd_model.h
)

# fft_toroidal requires Intel MKL; only compiled when MKL was detected.
if(VMECPP_FFT_BACKEND STREQUAL "mkl")
  list(APPEND _sources
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_toroidal.h
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_toroidal.cc
  )
endif()

list(APPEND vmecpp_sources ${_sources})
set(vmecpp_sources "${vmecpp_sources}" PARENT_SCOPE)
