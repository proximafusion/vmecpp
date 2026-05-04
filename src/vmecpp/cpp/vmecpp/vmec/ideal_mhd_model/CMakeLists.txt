list (APPEND vmecpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/dft_data.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dft_toroidal.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dft_toroidal.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fft_toroidal.h
  ${CMAKE_CURRENT_SOURCE_DIR}/fft_toroidal.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ideal_mhd_model.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ideal_mhd_model.h
)
set (vmecpp_sources "${vmecpp_sources}" PARENT_SCOPE)
