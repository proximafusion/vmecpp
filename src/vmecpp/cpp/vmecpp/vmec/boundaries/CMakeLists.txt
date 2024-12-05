list (APPEND vmecpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/boundaries.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/boundaries.h
  ${CMAKE_CURRENT_SOURCE_DIR}/guess_magnetic_axis.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/guess_magnetic_axis.h
)
set (vmecpp_sources "${vmecpp_sources}" PARENT_SCOPE)
