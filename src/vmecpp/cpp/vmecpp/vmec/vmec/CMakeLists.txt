list (APPEND vmecpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/anderson_acceleration.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/vmec.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/anderson_acceleration.h
  ${CMAKE_CURRENT_SOURCE_DIR}/vmec.h
)
set (vmecpp_sources "${vmecpp_sources}" PARENT_SCOPE)
