list(APPEND vmecpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/geometry.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/geometry.h
  ${CMAKE_CURRENT_SOURCE_DIR}/geometry_c_api.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/geometry_c_api.h
  ${CMAKE_CURRENT_SOURCE_DIR}/vmec_geometry.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/vmec_geometry.h
)
set(vmecpp_sources "${vmecpp_sources}" PARENT_SCOPE)
