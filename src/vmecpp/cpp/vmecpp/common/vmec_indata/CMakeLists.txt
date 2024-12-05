list (APPEND vmecpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/boundary_from_json.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/boundary_from_json.h
  ${CMAKE_CURRENT_SOURCE_DIR}/vmec_indata.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/vmec_indata.h
)
set (vmecpp_sources "${vmecpp_sources}" PARENT_SCOPE)
