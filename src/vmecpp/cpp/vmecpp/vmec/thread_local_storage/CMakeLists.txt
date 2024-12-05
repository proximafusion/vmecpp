list (APPEND vmecpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/thread_local_storage.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/thread_local_storage.h
)
set (vmecpp_sources "${vmecpp_sources}" PARENT_SCOPE)
