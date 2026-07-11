list (APPEND vmecpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/basis_integrals.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/basis_integrals.h
  ${CMAKE_CURRENT_SOURCE_DIR}/vac2_mode_assembly.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/vac2_mode_assembly.h
  ${CMAKE_CURRENT_SOURCE_DIR}/vac2_solver.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/vac2_solver.h
  ${CMAKE_CURRENT_SOURCE_DIR}/vac2.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/vac2.h
)
set (vmecpp_sources "${vmecpp_sources}" PARENT_SCOPE)
