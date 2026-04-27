# 1 "/home/runner/work/vmecpp/vmecpp/build/_deps/indata2json-src/src/vmec_input.f"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/home/runner/work/vmecpp/vmecpp/build/_deps/indata2json-src/src/vmec_input.f"
!-------------------------------------------------------------------------------
!  The @header, @table_section, @table_subsection, @item and @end_table commands
!  are custom defined commands in Doxygen.in. They are defined under ALIASES.
!  For the page created here, the 80 column limit is exceeded. Arguments of
!  aliases are separated by ','. If you intended ',' to be a string you must use
!  an escaped comma '\,'.
!
!>  @page vmec_namelist_sec Namelist indata definition
!>
!>  @tableofcontents
!>  @section vmec_namelist_intro_sec Introduction
!>  This page documents the contents of a namelist input file. VMEC namelist
!>  variables are defined in the @fixed_width{indata} common block.
!>
!>  @section vmec_namelist_var_sec Namelist Variables
!>  @header{Input variable, Description, Code Reference}
!>
!>  @table_section{vmec_control_param_sec, Control Parameters.}
!>     @item{omp_num_threads,        Number of openmp threads to use. @b DEPRECATED @b,      vmec_input::omp_num_threads}
!>     @item{gamma,                  Adiabatic factor for the pressure perturbation.,        vmec_input::gamma}
!>     @item{niter,                  Maximum number of iterations to run. @b DEPRECATED @b
!>                                   Use @ref vmec_input::niter_array instead.,              vmec_input::niter}
!>     @item{niter_array,            Array of iterations for the multigrid runs.,            vmec_input::niter_array}
!>     @item{time_slice,             Time index value to label the output files.,            vmec_input::time_slice}
!>     @item{nstep,                  Number of iterations between screen output.,            vmec_input::nstep}
!>     @item{nvacskip,               Number of iterations between vacuum responses.,         vmec_input::nvacskip}
!>     @item{delt,                   Time step value for minimization.,                      vmec_input::delt}
!>     @item{ftol,                   Force tolarance for minimization. @b DEPRECATED @b
!>                                   Use @ref vmec_input::ftol_array instead.,               vmec_input::ftol}
!>     @item{ftol_array,             Array of force tolarances for the multigrid runs.,      vmec_input::ftol_array}
!>     @item{tcon0,                  Weight factor for constraint force.,                    vmec_input::tcon0}
!>     @table_subsection{vmec_precon_control_param_sec, Precondicioner control parameters.}
!>        @item{precon_type,         Type of preconditioner.,                                vmec_input::precon_type}
!>        @item{prec2d_threshold,    Force tolarance where the preconditioner is turned on., vmec_input::prec2d_threshold}
!>     @table_subsection{vmec_giveup_control_param_sec, Early termination control parameters.}
!>        @item{lgiveup,             Stop early if convergence is poor.,                     vmec_input::lgiveup}
!>        @item{fgiveup,             Theshold tolarance for early exit.,                     vmec_input::lgiveup}
!>     @table_subsection{vmec_continue_control_param_sec, Continue control parameters.}
!>        @item{max_main_iterations, Number of additional iterations to use.,                vmec_input::lmoreiter}
!>  @end_table
!>
!>  @table_section{vmec_control_flags_sec, Control flags.}
!>     @item{lmove_axis,  Allow movement of the magnetic axis.,                       vmec_input::lmove_axis}
!>     @item{lmac,        UNKNOWN,                                                    vmec_input::lmac}
!>     @item{lforbal,     Use non-variational forces to ensure force balance.,        vmec_input::lforbal}
!>     @item{lasym,       Allow non-stellarator symmetric terms.,                     vmec_input::lasym}
!>     @item{lrfp,        Run in RFP mode. When true the iota profile becomes
!>                        the q profile.,                                             vmec_input::lrfp}
!>     @item{loldout,     Use text output when writting wout files.
!>                        @b DEPRECATED @b,                                           vmec_input::loldout}
!>     @item{ldiagno,     Get output for diagno 1.0 and 1.5.,                         vmec_input::ldiagno}
!>     @item{lbsubs,      Capture current sheets.,                                    vmec_input::lbsubs}
!>     @item{lfull3d1out, Write out full threed1 file if force tolarance is not met., vmec_input::lfull3d1out}
!>     @item{lwouttxt,    Write out text based woutfile. Note text based wout files
!>                        are deprecated.,                                            vmec_input::lwouttxt}
!>     @item{lnyquist,    Write out the full nyquest spectrum when true. Write out a
!>                        truncated spectrum to match mpol and ntor used for the
!>                        equilibrium.,                                               vmec_input::lnyquist}
!>  @end_table
!>
!>  @table_section{vmec_radial_parameters_sec, Radial parameters.}
!>     @item{nsin,     Number of radial flux surfaces. @b DEPRECATED @b
!>                     Use @ref vmec_input::ns_array instead.,             vmec_input::nsin}
!>     @item{ns_array, Number of radial flux surfaces for each grid size., vmec_input::ns_array}
!>     @item{aphi,     Radial surface redistribution factors.,             vmec_input::aphi}
!>  @end_table
!>
!>  @table_section{vmec_fourier_sizes_sec, Fourier sizes.}
!>     @item{mpol, Total number of poloidal modes.,      vmec_input::mpol}
!>     @item{ntor, Largest value of the toroidal modes., vmec_input::ntor}
!>     @item{nfp,  Number of field periods.,             vmec_input::nfp}
!>  @end_table
!>
!>  @table_section{vmec_realspace_sizes_sec, Real space grid sizes.}
!>     @item{ntheta, Number of poloidal grid points., vmec_input::ntheta}
!>     @item{nzeta,  Number of toroidal grid points., vmec_input::nzeta}
!>  @end_table
!>
!>  @table_section{vmec_extcur_param_sec, External current parameters.}
!>     @item{lfreeb,     Use external fields to determine the boundary shape., vmec_input::lfreeb}
!>     @item{extcur,     Array of external field coil currents.,               vmec_input::extcur}
!>     @item{mgrid_file, Path to the mgrid file.,                              vmec_input::mgrid_file}
!>  @end_table
!>
!>  @table_section{vmec_profile_param_sec, Profile Parameters.}
!>     @table_subsection{vmec_current_profile_param_sec, Current profile parameters.}
!>        @item{curtor,     Total toroidal current.,                                                                                vmec_input::curtor}
!>        @item{ncurr,      Control to use current profile.,                                                                        vmec_input::ncurr}
!>        @item{pcurr_type, Current profile type.
!>                          -# sum_cossq_s       Sum of cos^2 waves. I'
!>                          -# sum_cossq_sqrts   Sum of cos^2 with respect to sqrt(s). I'
!>                          -# sum_cossq_s_free  Sum of cos^2 waves up to 7. Free position and width. I'
!>                          -# gauss_trunc       Truncated gaussian. I'
!>                          -# two_power         ac(0)*(1 - s^ac(1))^ac(2).
!>                          -# two_power_gs      ac(0)*((1 - s^ac(1))^ac(2)*(1 + Sum_{i}ac(i)*exp(-((s - ac(i + 1))/ac(i + 2))^2)).
!>                          -# sum_atan          Sum of arctangents.
!>                          -# power_series_I    Power series.
!>                          -# power_series      Power series. I' (Default)
!>                          -# Akima_spline_Ip   Akima splines. I'
!>                          -# Akima_spline_I    Akima splines.
!>                          -# cublic_splines_Ip Cublic splines. I'
!>                          -# cublic_splines_I  Cublic splines. I
!>                          -# pedestal          Pedestal profile.
!>                          -# rational          Rational function (ratio of polynomials).
!>                          -# line_segment_Ip   Line segments. I'
!>                          -# line_segment_I    Line segments.
!>                          For more information @see pcurr,                                                                        vmec_input::pcurr_type}
!>        @item{ac,         Parameterized current profile coefficents.,                                                             vmec_input::ac}
!>        @item{ac_aux_s,   Radial knot position for the current profile splines.,                                                  vmec_input::ac_aux_s}
!>        @item{ac_aux_f,   Knot amplitudes for the current profile splines.,                                                       vmec_input::ac_aux_f}
!>        @item{bloat,      Expansion factor for the current profile.,                                                              vmec_input::bloat}
!>     @table_subsection{vmec_iota_profile_param_sec, Rotational transform profile parameters.}
!>        @item{piota_type, Rotational transform profile type.
!>                          -# sum_atan       Sum of arctangents.
!>                          -# Akima_spline   Akima_splines.
!>                          -# cublic_splines Cublic splines.
!>                          -# rational       Rational function (ratio of polynomials).
!>                          -# nice_quadratic Quadratic with rearranged coefficients.
!>                          -# line_segment   Line segments.
!>                          -# power_series   Power series. (Default)
!>                          For more information @see piota,                                                                        vmec_input::piota_type}
!>        @item{ai,         Parameterized rotational transform profile coefficents.,                                                vmec_input::ai}
!>        @item{ai_aux_s,   Radial knot position for the rotational transform profile splines.,                                     vmec_input::ai_aux_s}
!>        @item{ai_aux_f,   Knot amplitudes for the rotational transform profile splines.,                                          vmec_input::ai_aux_f}
!>     @table_subsection{vmec_pres_profile_param_sec, Pressure profile parameters.}
!>        @item{pres_scale, Scaling factor for the pressure profile.,                                                               vmec_input::pres_scale}
!>        @item{pmass_type, Pressure profile type.
!>                          -# gauss_trunc    Truncated Gaussian.
!>                          -# two_power      ac(0)*(1 - s^ac(1))^ac(2).
!>                          -# two_power_gs   ac(0)*((1 - s^ac(1))^ac(2)*(1 + Sum_{i}ac(i)*exp(-((s - ac(i + 1))/ac(i + 2))^2)).
!>                          -# two_Lorentz    Two lorentz type functions mapped form s=0 to s=1.
!>                          -# Akima_spline   Akima_splines.
!>                          -# cublic_splines Cublic splines.
!>                          -# pedestal       Pedestal profile.
!>                          -# rational       Rational function (ratio of polynomials).
!>                          -# line_segment   Line segments.
!>                          -# power_series   Power series. (Default)
!>                          For more information @see pmass,                                                                        vmec_input::pmass_type}
!>        @item{am,         Parameterized rotational pressure coefficents.,                                                         vmec_input::am}
!>        @item{am_aux_s,   Radial knot position for the pressure profile splines.,                                                 vmec_input::am_aux_s}
!>        @item{am_aux_f,   Knot amplitudes for the pressure profile splines.,                                                      vmec_input::am_aux_f}
!>        @item{spres_ped,  Value of s beyond which pressure profile is flat.,                                                      vmec_input::spres_ped}
!>  @end_table
!>
!>  @table_section{vmec_anisotropy_parameters_sec, Anisotropy parameters.}
!>     @item{bcrit, Hot particle energy deposition value for |B|. ANIMEC only., vmec_input::bcrit}
!>     @item{at,    TPERP/TPAR ANIMEC only.,                                    vmec_input::at}
!>     @item{ah,    PHOT/PTHERMAL ANIMEC only.,                                 vmec_input::ah}
!>  @end_table
!>
!>  @table_section{vmec_edge_parameters_sec, Edge Boundary parameters.}
!>     @item{phiedge,      Total toroidal flux.,                            vmec_input::phiedge}
!>     @item{rbc,          R boundary coefficients stellarator symmetric.,  vmec_input::rbc}
!>     @item{rbs,          R boundary coefficients stellarator asymmetric., vmec_input::rbs}
!>     @item{zbs,          Z boundary coefficients stellarator symmetric.,  vmec_input::rbs}
!>     @item{zbc,          Z boundary coefficients stellarator asymmetric., vmec_input::rbc}
!>     @item{mfilter_fbdy, Filter poloidal boundary terms.,                 vmec_input::mfilter_fbdy}
!>     @item{nfilter_fbdy, Filter toroidal boundary terms.,                 vmec_input::nfilter_fbdy}
!>  @end_table
!>
!>  @table_section{vmec_axis_parameters_sec, Inital magnetic axis guess.}
!>     @item{raxis_cc, R axis coefficients stellarator symmetric.,                 vmec_input::raxis_cc}
!>     @item{raxis_cs, R axis coefficients stellarator asymmetric.,                vmec_input::raxis_cs}
!>     @item{zaxis_cs, R axis coefficients stellarator symmetric.,                 vmec_input::zaxis_cs}
!>     @item{zaxis_cc, R axis coefficients stellarator asymmetric.,                vmec_input::zaxis_cc}
!>     @item{raxis,    R axis coefficients stellarator symmetric. @b DEPRECATED @b
!>                     Use @ref vmec_input::raxis_cc instead.,                     vmec_input::raxis}
!>     @item{zaxis,    R axis coefficients stellarator symmetric. @b DEPRECATED @b
!>                     Use @ref vmec_input::zaxis_cs instead.,                     vmec_input::zaxis}
!>  @end_table
!>
!>  @section vmec_namelist_example_sec Example File.
!>  @code
!>  &indata
!>    delt = 1.0,
!>    tcon0 = 2.0,
!>    nfp = 1,
!>    ns_array = 15,
!>    ftol_array = 1.0E-20,
!>    niter = 25000,
!>    nstep = 200,
!>    ntor = 0,
!>    mpol = 2,
!>    lasym = F,
!>    lfreeb = F,
!>
!>    gamma = 0.0,
!>    phiedge = -0.05,
!>    bloat = 1.0,
!>
!>    raxis_cc(0) = 1.0,
!>    zaxis_cs(0) = 0.0,
!>    rbc(0,0) = 1.0,
!>    rbc(0,1) = 0.25,
!>    zbs(0,0) = 0.0,
!>    zbs(0,1) = 0.25,
!>
!>    ncurr = 1,
!>    curtor = 40000.0,
!>    pcurr_type = 'line_segment_I',
!>    ac_aux_s = 0.0, 1.0,
!>    ac_aux_f = 1.0, 0.0,
!>
!>    spres_ped = 1.0,
!>    pres_scale = 400.0,
!>    pmass_type = 'line_segment',
!>    am_aux_s = 0.0, 1.0,
!>    am_aux_f = 1.0, 0.0,
!>  /
!>  @end code
!>
!>  @section vmec_namelist_prog_ref_sec Programmers Reference
!>  Reference material for the coding to implement this namelist is found in the
!>  @ref vmec_input module.
!-------------------------------------------------------------------------------
!*******************************************************************************
!>  @file vmec_input.f
!>  @brief Contains the module @ref vmec_input.
!
!  Note separating the Doxygen comment block here so detailed decription is
!  found in the Module not the file.
!
!>  This file contains all the variables for a VMEC namelist input file. The
!>  module contained within does not represent an object instance. Instead all
!>  variables are contained in a global context. This is required due to
!>  limitations of FORTRAN 95 and namelist inputs.
!>
!>  @ref vmec_namelist_sec "VMEC namelist indata definition"
!*******************************************************************************
      MODULE vmec_input
      USE vparams, ONLY: rprec, dp, mpol1d, ntord, ndatafmax
      USE vsvd0

      IMPLICIT NONE

!*******************************************************************************
!  VMEC input module parameters.
!*******************************************************************************
!>  Default value for the number of poloidal modes.
      INTEGER, PARAMETER :: mpol_default = 6
!>  Default value for the number of toroidal modes.
      INTEGER, PARAMETER :: ntor_default = 0
!>  Default value for the number of flux surfaces.
      INTEGER, PARAMETER :: ns_default   = 31
!>  Maximum number of grids.
      INTEGER, PARAMETER :: max_grids = 100
!>  Short name length.
      INTEGER, PARAMETER :: short_name = 20
!>  Long name length.
      INTEGER, PARAMETER :: long_name = 200
!>  Maximum number of spline knots.
      INTEGER, PARAMETER :: max_profile = 20
!> maximum number of iter2 values at which to dump debugging output
      integer, parameter :: num_iter2_to_dump = 20

!*******************************************************************************
!  VMEC input module variables.
!*******************************************************************************
!  Control Parameters
!>  @deprecated
!>  Number of OpenMP threads to use.
      INTEGER                            :: omp_num_threads
!>  Adiabatic factor for the pressure perturbation.
      REAL (rprec)                       :: gamma
!>  @deprecated
!>  Maximum number of iterations to run.
      INTEGER                            :: niter
!>  Array of iterations for the multigrid runs.
      INTEGER, DIMENSION(max_grids)      :: niter_array
!>  Time index value to label the output files.
      REAL (rprec)                       :: time_slice
!>  Number of iterations between screen output.
      INTEGER                            :: nstep
!>  Number of iterations between vacuum responses.
      INTEGER                            :: nvacskip
!>  Time step value for minimization.
      REAL (rprec)                       :: delt
!>  @deprecated
!>  Force tolarance for minimization.
      REAL (rprec)                       :: ftol
!>  Array of force tolarances for the multigrid runs.
      REAL (rprec), DIMENSION(max_grids) :: ftol_array
!>  Weight factor for constraint force.
      REAL (rprec)                       :: tcon0

!  Precondicioner control parameters.
!>  Type of preconditioner.
      CHARACTER(len=short_name) :: precon_type
!>  Force tolarance where the preconditioner is turned on.
      REAL (rprec)              :: prec2d_threshold

!  Early termination control parameters.
!>  Stop early if convergence is poor.
      LOGICAL      :: lgiveup
!>  Theshold tolarance for early exit.
      REAL (rprec) :: fgiveup
!  Continue control parameters.
!>  Add more iterations if force residules are not met.
      LOGICAL      :: lmoreiter
!>  Number of additional iterations to use.
      INTEGER      :: max_main_iterations

!  Control flags.
!>  Allow movement of the magnetic axis.
      LOGICAL :: lmove_axis
!>  UNKNOWN
      LOGICAL :: lmac
!>  Use non-variational forces to ensure force balance.
      LOGICAL :: lforbal
!>  Allow non-stellarator symmetric terms.
      LOGICAL :: lasym
!>  Run in RFP mode.
      LOGICAL :: lrfp
!>  Use text output when writting wout files.
      LOGICAL :: loldout
!>  Get output for diagno 1.0 and 1.5.
      LOGICAL :: ldiagno
!>  Capture current sheets.
      LOGICAL :: lbsubs
!>  Write out full threed1 file if force tolarance is not met.
      LOGICAL :: lfull3d1out
!>  Write out text based woutfile.
      LOGICAL :: lwouttxt
!>  VMEC called from inside v3fit.
      LOGICAL :: l_v3fit
!>  Print out full nyquest spectrum. When flase trunate the spectrum to just the
!>  mpol, ntor values.
      LOGICAL :: lnyquist

!  Radial parameters.
!>  @deprecated
!>  Number of radial flux surfaces.
      INTEGER                             :: nsin
!>  Number of radial flux surfaces for each grid size.
      INTEGER, DIMENSION(max_grids)       :: ns_array
!>  Radial surface redistribution factors.
      REAL(rprec), DIMENSION(max_profile) :: aphi

!  Fourier sizes.
!>  Total number of poloidal modes.
      INTEGER :: mpol
!>  Largest value of the toroidal modes.
      INTEGER :: ntor
!>  Number of field periods.
      INTEGER :: nfp

!  Real space grid sizes.
!>  Number of poloidal grid points.
      INTEGER :: ntheta
!>  Number of toroidal grid points.
      INTEGER :: nzeta

!  External current parameters.
!>  Use external fields to determine the boundary shape.
      LOGICAL                                  :: lfreeb
!>  Array of external field coil currents.
!>  @note v3fit needs to point to this array.
      REAL (rprec), DIMENSION(nigroup), TARGET :: extcur
!>  Path to the mgrid file.
      CHARACTER (len=long_name)                :: mgrid_file

!  Profile Parameters.

!  Current profile parameters.
!>  Total toroidal current.
      REAL (rprec)                           :: curtor
!>  Control to use current profile.
      INTEGER                                :: ncurr
!>  Current profile type.
      CHARACTER (len=short_name)             :: pcurr_type
!>  Parameterized current profile coefficents.
      REAL (rprec), DIMENSION(0:max_profile) :: ac
!>  Radial knot position for the current profile splines.
      REAL (rprec), DIMENSION(ndatafmax)     :: ac_aux_s
!>  Knot amplitudes for the current profile splines.
      REAL (rprec), DIMENSION(ndatafmax)     :: ac_aux_f
!>  Expansion factor for the current profile.
      REAL (rprec)                           :: bloat

!  Rotational transform profile parameters.
!>  Rotational transform profile type.
      CHARACTER (len=short_name)             :: piota_type
!>  Parameterized rotational transform profile coefficents.
      REAL (rprec), DIMENSION(0:max_profile) :: ai
!>  Radial knot position for the rotational transform profile splines.
      REAL (rprec), DIMENSION(ndatafmax)     :: ai_aux_s
!>  Knot amplitudes for the rotational transform profile splines.
      REAL (rprec), DIMENSION(ndatafmax)     :: ai_aux_f

!  Pressure profile parameters.
!>  Scaling factor for the pressure profile.
      REAL (rprec)                           :: pres_scale
!>  Pressure profile type.
      CHARACTER (len=short_name)             :: pmass_type
!>  Parameterized rotational pressure coefficents.
      REAL (rprec), DIMENSION(0:max_profile) :: am
!>  Radial knot position for the pressure profile splines.
      REAL (rprec), DIMENSION(ndatafmax)     :: am_aux_s
!>  Knot amplitudes for the pressure profile splines.
      REAL (rprec), DIMENSION(ndatafmax)     :: am_aux_f
!>  Value of s beyond which pressure profile is flat.
      REAL (rprec)                           :: spres_ped

!  Anisotropy parameters.
!>  Hot particle energy deposition value for |B|.
      REAL (rprec)                           :: bcrit
!>  TPERP/TPAR ANIMEC only.
      REAL (rprec), DIMENSION(0:max_profile) :: at
!>  PHOT/PTHERMAL ANIMEC only.
      REAL (rprec), DIMENSION(0:max_profile) :: ah

!  Edge Boundary parameters.
!>  Total toroidal flux.
      REAL (rprec)                                   :: phiedge
!>  R boundary coefficients stellarator symmetric.
      REAL (rprec), DIMENSION(-ntord:ntord,0:mpol1d) :: rbc
!>  R boundary coefficients stellarator asymmetric.
      REAL (rprec), DIMENSION(-ntord:ntord,0:mpol1d) :: rbs
!>  Z boundary coefficients stellarator symmetric.
      REAL (rprec), DIMENSION(-ntord:ntord,0:mpol1d) :: zbs
!>  Z boundary coefficients stellarator asymmetric.
      REAL (rprec), DIMENSION(-ntord:ntord,0:mpol1d) :: zbc
!>  Filter poloidal boundary terms.
      INTEGER                                        :: mfilter_fbdy
!>  Filter toroidal boundary terms.
      INTEGER                                        :: nfilter_fbdy

!  Inital magnetic axis guess.
!>  R axis coefficients stellarator symmetric.
      REAL (rprec), DIMENSION(0:ntord) :: raxis_cc
!>  R axis coefficients stellarator asymmetric.
      REAL (rprec), DIMENSION(0:ntord) :: raxis_cs
!>  Z axis coefficients stellarator symmetric.
      REAL (rprec), DIMENSION(0:ntord) :: zaxis_cs
!>  Z axis coefficients stellarator asymmetric.
      REAL (rprec), DIMENSION(0:ntord) :: zaxis_cc
!>  @deprecated
!>  R axis coefficients stellarator symmetric.
      REAL (rprec), DIMENSION(0:ntord) :: raxis
!>  @deprecated
!>  Z axis coefficients stellarator symmetric.
      REAL (rprec), DIMENSION(0:ntord) :: zaxis

!  Unclassified.
      INTEGER                              :: imse
      INTEGER                              :: isnodes
      INTEGER                              :: itse
      INTEGER                              :: ipnodes
      INTEGER                              :: iopt_raxis
      INTEGER                              :: imatch_phiedge
      INTEGER                              :: nflxs
      INTEGER, DIMENSION(nbsetsp)          :: nbfld
      INTEGER, DIMENSION(nfloops)          :: indxflx
      INTEGER, DIMENSION(nbcoilsp,nbsetsp) :: indxbfld

      REAL (rprec) :: phidiam
      REAL (rprec) :: sigma_current
      REAL (rprec) :: sigma_delphid
      REAL (rprec) :: tensi
      REAL (rprec) :: tensp
      REAL (rprec) :: tensi2
      REAL (rprec) :: fpolyi
      REAL (rprec) :: presfac
      REAL (rprec) :: mseangle_offset
      REAL (rprec) :: pres_offset
      REAL (rprec) :: mseangle_offsetm

      REAL (rprec), DIMENSION(nmse)             :: mseprof
      REAL (rprec), DIMENSION(ntse)             :: rthom
      REAL (rprec), DIMENSION(ntse)             :: datathom
      REAL (rprec), DIMENSION(ntse)             :: sigma_thom
      REAL (rprec), DIMENSION(nmse)             :: rstark
      REAL (rprec), DIMENSION(nmse)             :: datastark
      REAL (rprec), DIMENSION(nmse)             :: sigma_stark
      REAL (rprec), DIMENSION(nfloops)          :: dsiobt
      REAL (rprec), DIMENSION(nfloops)          :: sigma_flux
      REAL (rprec), DIMENSION(nbcoilsp,nbsetsp) :: bbc
      REAL (rprec), DIMENSION(nbcoilsp,nbsetsp) :: sigma_b
      REAL (rprec), DIMENSION(ndatafmax)        :: psa
      REAL (rprec), DIMENSION(ndatafmax)        :: pfa
      REAL (rprec), DIMENSION(ndatafmax)        :: isa
      REAL (rprec), DIMENSION(ndatafmax)        :: ifa
      LOGICAL                                   :: lrecon
      LOGICAL                                   :: ledge_dump
      LOGICAL                                   :: lspectrum_dump
      LOGICAL                                   :: loptim
      LOGICAL                                   :: lpofr           !!Obsolete

      !> values of iter2 for which to dump data
      integer, dimension(num_iter2_to_dump) :: iter2_to_dump

      !> individual flags to control debug output loosely related to
      !> similarly-named routines (checkpoints along iterations)
      logical :: dump_add_fluxes            = .false.
      logical :: dump_metric                = .false.
      logical :: dump_volume                = .false.
      logical :: dump_bcontrav              = .false.
      logical :: dump_bcov                  = .false.
      logical :: dump_lambda_forces         = .false.
      logical :: dump_bcov_full             = .false.
      logical :: dump_precondn              = .false.
      logical :: dump_forceNorms_tcon       = .false.
      logical :: dump_lulv_comb             = .false.
      logical :: dump_calc_fbal             = .false.
      logical :: dump_evolve                = .false.
      logical :: dump_fixaray               = .false.
      logical :: dump_spectral_constraint   = .false.
      logical :: dump_forces                = .false.
      logical :: dump_totzsp_input          = .false.
      logical :: dump_funct3d_geometry      = .false.
      logical :: dump_constraint_force      = .false.
      logical :: dump_guess_axis            = .false.
      logical :: dump_interp                = .false.
      logical :: dump_con0                  = .false.
      logical :: dump_jacobian              = .false.
      logical :: dump_lamcal                = .false.
      logical :: dump_profil1d              = .false.
      logical :: dump_profil3d              = .false.
      logical :: dump_readin_boundary       = .false.
      logical :: dump_phys_gc               = .false.
      logical :: dump_fsq                   = .false.
      logical :: dump_scale_m1              = .false.
      logical :: dump_scalfor_out           = .false.
      logical :: dump_fsq1                  = .false.
      logical :: dump_scalfor_R             = .false.
      logical :: dump_scalfor_Z             = .false.
      logical :: dump_symforce              = .false.
      logical :: dump_tomnsps               = .false.
      logical :: dump_tomnspa               = .false.
      logical :: dump_multigrid_result      = .false.
      logical :: dump_rbsq                  = .false.
      logical :: dump_printout              = .false.

      ! fileout
      logical :: dump_bcovar_fileout        = .false.
      logical :: dump_bss                   = .false.
      logical :: dump_jxbforce_bsub_lowpass = .false.
      logical :: dump_jxbout                = .false.
      logical :: dump_mercier               = .false.
      logical :: dump_threed1_firstTable    = .false.
      logical :: dump_threed1_geomag        = .false.
      logical :: dump_threed1_volquant      = .false.
      logical :: dump_threed1_axis          = .false.
      logical :: dump_threed1_beta          = .false.
      logical :: dump_threed1_shafrint      = .false.
      logical :: dump_freeb_data            = .false.

      ! debugging output flags for NESTOR
      logical :: dump_vac1n_vacuum          = .false.
      logical :: dump_vac1n_precal          = .false.
      logical :: dump_vac1n_surface         = .false.
      logical :: dump_vac1n_bextern         = .false.
      logical :: dump_vac1n_analyt          = .false.
      logical :: dump_vac1n_greenf          = .false.
      logical :: dump_vac1n_fourp           = .false.
      logical :: dump_vac1n_fouri           = .false.
      logical :: dump_vac1n_solver          = .false.
      logical :: dump_vac1n_bsqvac          = .false.

      logical :: dump_vac2_vacuum           = .false.
      logical :: dump_vac2_precal           = .false.
      logical :: dump_vac2_surface          = .false.
      logical :: dump_vac2_bexmat           = .false.
      logical :: dump_vac2_matrix           = .false.
      logical :: dump_vac2_foumat_unreg     = .false.
      logical :: dump_vac2_analin           = .false.
      logical :: dump_vac2_analyt           = .false.
      logical :: dump_vac2_foumat           = .false.
      logical :: dump_vac2_linsys           = .false.
      logical :: dump_vac2_linslv           = .false.
      logical :: dump_vac2_bsqvac           = .false.

      CHARACTER(len=120) :: arg1

!>  Extension for the namelist input file name.
      CHARACTER (len=long_name) :: input_extension

!  Declare namelist
      NAMELIST /indata/                                                        &
!  Control Parameters.
     &   omp_num_threads, gamma, niter, niter_array, time_slice, nstep,        &
     &   nvacskip, delt, ftol, ftol_array, tcon0,                              &
!  Precondicioner control parameters.
     &   precon_type, prec2d_threshold,                                        &
!  Early termination control parameters.
     &   lgiveup, fgiveup,                                                     &
!  Continue control parameters.
     &   max_main_iterations,                                                  &
!  Control flags.
     &   lmove_axis, lmac, lforbal, lasym, lrfp, loldout, ldiagno,             &
     &   lbsubs, lfull3d1out, lwouttxt, lnyquist,                              &
!  Radial parameters.
     &   nsin, ns_array, aphi,                                                 &
!  Fourier sizes.
     &   mpol, ntor, nfp,                                                      &
!  Real space grid sizes.
     &   ntheta, nzeta,                                                        &
!  External current parameters.
     &   lfreeb, extcur, mgrid_file,                                           &
!  Current profile parameters.
     &   curtor, ncurr, pcurr_type, ac, ac_aux_s, ac_aux_f, bloat,             &
!  Rotational transform profile parameters.
     &   piota_type, ai, ai_aux_s, ai_aux_f,                                   &
!  Pressure profile parameters.
     &   pres_scale, pmass_type, am, am_aux_s, am_aux_f, spres_ped,            &
!  Anisotropy parameters.
     &   bcrit, at, ah,                                                        &
!  Edge Boundary parameters.
     &   phiedge, rbc, rbs, zbs, zbc, mfilter_fbdy, nfilter_fbdy,              &
!  Inital magnetic axis guess.
     &   raxis_cc, raxis_cs, zaxis_cs, zaxis_cc, raxis, zaxis,                 &
!  Unclassified.
     &   sigma_current, psa, pfa, isa, ifa, imatch_phiedge, iopt_raxis,        &
     &   tensi, tensp, mseangle_offset, mseangle_offsetm, imse,                &
     &   isnodes, rstark, datastark, sigma_stark, itse, ipnodes,               &
     &   presfac, pres_offset, rthom, datathom, sigma_thom, phidiam,           &
     &   sigma_delphid, tensi2, fpolyi, nflxs, indxflx, dsiobt,                &
     &   sigma_flux, nbfld, indxbfld, bbc, sigma_b, lpofr, lrecon,             &
     &   ledge_dump, lspectrum_dump, loptim,                                   &
     &   iter2_to_dump             ,                                           & ! dbgout
     &   dump_add_fluxes           ,                                           &
     &   dump_metric               ,                                           &
     &   dump_volume               ,                                           &
     &   dump_bcontrav             ,                                           &
     &   dump_bcov                 ,                                           &
     &   dump_lambda_forces        ,                                           &
     &   dump_bcov_full            ,                                           &
     &   dump_precondn             ,                                           &
     &   dump_forceNorms_tcon      ,                                           &
     &   dump_lulv_comb            ,                                           &
     &   dump_calc_fbal            ,                                           &
     &   dump_evolve               ,                                           &
     &   dump_fixaray              ,                                           &
     &   dump_spectral_constraint  ,                                           &
     &   dump_forces               ,                                           &
     &   dump_totzsp_input         ,                                           &
     &   dump_funct3d_geometry     ,                                           &
     &   dump_constraint_force     ,                                           &
     &   dump_guess_axis           ,                                           &
     &   dump_interp               ,                                           &
     &   dump_con0                 ,                                           &
     &   dump_jacobian             ,                                           &
     &   dump_lamcal               ,                                           &
     &   dump_profil1d             ,                                           &
     &   dump_profil3d             ,                                           &
     &   dump_readin_boundary      ,                                           &
     &   dump_phys_gc              ,                                           &
     &   dump_fsq                  ,                                           &
     &   dump_scale_m1             ,                                           &
     &   dump_scalfor_out          ,                                           &
     &   dump_fsq1                 ,                                           &
     &   dump_scalfor_R            ,                                           &
     &   dump_scalfor_Z            ,                                           &
     &   dump_symforce             ,                                           &
     &   dump_tomnsps              ,                                           &
     &   dump_tomnspa              ,                                           &
     &   dump_multigrid_result     ,                                           &
     &   dump_rbsq                 ,                                           &
     &   dump_printout             ,                                           &
     &   dump_bcovar_fileout       ,                                           & ! dbgout from fileout
     &   dump_bss                  ,                                           &
     &   dump_jxbforce_bsub_lowpass,                                           &
     &   dump_jxbout               ,                                           &
     &   dump_mercier              ,                                           &
     &   dump_threed1_firstTable   ,                                           &
     &   dump_threed1_geomag       ,                                           &
     &   dump_threed1_volquant     ,                                           &
     &   dump_threed1_axis         ,                                           &
     &   dump_threed1_beta         ,                                           &
     &   dump_threed1_shafrint     ,                                           &
     &   dump_freeb_data           ,                                           &
     &   dump_vac1n_vacuum         ,                                           & ! NESTOR vac1
     &   dump_vac1n_precal         ,                                           &
     &   dump_vac1n_surface        ,                                           &
     &   dump_vac1n_bextern        ,                                           &
     &   dump_vac1n_analyt         ,                                           &
     &   dump_vac1n_greenf         ,                                           &
     &   dump_vac1n_fourp          ,                                           &
     &   dump_vac1n_fouri          ,                                           &
     &   dump_vac1n_solver         ,                                           &
     &   dump_vac1n_bsqvac         ,                                           &
     &   dump_vac2_vacuum          ,                                           & ! NESTOR vac2
     &   dump_vac2_precal          ,                                           &
     &   dump_vac2_surface         ,                                           &
     &   dump_vac2_bexmat          ,                                           &
     &   dump_vac2_matrix          ,                                           &
     &   dump_vac2_foumat_unreg    ,                                           &
     &   dump_vac2_analin          ,                                           &
     &   dump_vac2_analyt          ,                                           &
     &   dump_vac2_foumat          ,                                           &
     &   dump_vac2_linsys          ,                                           &
     &   dump_vac2_linslv          ,                                           &
     &   dump_vac2_bsqvac

      NAMELIST /mseprofile/ mseprof

      CONTAINS

!-------------------------------------------------------------------------------
!>  @brief Initalize and read in namelist variables.
!>
!>  @param[in] iunit Fortran io unit for the open namelist file.
!-------------------------------------------------------------------------------
      SUBROUTINE read_indata_namelist(iunit, istat)

      IMPLICIT NONE

!  Declare Arguments
      INTEGER, INTENT(in)  :: iunit
      INTEGER, INTENT(out) :: istat

!  Start of executable code

!  Initializations

!  Control Parameters.
      omp_num_threads = 8
      gamma = 0
      niter_array = -1;
      time_slice = 0
      niter = 100
      nstep = 10
      nvacskip = 1
      delt = 1
      ftol = 1.E-10_dp
      ftol_array = 0
      ftol_array(1) = ftol
      tcon0 = 1

!  Precondition Parameters.
      precon_type = 'NONE'
      prec2d_threshold = 1.E-30_dp

!  Earily termination parameters.
      lgiveup = .false.        ! inserted M.Drevlak
      fgiveup = 3.E+01_dp      ! inserted M.Drevlak

!  Control flags.
      lmove_axis = .true.
      lmac = .false.
      lforbal = .false.        ! SPH: changed 05-14-14
      lasym = .false.
      lrfp = .false.
      loldout = .false.        ! J Geiger 2010-05-04 start
      ldiagno = .false.
      lbsubs = .false.         ! J Hanson. See jxbforce coding
      lfull3d1out = .true.     ! J Geiger & SPH (5-21-15)
      lmoreiter = .false.      ! default value if no max_main_iterations given.
      max_main_iterations = 1  ! to keep a presumably expected standard behavior.



      lwouttxt = .true.        ! and without netcdf

                               ! J Geiger 2010-05-04 end
      lnyquist = .true.

!  Radial grid size.
      nsin = ns_default
      ns_array = 0
      ns_array(1) = ns_default
      aphi = 0
      aphi(1) = 1

!  Fourier sizes.
      mpol = mpol_default
      ntor = ntor_default
      nfp = 1

!  Real space size terms.
      ntheta = 0
      nzeta = 0

!  External field terms.
      lfreeb = .true.

! HACK: Need to fill extcur with cbig in order to figure out which elements
!       were set by the user in the INDATA namelist.
!      extcur = 0
      mgrid_file = 'NONE'

!  Plasma current parameters.
      curtor = 0
      ncurr = 0
      pcurr_type = 'power_series'
      ac = 0
      ac_aux_s = 0
      ac_aux_f = 0
      bloat = 1

!  Rotational transform parameters.
      piota_type = 'power_series'
      ai = 0
      ai_aux_s = 0
      ai_aux_f = 0

!  Pressure profile parameters.
      pres_scale = 1
      pmass_type = 'power_series'
      am = 0
      am_aux_s = 0
      am_aux_f = 0
      spres_ped = 1

!  Anisotropy parameters
      bcrit = 1
      at(0) = 1
      at(1:) = 0
      ah = 0

!  Edge boundary terms
      phiedge = 1
      rbc = 0
      rbs = 0
      zbs = 0
      zbc = 0
      mfilter_fbdy = -1
      nfilter_fbdy = -1

!  Initial axis guess.
      raxis_cc = 0
      zaxis_cs = 0
      raxis_cs = 0
      zaxis_cc = 0
!  Backwards compatibility.
      raxis = 0
      zaxis = 0

      READ (iunit, nml=indata, iostat=istat)

      IF (ALL(niter_array == -1)) THEN
         niter_array = niter
      END IF

!  Work around a bug in gfortran. When performing an optimized build, the WHERE
!  statement would produce incorrect results. Work around this bug by expanding
!  the full WHERE statment. This should have no adverse effects on any other
!  compiler since these statements are equivalent to the older code statement.
!
!     WHERE (raxis .ne. 0.0_dp) raxis_cc = raxis
!     WHERE (zaxis .ne. 0.0_dp) zaxis_cs = zaxis
!
!  The effect of this bug optimized to code to effectively ignore the WHERE
!  statement and assign all value values of the r/zaxis to the r/zaxis_cc/s
!  arrays. Explicitly adding the r/zaxis .eq. 0.0_dp section prevents this. This
!  bug is known to exist in gfortran 4.9. It may manifest in other versions.
      WHERE (raxis .ne. 0.0_dp)
         raxis_cc = raxis
      ELSEWHERE
         raxis_cc = raxis_cc
      ENDWHERE
      WHERE (zaxis .ne. 0.0_dp)
         zaxis_cs = zaxis
      ELSEWHERE
         zaxis_cs = zaxis_cs
      ENDWHERE

      raxis_cs(0) = 0
      zaxis_cs(0) = 0

      IF (max_main_iterations .GT. 1) THEN
         lmoreiter = .true.  !J Geiger: if more iterations are requested.
      END IF

      END SUBROUTINE read_indata_namelist

      SUBROUTINE read_mse_namelist (iunit, istat)
      INTEGER :: iunit, istat

      READ (iunit, nml=mseprofile, iostat=istat)

      END SUBROUTINE

      END MODULE
