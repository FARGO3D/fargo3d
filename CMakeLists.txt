cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0079 NEW)

cmake_minimum_required(VERSION 3.23)
project(fargo3d LANGUAGES C CUDA)

include(Citations.txt)
set(citations "")
list(APPEND citations ${citation_main})

################################################################################
# BASIC OPTIONS
################################################################################

set(SETUP "fargo" CACHE STRING "Name of setup directory to use")
set(setup_dir ${CMAKE_SOURCE_DIR}/setups/${SETUP})

set(UNITS "NONE" CACHE BOOL "System of units to use for rescaling")
set_property(CACHE UNITS PROPERTY STRINGS "NONE" "CGS" "MKS")
add_compile_definitions(
    CGS=$<BOOL:$<STREQUAL:$<UPPER_CASE:${UNITS}>,CGS>>
    MKS=$<BOOL:$<STREQUAL:$<UPPER_CASE:${UNITS}>,MKS>>)

set(RESCALE OFF CACHE BOOL "Rescale parameters to units `UNITS`")
add_compile_definitions(RESCALE=$<BOOL:${RESCALE}>)

set(COORDSYS "CARTESIAN" CACHE STRING "Coordinate system")
set_property(CACHE COORDSYS PROPERTY STRINGS "CARTESIAN" "CYLINDRICAL" "SPHERICAL")
add_compile_definitions(
    CARTESIAN=$<BOOL:$<STREQUAL:$<UPPER_CASE:${COORDSYS}>,CARTESIAN>>
    CYLINDRICAL=$<BOOL:$<STREQUAL:$<UPPER_CASE:${COORDSYS}>,CYLINDRICAL>>
    SPHERICAL=$<BOOL:$<STREQUAL:$<UPPER_CASE:${COORDSYS}>,SPHERICAL>>)

set(SHEARINGBOX OFF CACHE BOOL "Enable shearing box non-inertial reference frame")
add_compile_definitions(SHEARINGBOX=$<BOOL:${SHEARINGBOX}>)

set(SHEARINGSHEET2D OFF CACHE BOOL "Enable shearing sheet in 2d")
add_compile_definitions(SHEARINGSHEET2D=$<BOOL:${SHEARINGSHEET2D}>)

set(SHEARINGBOX3D OFF CACHE BOOL "Enable shearing box in 3d")
add_compile_definitions(SHEARINGBOX3D=$<BOOL:${SHEARINGBOX3D}>)

set(SHEARINGBC OFF CACHE BOOL "Enable shearing boundary condition")
add_compile_definitions(SHEARINGBC=$<BOOL:${SHEARINGBC}>)

set(STANDARD OFF CACHE BOOL "Force standard advection in x")
add_compile_definitions(STANDARD=$<BOOL:${STANDARD}>)
if(NOT STANDARD)
    list(APPEND citations ${citation_advection})
endif()

set(RAM OFF CACHE BOOL "Enable rapid advection on arbitrary mesh")
add_compile_definitions(RAM=$<BOOL:${RAM}>)
if(RAM)
    list(APPEND citations ${citation_ram})
endif()

set(DONOR OFF CACHE BOOL "Activate donor cell flux limiter (deactivating van Leer limiter)")
add_compile_definitions(DONOR=$<BOOL:${DONOR}>)

set(PPA_STEEPENER OFF CACHE BOOL "Activate PPA steepener")
add_compile_definitions(PPA_STEEPENER=$<BOOL:${PPA_STEEPENER}>)

set(GHOSTSX OFF CACHE BOOL "Enable ghost cells in x")
add_compile_definitions(GHOSTSX=$<BOOL:${GHOSTSX}>)

set(HALFDISK OFF CACHE BOOL "Simulate half-disk")
add_compile_definitions(HALFDISK=$<BOOL:${HALFDISK}>)

set(XDIM ON CACHE BOOL "Enable x-dimension")
add_compile_definitions(XDIM=$<BOOL:${XDIM}>)

set(YDIM OFF CACHE BOOL "Enable y-dimension")
add_compile_definitions(YDIM=$<BOOL:${YDIM}>)

set(ZDIM OFF CACHE BOOL "Enable z-dimension")
add_compile_definitions(ZDIM=$<BOOL:${ZDIM}>)

set(FLOOR OFF CACHE BOOL "Enable density floor")
add_compile_definitions(FLOOR=$<BOOL:${FLOOR}>)

################################################################################
# OUTPUT OPTIONS
################################################################################

set(NOOUTPUTS OFF CACHE BOOL "Disable outputs")
add_compile_definitions(NOOUTPUTS=$<BOOL:${NOOUTPUTS}>)

set(NOFLUSH OFF CACHE BOOL "Disable stdout flushing")
add_compile_definitions(NOFLUSH=$<BOOL:${NOFLUSH}>)

set(LEGACY OFF CACHE BOOL "Enable legacy outputs")
add_compile_definitions(LEGACY=$<BOOL:${LEGACY}>)

set(WRITEGHOSTS OFF CACHE BOOL "Write ghost cells to output")
add_compile_definitions(WRITEGHOSTS=$<BOOL:${WRITEGHOSTS}>)

set(MPIIO OFF CACHE BOOL "Use MPI I/O for output")
add_compile_definitions(MPIIO=$<BOOL:${MPIIO}>)

set(LONGSUMMARY OFF CACHE BOOL "Ouptut longer summary files at coarse timesteps")
add_compile_definitions(LONGSUMMARY=$<BOOL:${LONGSUMMARY}>)

set(MATPLOTLIB OFF CACHE BOOL "Build with Matplotlib plotting")
add_compile_definitions(MATPLOTLIB=$<BOOL:${MATPLOTLIB}>)
if(MATPLOTLIB)
    find_package(Python REQUIRED COMPONENTS Development)
    target_link_libraries(fargo3d Python::Python)
endif()

################################################################################
# PERFORMANCE OPTIONS
################################################################################

set(BIGMEM ON CACHE BOOL "Store 1d arrays in global (not constant) GPU memory")
add_compile_definitions(BIGMEM=$<BOOL:${BIGMEM}>)

set(PARALLEL OFF CACHE BOOL "Build CPU parallel version with MPI")
add_compile_definitions(PARALLEL=$<BOOL:${PARALLEL}>)

set(MPICUDA OFF CACHE BOOL "Activate MPI communication between GPUs")
add_compile_definitions(MPICUDA=$<BOOL:${MPICUDA}>)

set(GPU OFF CACHE BOOL "Build GPU version with CUDA")
add_compile_definitions(GPU=$<BOOL:${GPU}>)

set(BLOCK_X "1" CACHE STRING "Number of GPU blocks in x")
add_compile_definitions(BLOCK_X=${BLOCK_X})

set(BLOCK_Y "1" CACHE STRING "Number of GPU blocks in y")
add_compile_definitions(BLOCK_Y=${BLOCK_Y})

set(BLOCK_Z "1" CACHE STRING "Number of GPU blocks in z")
add_compile_definitions(BLOCK_Z=${BLOCK_Z})

set(FLOAT OFF CACHE BOOL "Use single-precision floating point")
add_compile_definitions(FLOAT=$<BOOL:${FLOAT}>)

set(PROFILING OFF CACHE BOOL "Enable profiling mode")
add_compile_definitions(PROFILING=$<BOOL:${PROFILING}>)

# CMake already has a nice way to handle debug mode, so just forward that along
# rather than defining a separate option
add_compile_definitions(DEBUG=$<CONFIG:Debug>)

################################################################################
# N-BODY OPTIONS
################################################################################

set(BM08 OFF CACHE BOOL "Enable Baruteau & Masset (2008) resonance workaround")
add_compile_definitions(BM08=$<BOOL:${BM08}>)

set(GASINDIRECTTERM OFF CACHE BOOL "Enable additional potential term from frame acceleration")
add_compile_definitions(GASINDIRECTTERM=$<BOOL:${GASINDIRECTTERM}>)

set(NODEFAULTSTAR OFF CACHE BOOL "Remove the default central star")
add_compile_definitions(NODEFAULTSTAR=$<BOOL:${NODEFAULTSTAR}>)

################################################################################
# EQUATION OF STATE OPTIONS
################################################################################

set(EQNOFSTATE "ISOTHERMAL" CACHE STRING "Equation of state")
set_property(CACHE EQNOFSTATE PROPERTY STRINGS "ISOTHERMAL" "ADIABATIC" "POLYTROPIC")
add_compile_definitions(
    ISOTHERMAL=$<BOOL:$<STREQUAL:$<UPPER_CASE:${EQNOFSTATE}>,ISOTHERMAL>>
    ADIABATIC=$<BOOL:$<STREQUAL:$<UPPER_CASE:${EQNOFSTATE}>,ADIABATIC>>
    POLYTROPIC=$<BOOL:$<STREQUAL:$<UPPER_CASE:${EQNOFSTATE}>,POLYTROPIC>>)

################################################################################
# MULTIFLUID OPTIONS
################################################################################

set(NFLUIDS 1 CACHE STRING "Number of fluids, > 1 for multifluid version")
add_compile_definitions(NFLUIDS=${NFLUIDS})
if(NFLUIDS GREATER 1)
    list(APPEND citations ${citation_multifluid})
endif()

set(DRAGFORCE OFF CACHE BOOL "Activate drag force between fluids")
add_compile_definitions(DRAGFORCE=$<BOOL:${DRAGFORCE}>)

set(STOKESNUMBER OFF CACHE BOOL "Collision rate assumes constant Stokes number with Keplerian Omega")
add_compile_definitions(STOKESNUMBER=$<BOOL:${STOKESNUMBER}>)

set(CONSTANTSTOKESNUMBER OFF CACHE BOOL "Collision rate assumes constant Stokes number with constant Omega")
add_compile_definitions(CONSTANTSTOKESNUMBER=$<BOOL:${CONSTANTSTOKESNUMBER}>)

set(CONSTANTDRAG OFF CACHE BOOL "Collision rate assumes constant drag force")
add_compile_definitions(CONSTANTDRAG=$<BOOL:${CONSTANTDRAG}>)

set(COLLISIONPREDICTOR OFF CACHE BOOL "Enable collision prediction step")
add_compile_definitions(COLLISIONPREDICTOR=$<BOOL:${COLLISIONPREDICTOR}>)

set(DUSTDIFFUSION OFF CACHE BOOL "Enable dust diffusion module")
add_compile_definitions(DUSTDIFFUSION=$<BOOL:${DUSTDIFFUSION}>)
if(DUSTDIFFUSION)
    list(APPEND citations ${citation_dust_diffusion})
endif()

################################################################################
# MAGNETOHYDRODYNAMICS OPTIONS
################################################################################

set(MHD OFF CACHE BOOL "Enable magnetohydrodynamics module")
add_compile_definitions(MHD=$<BOOL:${MHD}>)

set(PASSIVEMHD OFF CACHE BOOL "Enble passive MHD")
add_compile_definitions(PASSIVEMHD=$<BOOL:${PASSIVEMHD}>)

set(STRICTSYM OFF CACHE BOOL "Enforce strict symmetry in MHD solver")
add_compile_definitions(STRICTSYM=$<BOOL:${STRICTSYM}>)

set(OHMICDIFFUSION OFF CACHE BOOL "Activate MHD Ohmic resistivity")
add_compile_definitions(OHMICDIFFUSION=$<BOOL:${OHMICDIFFUSION}>)

set(AMBIPOLARDIFFUSION OFF CACHE BOOL "Activate MHD ambipolar resistivity")
add_compile_definitions(AMBIPOLARDIFFUSION=$<BOOL:${AMBIPOLARDIFFUSION}>)

set(HALLEFFECT OFF CACHE BOOL "Activate MHD Hall effect")
add_compile_definitions(HALLEFFECT=$<BOOL:${HALLEFFECT}>)

if(MHD)
    if(NOT STANDARD)
        # using MHD and FARGO algorithm
        list(APPEND citations ${citation_mhd_advection})
    endif()

    if (HALLEFFECT OR AMBIPOLARDIFFUSION)
        # using Hall effect or ambipolar diffusion
        list(APPEND citations ${citation_hall_ambipolar})
    endif()
endif()

################################################################################
# ADDITIONAL PHYSICS OPTIONS
################################################################################

set(VISCOSITY OFF CACHE BOOL "Enable gas viscosity")
add_compile_definitions(VISCOSITY=$<BOOL:${VISCOSITY}>)

set(ALPHAVISCOSITY OFF CACHE BOOL "Enable alpha gas viscosity")
add_compile_definitions(ALPHAVISCOSITY=$<BOOL:${ALPHAVISCOSITY}>)

set(POTENTIAL OFF CACHE BOOL "Enable gravity module")
add_compile_definitions(POTENTIAL=$<BOOL:${POTENTIAL}>)

set(STOCKHOLM OFF CACHE BOOL "Enable wave-killing Stockholm boundary")
add_compile_definitions(STOCKHOLM=$<BOOL:${STOCKHOLM}>)

set(HILLCUT OFF CACHE BOOL "Enable cut for the force computation")
add_compile_definitions(HILLCUT=$<BOOL:${HILLCUT}>)

set(NOSUBSTEP2 OFF CACHE BOOL "Disable artificial viscosity module")
add_compile_definitions(NOSUBSTEP2=$<BOOL:${NOSUBSTEP2}>)

set(STRONG_SHOCK OFF CACHE BOOL "Enable linear artificial pressure (may help with strong shocks)")
add_compile_definitions(STRONG_SHOCK=$<BOOL:${STRONG_SHOCK}>)

set(MONITOR_SCALAR "" CACHE STRING "Pipe-delimited scalars to monitor")
set(MONITOR_2D "" CACHE STRING "Pipe-delimited 2d scalars to monitor")
set(MONITOR_Y "" CACHE STRING "Pipe-delimited y-averaged scalars to monitor")
set(MONITOR_Y_RAW "" CACHE STRING "Pipe-delimited raw y-averaged scalars to monitor")
set(MONITOR_Z "" CACHE STRING "Pipe-delimited z-averaged scalars to monitor")
set(MONITOR_Z_RAW "" CACHE STRING "Pipe-delimited raw z-averaged scalars to monitor")
add_compile_definitions(MONITOR_SCALAR=${MONITOR_SCALAR}
    MONITOR_2D=${MONITOR_2D} MONITOR_Y=${MONITOR_Y}
    MONITOR_Y_RAW=${MONITOR_Y_RAW} MONITOR_Z=${MONITOR_Z}
    MONITOR_Z_RAW=${MONITOR_Z_RAW} SETUPNAME=${SETUP})

add_executable(fargo3d)
add_subdirectory(${setup_dir})
add_subdirectory(src)

file(WRITE ${CMAKE_BINARY_DIR}/citations.bib ${citations})
add_custom_target(bibliography DEPENDS ${CMAKE_BINARY_DIR}/citations.bib)
add_dependencies(fargo3d bibliography)

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/std/func_arch.cfg
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/std/func_arch.cfg
    ${CMAKE_CURRENT_BINARY_DIR}/std/func_arch.cfg
    DEPENDS ${CMAKE_SOURCE_DIR}/std/func_arch.cfg)
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/in/${SETUP}.par
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/in/${SETUP}.par
    ${CMAKE_CURRENT_BINARY_DIR}/in/${SETUP}.par
    DEPENDS ${CMAKE_SOURCE_DIR}/in/${SETUP}.par)
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/planets
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/planets
    ${CMAKE_CURRENT_BINARY_DIR}/planets DEPENDS ${CMAKE_SOURCE_DIR}/planets)
add_custom_target(config DEPENDS ${CMAKE_BINARY_DIR}/std/func_arch.cfg
    ${CMAKE_BINARY_DIR}/planets ${CMAKE_BINARY_DIR}/in/${SETUP}.par)
add_dependencies(fargo3d config)
