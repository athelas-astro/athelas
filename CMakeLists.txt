# ===
# This is probably a _hot mess_
# But it works.
# ===

cmake_minimum_required( VERSION 3.20 )
cmake_policy(SET CMP0074 NEW)

# If the user doesn't specify a build type, prefer RelWithDebInfo
set(default_build_type "RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# export compile flags for clangd LSP
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR}/compile_commands.json
)

# If this is a debug build, set kokkos debug on
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message(STATUS "Enabling Kokkos debug mode")
  set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "Most general debug settings")
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL
     "Bounds checking on Kokkos views")
  set(Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK ON CACHE BOOL
      "Sanity checks on Kokkos DualView")
    add_compile_definitions(ATHELAS_DEBUG)
endif()

set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL
    "Kokkos aggressive vectorization" FORCE)

### c++23 standards
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS False)

set(CMAKE_FIND_FRAMEWORK NEVER)
set(CMAKE_FIND_APPBUNDLE NEVER)

project( athelas VERSION 25.10 LANGUAGES CXX )
enable_language(CXX)

# for tracking git hash
include("cmake/build_info.cmake")
CheckGitSetup()

include(CTest)
option(ATHELAS_ENABLE_UNIT_TESTS "Enable unit tests" OFF)
option(ATHELAS_ENABLE_INTEGRATION_TESTS "Enable integration tests" OFF) # Future
option(ATHELAS_ENABLE_REGRESSION_TESTS "Enable regression tests" OFF)   # Future


# Don't allow in-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
  message(FATAL_ERROR
    "You cannot build in a source directory (or any directory with a CMakeLists.txt file). "
    "Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

# Default to an external Kokkos package if the submodule is not populated
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/Kokkos/CMakeLists.txt" AND NOT EXISTS "${Kokkos_ROOT}/CMakeLists.txt")
    option(ATHELAS_IMPORT_KOKKOS "If ON, attempt to link to an external Kokkos src. If OFF, build Kokkos from source and package with ATHELAS" ON)
else()
    option(ATHELAS_IMPORT_KOKKOS "If ON, attempt to link to an external Kokkos src. If OFF, build Kokkos from source and package with ATHELAS" OFF)
endif()

# === Sanitizer Support ===
option(ATHELAS_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(ATHELAS_ENABLE_SANITIZERS)
    message(STATUS "Sanitizers enabled: AddressSanitizer + UndefinedBehaviorSanitizer")

    # List of sanitizer-related compile and link flags
    set(SANITIZER_FLAGS "-fsanitize=address,undefined" "-fno-omit-frame-pointer")

    # Apply to all targets globally (alternatively, apply per target)
    add_compile_options(${SANITIZER_FLAGS})
    add_link_options(${SANITIZER_FLAGS})
endif()

include_directories(src)
include_directories("src/composition")
include_directories("src/concepts")
include_directories("src/engines")
include_directories("src/geometry")
include_directories("src/heating")
include_directories("src/history")
include_directories("src/interface")
include_directories("src/quadrature")
include_directories("src/utils")
include_directories("src/limiters")
include_directories("src/basis")
include_directories("src/fluid")
include_directories("src/radiation")
include_directories("src/pgen")
include_directories("src/linalg")
include_directories("src/eos")
include_directories("src/bc")
include_directories("src/io")
include_directories("src/opacity")
include_directories("src/state")
include_directories("src/solvers")
include_directories("src/timestepper")

include_directories(athelas PUBLIC
  "${PROJECT_BINARY_DIR}"
  )

# --- Source Files ---
set( ATHELAS_SOURCES 
  "src/driver.cpp"
  "src/composition/compdata.cpp"
  "src/composition/composition.cpp"
  "src/pgen/problem_in.cpp"
  #  "src/basis/polynomial_basis.cpp"
  "src/basis/nodal_basis.cpp"
  "src/bc/boundary_conditions.cpp"
  "src/engines/thermal.cpp"
  "src/eos/eos_ideal.cpp"
  "src/eos/eos_marshak.cpp"
  "src/eos/eos_paczynski.cpp"
  "src/eos/eos_polytropic.cpp"
  "src/fluid/hydro_package.cpp"
  "src/heating/nickel_package.cpp"
  "src/history/history.cpp"
  "src/interface/params.cpp"
  "src/radiation/radhydro_package.cpp"
  "src/geometry/geometry_package.cpp"
  "src/geometry/grid.cpp"
  "src/gravity/gravity_package.cpp"
  "src/io/io.cpp"
  "src/linalg/linear_algebra.cpp"
  "src/quadrature/quadrature.cpp"
  "src/limiters/bound_enforcing_limiter.cpp"
  "src/limiters/slope_limiter_unlimited.cpp"
  "src/limiters/slope_limiter_utilities.cpp"
  "src/limiters/slope_limiter_tvdminmod.cpp"
  "src/limiters/slope_limiter_weno.cpp"
  "src/opacity/opac_constant.cpp"
  "src/opacity/opac_powerlaw.cpp"
  "src/solvers/hydrostatic_equilibrium.cpp"
  "src/state/state.cpp"
  "src/timestepper/tableau.cpp"
  "src/timestepper/timestepper.cpp"
  "src/timestepper/operator_split_stepper.cpp" )

# hack because Clang is legacy..
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -fopenmp=libomp --std=c++23")
    set(OpenMP_CXX_FLAGS="-fopenmp=libomp")
    include_directories(${OpenMP_CXX_INCLUDE_DIRS})
endif()
# Check for FindOpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_CXX_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OPEN MP ${OpenMP_CXX_SPEC_DATE}")
    set(ENABLE_OPENMP ON)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Allow Kokkos to use OpenMP as execution space.")
    include_directories(${OpenMP_CXX_INCLUDE_DIRS})
endif(OPENMP_CXX_FOUND)

set(ENABLE_COMPILER_WARNINGS False)
if (ENABLE_COMPILER_WARNINGS)
  message(STATUS "Enabling -Wall and setting Kokkos_ENABLE_COMPILER_WARNINGS=True")
  set(Kokkos_ENABLE_COMPILER_WARNINGS True CACHE BOOL
      "Make the compiler warn us about things")
  add_compile_options(-Wall)
endif()

# Kokkos
if (NOT TARGET Kokkos::kokkos)
  if (ATHELAS_IMPORT_KOKKOS)
    find_package(Kokkos 4.7)
    if (NOT Kokkos_FOUND)
      unset(ATHELAS_IMPORT_KOKKOS CACHE)
      message(FATAL_ERROR "Could not find external Kokkos. Consider importing a Kokkos installation into your environment or disabling external Kokkos with e.g. -DATHELAS_IMPORT_KOKKOS=OFF")
    endif()
  else()
    if (EXISTS ${Kokkos_ROOT}/CMakeLists.txt)
      add_subdirectory(${Kokkos_ROOT} Kokkos)
      message(STATUS "Using Kokkos source from Kokkos_ROOT=${Kokkos_ROOT}")
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/Kokkos/CMakeLists.txt)
      add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/Kokkos)
      include_directories(${Kokkos_INCLUDE_DIRS_RET})
      message(STATUS "Using Kokkos source from Athelas submodule at ${CMAKE_CURRENT_SOURCE_DIR}/external/Kokkos")
    else()
      message(FATAL_ERROR "Could not find Kokkos source. Consider running `git submodule update --init`, providing the path to a Kokkos source directory with Kokkos_ROOT, or setting ATHELAS_IMPORT_KOKKOS=ON to link to an external Kokkos installation.")
    endif()
  endif()
endif()

# The following lines determine the default loop pattern by setting the
# default defines to the appropriate "tags" in the config.hpp file.
# See `kokkos_abstraction.hpp` for available tags and what they translate to.
if (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP)
  set(PAR_LOOP_LAYOUT "MANUAL1D_LOOP" CACHE STRING
    "Default loop layout for parallel_for wrapper")

  set(PAR_LOOP_FLAT_LAYOUT "MANUAL1D_LOOP" CACHE STRING
    "Default loop layout for 1d parallel_for wrapper")

  set(PAR_LOOP_LAYOUT_VALUES "MANUAL1D_LOOP;MDRANGE_LOOP;TPTTR_LOOP;TPTTRTVR_LOOP"
    CACHE STRING "Possible loop layout options.")

  set(PAR_LOOP_INNER_LAYOUT "TVR_INNER_LOOP" CACHE STRING
    "Default loop layout for par_for_inner wrapper")

  set(PAR_LOOP_INNER_LAYOUT_VALUES "TVR_INNER_LOOP"
    CACHE STRING "Possible inner loop layout options.")

elseif(Kokkos_ENABLE_HPX)
  message( FATAL_ERROR "Need to add/fix/test default loop layouts for HPX backend.")

else()
  # use manual1d on host
  set(PAR_LOOP_LAYOUT "MANUAL1D_LOOP" CACHE STRING
    "Default loop layout for parallel_for wrapper")
  set(PAR_LOOP_FLAT_LAYOUT "MANUAL1D_LOOP" CACHE STRING
    "Default loop layout for parallel_for wrapper")
  set(PAR_LOOP_LAYOUT_VALUES "SIMDFOR_LOOP;MANUAL1D_LOOP;MDRANGE_LOOP;TPTTR_LOOP;TPTVR_LOOP;TPTTRTVR_LOOP"
    CACHE STRING "Possible loop layout options.")

  set(PAR_LOOP_INNER_LAYOUT "SIMDFOR_INNER_LOOP" CACHE STRING
    "Default loop layout for par_for_inner wrapper")

  set(PAR_LOOP_INNER_LAYOUT_VALUES "SIMDFOR_INNER_LOOP;TVR_INNER_LOOP"
    CACHE STRING "Possible inner loop layout options.")

endif()

set_property(CACHE PAR_LOOP_LAYOUT PROPERTY STRINGS ${PAR_LOOP_LAYOUT_VALUES})

set_property(CACHE PAR_LOOP_FLAT_LAYOUT PROPERTY STRINGS ${PAR_LOOP_FLAT_LAYOUT_VALUES})

set_property(CACHE PAR_LOOP_INNER_LAYOUT PROPERTY STRINGS ${PAR_LOOP_INNER_LAYOUT_VALUES})

message(STATUS "PAR_LOOP_LAYOUT='${PAR_LOOP_LAYOUT}' (default par_for wrapper layout)")

message(STATUS "PAR_LOOP_FLAT_LAYOUT='${PAR_LOOP_FLAT_LAYOUT}' (default 1d par_for wrapper layout)")

message(STATUS "PAR_LOOP_INNER_LAYOUT='${PAR_LOOP_INNER_LAYOUT}' (default par_for_inner wrapper layout)")

if (${PAR_LOOP_LAYOUT} STREQUAL "MANUAL1D_LOOP")
  set(PAR_LOOP_LAYOUT_TAG loop_pattern_flatrange_tag)
elseif (${PAR_LOOP_LAYOUT} STREQUAL "SIMDFOR_LOOP")
  set(PAR_LOOP_LAYOUT_TAG loop_pattern_simdfor_tag)
elseif (${PAR_LOOP_LAYOUT} STREQUAL "MDRANGE_LOOP")
  set(PAR_LOOP_LAYOUT_TAG loop_pattern_mdrange_tag)
elseif (${PAR_LOOP_LAYOUT} STREQUAL "TP_TTR_LOOP")
  set(PAR_LOOP_LAYOUT_TAG loop_pattern_tpttr_tag)
elseif (${PAR_LOOP_LAYOUT} STREQUAL "TP_TVR_LOOP")
  set(PAR_LOOP_LAYOUT_TAG loop_pattern_tptvr_tag)
elseif (${PAR_LOOP_LAYOUT} STREQUAL "TPTTRTVR_LOOP")
  set(PAR_LOOP_LAYOUT_TAG loop_pattern_tpttrtvr_tag)
else()
  set(PAR_LOOP_LAYOUT_TAG loop_pattern_undefined_tag)
endif()

if (${PAR_LOOP_FLAT_LAYOUT} STREQUAL "MANUAL1D_LOOP")
  set(PAR_LOOP_FLAT_LAYOUT_TAG loop_pattern_flatrange_tag)
elseif (${PAR_LOOP_FLAT_LAYOUT} STREQUAL "SIMDFOR_LOOP")
  set(PAR_LOOP_FLAT_LAYOUT_TAG loop_pattern_simdfor_tag)
endif()

if (${PAR_LOOP_INNER_LAYOUT} STREQUAL "TVR_INNER_LOOP")
  set(PAR_LOOP_INNER_LAYOUT_TAG "InnerLoopPatternTVR()")
elseif (${PAR_LOOP_INNER_LAYOUT} STREQUAL "SIMDFOR_INNER_LOOP")
  set(PAR_LOOP_INNER_LAYOUT_TAG inner_loop_pattern_simdfor_tag)
else()
  set(PAR_LOOP_INNER_LAYOUT_TAG loop_pattern_undefined_tag)
endif()

configure_file(
  ${CMAKE_SOURCE_DIR}/src/loop_layout.hpp.in
  ${CMAKE_SOURCE_DIR}/src/loop_layout.hpp
  @ONLY
)

# Eigen
add_subdirectory(external/eigen)

# HDF5
find_package ( HDF5 REQUIRED COMPONENTS CXX HL )
message(STATUS "HDF5_LIBRARIES = ${HDF5_CXX_LIBRARIES}")
include_directories(${HDF5_INCLUDE_DIRS}) # seems to be needed for compilation on github actions...

# toml++
include_directories("external/tomlplusplus")

# --- main ---

# --- athelas executable ---
add_executable(athelas src/main.cpp ${ATHELAS_SOURCES})
target_compile_features(athelas PRIVATE cxx_std_23)
set_target_properties(athelas PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
    #RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

# Set compile options
target_compile_options(athelas PRIVATE 
        #    -fPIC 
    -Wall 
    ${OpenMP_CXX_FLAGS}
)

# Set compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(athelas PRIVATE -stdlib=libstdc++ -fopenmp=libomp)
    target_link_options(athelas PRIVATE -stdlib=libstdc++)

    # Find and link libomp explicitly for Clang
    find_library(LIBOMP_LIBRARY NAMES omp)
    if(LIBOMP_LIBRARY)
        target_link_libraries(athelas PRIVATE ${LIBOMP_LIBRARY})
    endif()
endif()

# Link libraries (do this only once, with all libraries)
target_link_libraries(athelas PRIVATE
        #    libathelas
    kokkos
    Eigen3::Eigen
    git_version
    OpenMP::OpenMP_CXX
    ${HDF5_C_LIBRARIES}
    ${HDF5_LIBRARIES}
    stdc++exp
)

if ( ATHELAS_ENABLE_UNIT_TESTS )
  message("\nConfiguring tests")
  enable_testing()
  add_subdirectory( test )
endif()
