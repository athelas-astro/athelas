# ===
# This is probably a _hot mess_
# But it works.
# ===

cmake_minimum_required( VERSION 3.10 )
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

### c++11 standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS False)

set(CMAKE_FIND_FRAMEWORK NEVER)
set(CMAKE_FIND_APPBUNDLE NEVER)

project( athelas VERSION 0.2 LANGUAGES CXX )
enable_language(CXX)

include(CTest)
option(ATHELAS_ENABLE_UNIT_TESTS "Enable unit tests" OFF)
option(ATHELAS_ENABLE_INTEGRATION_TESTS "Enable integration tests" OFF) # Future
option(ATHELAS_ENABLE_REGRESSION_TESTS "Enable regression tests" OFF)   # Future

set( MACHINE "LINUX" CACHE STRING "Set OS")
message( "Building On ${MACHINE} ")

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

include_directories(src)
include_directories("src/geometry")
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
include_directories("src/state")
include_directories("src/timestepper")

include_directories(athelas PUBLIC
  "${PROJECT_BINARY_DIR}"
  )

# --- Source Files ---
set( ATHELAS_SOURCES 
  "src/pgen/problem_in.cpp"
  "src/basis/polynomial_basis.cpp"
  "src/bc/boundary_conditions.cpp"
  "src/eos/eos_ideal.cpp"
  "src/fluid/fluid_utilities.cpp"
  "src/radiation/rad_utilities.cpp"
  "src/radiation/rad_discretization.cpp"
  "src/fluid/fluid_discretization.cpp"
  "src/geometry/grid.cpp"
  "src/io/io.cpp"
  "src/linalg/linear_algebra.cpp"
  "src/quadrature/quadrature.cpp"
  "src/limiters/slope_limiter.cpp"
  "src/limiters/bound_enforcing_limiter.cpp"
  "src/limiters/slope_limiter_utilities.cpp"
  "src/state/state.cpp"
  "src/timestepper/timestepper.cpp" )

add_library(libathelas ${ATHELAS_SOURCES})
set_target_properties(libathelas
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ##RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    PREFIX ""
    SUFFIX ".so"
)

# Check for FindOpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OPEN MP ${OpenMP_CXX_SPEC_DATE}")
    if((OpenMP_CXX_SPEC_DATE GREATER_EQUAL "201511"))
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOMP45")
        message(STATUS "Using OpenMP 4.5 directives")
    else()
        message(STATUS "Not using OpenMP 4.5 directives: ${OpenMP_C_VERSION_MAJOR} ${OpenMP_C_VERSION_MINOR}")
    endif()
    set(ENABLE_OPENMP ON)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Allow Kokkos to use OpenMP as execution space.")
endif(OPENMP_FOUND)

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
    find_package(Kokkos 3)
    if (NOT Kokkos_FOUND)
      unset(ATHELAS_IMPORT_KOKKOS CACHE)
      message(FATAL_ERROR "Could not find external Kokkos. Consider importing a Kokkos installation into your environment or disabling external Kokkos with e.g. -DATHELAS_IMPORT_KOKKOS=OFF")
    endif()
  else()
    if (EXISTS ${Kokkos_ROOT}/CMakeLists.txt)
      add_subdirectory(${Kokkos_ROOT} Kokkos)
      message(STATUS "Using Kokkos source from Kokkos_ROOT=${Kokkos_ROOT}")
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/Kokkos/CMakeLists.txt)
      add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/Kokkos ${PROJECT_SOURCE_DIR}/build/Kokkos)
      message(STATUS "Using Kokkos source from Athelas submodule at ${CMAKE_CURRENT_SOURCE_DIR}/external/Kokkos")
    else()
      message(FATAL_ERROR "Could not find Kokkos source. Consider running `git submodule update --init`, providing the path to a Kokkos source directory with Kokkos_ROOT, or setting ATHELAS_IMPORT_KOKKOS=ON to link to an external Kokkos installation.")
    endif()
  endif()
endif()


# HDF5
find_package ( HDF5 REQUIRED COMPONENTS CXX HL )
message(STATUS "HDF5_LIBRARIES = ${HDF5_CXX_LIBRARIES}")
include_directories(${HDF5_INCLUDE_DIRS}) # seems to be needed for compilation on github actions...

# LAPACK
# ubuntu's openblas does not include the necessary lapacke headers...
if (${MACHINE} STREQUAL UBUNTU)
    set(LAPACK_LIBRARIES "/usr/lib/x86_64-linux-gnu/liblapacke.so.3;-lm;-ldl")
    message(STATUS "LAPACK_LIBRARIES = ${LAPACK_LIBRARIES}")
    target_link_libraries( libathelas ${LAPACK_LIBRARIES} )
  elseif( ${MACHINE} STREQUAL MACOS )
    set( LAPACK_LIBRARIES "/usr/local/Cellar/lapack/3.10.1/lib/liblapacke.3.dylib;-lm;-ldl;-llapack" )
    set( LAPACK_INCLUDE "/usr/local/Cellar/lapack/3.10.1/include/" )
    include_directories( ${LAPACK_INCLUDE} )
    #set( LAPACK_LIBRARIES "/usr/local/Cellar/openblas/0.3.20/lib/libopenblas.dylib;-lm;-ldl;-llapack" )
    message(STATUS "LAPACK_LIBRARIES = ${LAPACK_LIBRARIES}")
    target_link_libraries( libathelas ${LAPACK_LIBRARIES} )
else()
  find_package( LAPACK REQUIRED  )
  message(STATUS "LAPACKE_LIBRARIES = ${LAPACK_LIBRARIES}")
  include_directories(${LAPACK_INCLUDE_DIRS})
    target_link_libraries( libathelas ${LAPACK_LIBRARIES} )
endif()
include_directories(${LAPACK_INCLUDE_DIRS})

# toml++
include_directories("external/tomlplusplus")

# --- main ---

add_executable( main src/driver.cpp ${ATHELAS_SOURCES} )
set_target_properties(main
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
    #RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

# -- finalize ---
# yeah it's a mess but it works

target_link_libraries( libathelas Kokkos::kokkos )
target_link_libraries( libathelas ${HDF5_C_LIBRARIES} ${HDF5_LIBRARIES} ${LAPACK_LIBRARIES} )

target_compile_options( main PRIVATE -fPIC -O2 -Wall -std=c++17 -llapacke )
add_dependencies( main libathelas )

target_link_libraries( main ${HDF5_C_LIBRARIES} ${HDF5_LIBRARIES} ${LAPACK_LIBRARIES} Kokkos::kokkos )

if ( ATHELAS_ENABLE_UNIT_TESTS )
  message("\nConfiguring tests")
  enable_testing()
  add_subdirectory( test )
endif()
