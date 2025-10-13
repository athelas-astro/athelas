# unit testing

Include( FetchContent )

message( STATUS "Fetching Catch2 as needed" )
FetchContent_Declare( 
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.9.0 # or later
  )
FetchContent_MakeAvailable(Catch2)

add_executable( unit_tests
  test_utils.hpp
  unit_tests_main.cpp

  test_catch_integration.cpp
  test_quadrature.cpp
  ../../src/quadrature/quadrature.cpp
  test_parser.cpp
  test_paczynski.cpp
  test_root_finders.cpp
  )

target_link_libraries( unit_tests
  PRIVATE
    Catch2::Catch2
    kokkos
)

# add_subdirectory( basis )

include( Catch )
catch_discover_tests( unit_tests )
