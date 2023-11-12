# unit testing

Include( FetchContent )

message( STATUS "Fetching Catch2 as needed" )
FetchContent_Declare( 
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.0.1 # or later
  )
FetchContent_MakeAvailable(Catch2)

add_executable( unit_tests
  test_utils.hpp
  unit_tests_main.cpp

  test_catch_integration.cpp
  )

target_link_libraries( unit_tests
  PRIVATE
    Catch2::Catch2
)

# add_subdirectory( basis )

include( Catch )
catch_discover_tests( unit_tests )
