#include "test_utils.hpp"
#include <catch2/catch_session.hpp>

int main(int argc, char *argv[]) {
  int result = 0;
  Kokkos::initialize();
  {
    result = Catch::Session().run(argc, argv);
  }
  Kokkos::finalize();
  return result;
}
