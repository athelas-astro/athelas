#include <catch2/catch_session.hpp>

int main( int argc, char *argv[] ) {
  int result = 0;
  { result = Catch::Session().run(argc, argv); }
  return result;
}
