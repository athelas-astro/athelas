#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "basic_types.hpp"
#include "test_utils.hpp"

#include "math/difference.hpp"

using namespace athelas::math::difference;
using athelas::DiffScheme;

// Some simple finite differences
TEST_CASE("Finite Difference: Scalar Functions", "[math][derivative]") {
  const double h = 1e-6;
  const double tol = 1e-5;

  SECTION("Polynomial: f(x) = x^3 -> f'(x) = 3x^2") {
    auto cube = [](double x) { return x * x * x; };
    const double x = 2.0;
    const double expected = 3.0 * x * x; // 12

    double df_forward = finite_difference<DiffScheme::Forward>(h, cube, x);
    double df_central = finite_difference<DiffScheme::Central>(h, cube, x);

    REQUIRE(soft_equal(df_forward, expected, tol));
    REQUIRE(soft_equal(df_central, expected, tol / 10.0));
  }

  SECTION("Transcendental: f(x) = sin(x) -> f'(x) = cos(x)") {
    auto f = [](double x) { return std::sin(x); };
    const double x = 0.0; // cos(0) = 1.0
    const double expected = std::cos(0.0);

    double ans = finite_difference<DiffScheme::Central>(1.0e-5, f, x);
    REQUIRE(soft_equal(ans, expected, 1.0e-10));
  }
}

TEST_CASE("Finite Difference: Variadic Arguments", "[math][variadic]") {
  // Testing f(x, a, b) = a*x + b -> df/dx = a
  auto linear = [](double x, double a, double b) { return a * x + b; };
  const double h = 1e-7;
  const double tol = 1.0e-7;
  const double expected = 5.0;

  double result =
      finite_difference<DiffScheme::Forward>(h, linear, 10.0, 5.0, 2.0);

  REQUIRE(soft_equal(result, expected, tol));
}
