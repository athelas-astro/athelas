#include "nodal_basis.hpp"
#include "test_utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cmath>

TEST_CASE("Legendre Derivatives - q=0 (Standard Polynomials)", "[legendre]") {
  // P_2(0.5) = 0.5 * (3 * 0.25 - 1) = -0.125
  REQUIRE(soft_equal(athelas::basis::d_legendre_n3(2, 0, 0.5), -0.125));

  // P_3(1.0) = 1.0
  REQUIRE(soft_equal(athelas::basis::d_legendre_n3(3, 0, 1.0), 1.0));
}

TEST_CASE("Legendre Derivatives - q=1 (First Derivative)", "[legendre]") {
  REQUIRE(soft_equal(athelas::basis::d_legendre_n3(1, 1, 0.5), 1.0));

  // P_2'(x) = 3x -> P_2'(0.5) = 1.5
  REQUIRE(soft_equal(athelas::basis::d_legendre_n3(2, 1, 0.5), 1.5));

  // P_3'(x) = 1.5 * (5x^2 - 1) -> P_3'(1.0) = 6.0
  REQUIRE(soft_equal(athelas::basis::d_legendre_n3(3, 1, 1.0), 6.0));
}

TEST_CASE("Legendre Derivatives - q=2 (Second Derivative)", "[legendre]") {
  // P_2''(x) = 3 (Constant)
  REQUIRE(soft_equal(athelas::basis::d_legendre_n3(2, 2, -0.5), 3.0));

  // P_3''(x) = 15x -> P_3''(0.5) = 7.5
  REQUIRE(soft_equal(athelas::basis::d_legendre_n3(3, 2, 0.5), 7.5));
}

TEST_CASE("Legendre Derivatives - q=3 (Third Derivative)", "[legendre]") {
  // P_3'''(x) = 15 (Constant)
  REQUIRE(soft_equal(athelas::basis::d_legendre_n3(3, 3, 0.123), 15.0));

  // P_2'''(x) = 0 (Order > Degree)
  REQUIRE(soft_equal(athelas::basis::d_legendre_n3(2, 3, 0.5), 0.0));
}

TEST_CASE("Legendre Derivatives - Boundary Property at x=1", "[legendre]") {
  // The k-th derivative of P_n at x=1 is (n+k)! / (2^k * k! * (n-k)!)
  // For n=4, q=2: (4+2)! / (2^2 * 2! * (4-2)!) = 720 / (4 * 2 * 2) = 720 / 16 =
  // 45
  REQUIRE(soft_equal(athelas::basis::d_legendre_n3(4, 2, 1.0), 45.0));

  // For n=5, q=1: (5+1)! / (2^1 * 1! * 4!) = 720 / (2 * 24) = 15
  // P_5'(1) = 0.5 * n * (n+1) = 0.5 * 5 * 6 = 15. (Standard identity)
  REQUIRE(soft_equal(athelas::basis::d_legendre_n3(5, 1, 1.0), 15.0));
}
