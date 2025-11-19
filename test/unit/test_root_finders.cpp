#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"

#include "solvers/root_finders.hpp"

using athelas::root_finders::AbsoluteError,
    athelas::root_finders::RelativeError, athelas::root_finders::HybridError,
    athelas::root_finders::RootFinder, athelas::root_finders::NewtonAlgorithm,
    athelas::root_finders::AANewtonAlgorithm,
    athelas::root_finders::FixedPointAlgorithm,
    athelas::root_finders::AAFixedPointAlgorithm,
    athelas::root_finders::BisectionAlgorithm,
    athelas::root_finders::RegulaFalsiAlgorithm,
    athelas::root_finders::ToleranceConfig;

constexpr double sqrt2 = std::numbers::sqrt2;
constexpr double cos_fixed_point = 0.7390851332151607;

constexpr double tol = 1e-10;

TEST_CASE("Newton algorithm: x^2 - 2", "[newton]") {
  auto f = [](double x) { return x * x - 2.0; };
  auto df = [](double x) { return 2.0 * x; };

  RootFinder<double, NewtonAlgorithm<double>> solver;
  solver.set_tolerance(1e-14, 1e-14).set_max_iterations(100);

  double root = solver.solve(f, df, 2.0);
  REQUIRE(soft_equal(root, sqrt2, tol));
}

TEST_CASE("Anderson accelerated Newton algorithm: x^2 - 2", "[aa_newton]") {
  auto f = [](double x) { return x * x - 2.0; };
  auto df = [](double x) { return 2.0 * x; };

  RootFinder<double, AANewtonAlgorithm<double>> solver;
  solver.set_tolerance(1e-14, 1e-14).set_max_iterations(100);

  double root = solver.solve(f, df, 2.0);
  REQUIRE(soft_equal(root, sqrt2, tol));
}

TEST_CASE("Fixed point iteration for x = cos(x)", "[fixed_point]") {
  auto g = [](double x) { return std::cos(x); };

  RootFinder<double, FixedPointAlgorithm<double>> solver;
  solver.set_tolerance(1e-12, 1e-12).set_max_iterations(100);

  double root = solver.solve(g, 2.0);
  REQUIRE(soft_equal(root, cos_fixed_point, tol));
}

TEST_CASE("Anderson accelerated fixed point iteration for x = cos(x)",
          "[aa_fixed_point]") {
  auto g = [](double x) { return std::cos(x); };

  RootFinder<double, AAFixedPointAlgorithm<double>> solver;
  solver.set_tolerance(1e-12, 1e-12).set_max_iterations(100);

  double root = solver.solve(g, 2.0);
  REQUIRE(soft_equal(root, cos_fixed_point, tol));
}

TEST_CASE("Bisection algorithm: x^2 - 2", "[bisection]") {
  auto f = [](double x) { return x * x - 2.0; };

  BisectionAlgorithm<double> bisect;
  ToleranceConfig<double> config{1e-14, 1e-14, 100};

  // Bracket [1, 2] contains sqrt(2)
  double root = bisect(f, 1.0, 2.0, config);
  REQUIRE(soft_equal(root, sqrt2, tol));
}

TEST_CASE("Bisection algorithm: x - 3", "[bisection]") {
  auto f = [](double x) { return x - 3.0; };

  BisectionAlgorithm<double> bisect;
  ToleranceConfig<double> config{1e-12, 1e-12, 100};

  // Bracket [0, 5] contains root at 3
  double root = bisect(f, 0.0, 5.0, config);
  REQUIRE(soft_equal(root, 3.0, tol));
}

TEST_CASE("Bisection algorithm: x^3 - 8", "[bisection]") {
  auto f = [](double x) { return x * x * x - 8.0; };

  BisectionAlgorithm<double> bisect;
  ToleranceConfig<double> config{1e-12, 1e-12, 100};

  // Bracket [0, 3] contains root at 2
  double root = bisect(f, 0.0, 3.0, config);
  REQUIRE(soft_equal(root, 2.0, tol));
}

TEST_CASE("Bisection algorithm: reversed bracket", "[bisection]") {
  auto f = [](double x) { return x * x - 2.0; };

  BisectionAlgorithm<double> bisect;
  ToleranceConfig<double> config{1e-14, 1e-14, 100};

  // Bracket [2, 1] is reversed, should still work
  double root = bisect(f, 2.0, 1.0, config);
  REQUIRE(soft_equal(root, sqrt2, tol));
}

TEST_CASE("Regula Falsi algorithm: x^2 - 2", "[regula_falsi]") {
  auto f = [](double x) { return x * x - 2.0; };

  RegulaFalsiAlgorithm<double> regula_falsi;
  ToleranceConfig<double> config{1e-14, 1e-14, 100};

  // Bracket [1, 2] contains sqrt(2)
  double root = regula_falsi(f, 1.0, 2.0, 1.5, config);
  REQUIRE(soft_equal(root, sqrt2, tol));
}

TEST_CASE("Regula Falsi algorithm: x - 3", "[regula_falsi]") {
  auto f = [](double x) { return x - 3.0; };

  RegulaFalsiAlgorithm<double> regula_falsi;
  ToleranceConfig<double> config{1e-12, 1e-12, 100};

  // Bracket [0, 5] contains root at 3
  double root = regula_falsi(f, 0.0, 5.0, 0.3, config);
  REQUIRE(soft_equal(root, 3.0, tol));
}

TEST_CASE("Regula Falsi algorithm: reversed bracket", "[regula_falsi]") {
  auto f = [](double x) { return x * x - 2.0; };

  RegulaFalsiAlgorithm<double> regula_falsi;
  ToleranceConfig<double> config{1e-14, 1e-14, 100};

  // Bracket [2, 1] is reversed, should still work
  double root = regula_falsi(f, 2.0, 1.0, 0.0, config);
  REQUIRE(soft_equal(root, sqrt2, tol));
}
