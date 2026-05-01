#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"

#include "quadrature.hpp"

#include <print>

TEST_CASE("Quadrature are constructed correctly", "[quadrature]") {

  int n_nodes = 3;
  std::vector<double> nodes = {0.0, 0.0, 0.0};
  std::vector<double> weights = {0.0, 0.0, 0.0};

  athelas::math::quadrature::lg_quadrature(n_nodes, nodes, weights);

  std::vector<double> nodes_ans = {-0.5 * std::sqrt(3.0 / 5.0), 0.0,
                                   0.5 * std::sqrt(3.0 / 5.0)};
  std::vector<double> weights_ans = {2.5 / 9.0, 4.0 / 9.0, 2.5 / 9.0};

  for (int i = 0; i < 3; ++i) {
    std::println("n nans {} {}", nodes[i], nodes_ans[i]);
    std::println("w wans {} {}", weights[i], weights_ans[i]);
    REQUIRE(soft_equal(nodes[i], nodes_ans[i], 1.0e-2));
    REQUIRE(soft_equal(weights[i], weights_ans[i], 1.0e-10));
  }
}
