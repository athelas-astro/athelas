#pragma once

#include <vector>

namespace athelas::math::quadrature {
auto jacobi_matrix(int m, std::vector<double> &aj, std::vector<double> &bj)
    -> double;
void lg_quadrature(int m, std::vector<double> &nodes,
                   std::vector<double> &weights);
} // namespace athelas::math::quadrature
