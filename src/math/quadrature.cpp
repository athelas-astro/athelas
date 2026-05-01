#include <cmath>
#include <vector>

#include "math/linear_algebra.hpp"
#include "math/quadrature.hpp"

namespace athelas::math::quadrature {

/**
 * @brief Computes the Jacobi matrix for legendre-Gauss quadrature rule
 *
 * @details Constructs the symmetric tridiagonal Jacobi matrix
 *          needed for legendre-Gauss quadrature.
 *
 * @param m Number of quadrature nodes (matrix dimension)
 * @param aj Output array for matrix diagonal elements
 * @param bj Output array for matrix subdiagonal elements
 * @return double The zero-th moment (zemu) needed for weight computation
 */
auto jacobi_matrix(int m, std::vector<double> &aj, std::vector<double> &bj)
    -> double {

  double ab = NAN;
  double zemu = NAN;
  double abi = NAN;
  double abj = NAN;

  ab = 0.0;
  zemu = 2.0 / (ab + 1.0);
  for (int i = 0; i < m; i++) {
    aj[i] = 0.0;
  }

  for (int i = 1; i <= m; i++) {
    abi = i + ab * (i % 2);
    abj = 2 * i + ab;
    bj[i - 1] = sqrt(abi * abi / (abj * abj - 1.0));
  }

  return zemu;
}

/**
 * @brief Generates a legendre-Gauss quadrature rule with specified number of
 *        points
 *
 * @details This function computes a complete legendre-Gauss quadrature rule
 *          with m points. It generates both the quadrature nodes (abscissas)
 *          and weights for accurate integration of polynomial functions.
 *
 *          The resulting quadrature rule is optimal for integrating
 *          polynomial functions up to degree 2m-1.
 *
 * @param m Number of quadrature points (must be positive)
 * @param nodes Output array for quadrature nodes (abscissas)
 * @param weights Output array for quadrature weights
 */
void lg_quadrature(int m, std::vector<double> &nodes,
                   std::vector<double> &weights) {
  std::vector<double> aj(m);
  std::vector<double> bj(m);

  double zemu = NAN;

  //  Get the Jacobi matrix and zero-th moment.
  zemu = jacobi_matrix(m, aj, bj);

  // Nodes and Weights
  for (int i = 0; i < m; i++) {
    nodes[i] = aj[i];
  }

  weights[0] = sqrt(zemu);
  for (int i = 1; i < m; i++) {
    weights[i] = 0.0;
  }

  // --- Diagonalize the Jacobi matrix. ---
  math::linalg::tri_sym_diag(m, nodes, bj, weights); // imtqlx

  for (int i = 0; i < m; i++) {
    weights[i] = weights[i] * weights[i];

    // Shift to interval [-0.5, 0.5]
    weights[i] *= 0.5;
    nodes[i] *= 0.5;
  }
}

} // namespace athelas::math::quadrature
