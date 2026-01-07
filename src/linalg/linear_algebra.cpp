/**
 * @file linear_algebra.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Basic linear algebra functions.
 *
 * @details Linear algebra routines for quadrature and limiters.
 *          - tri_sym_diag
 *          - invert_matrix
 */

#include <cstddef>
#include <vector>

#include <Eigen/Dense>
#include <Kokkos_Core.hpp>

#include "linalg/linear_algebra.hpp"
#include "utils/error.hpp"

namespace athelas {

/**
 * @brief Diagonalizes a symmetric tridiagonal matrix using Eigen.
 */
void tri_sym_diag(int n, std::vector<double> &d, std::vector<double> &e,
                  std::vector<double> &array) {

  assert(n > 0 && "Matrix dim must be > 0");

  // trivial
  if (n == 1) {
    return;
  }

  // Create tridiagonal matrix
  Eigen::MatrixXd tri_matrix = Eigen::MatrixXd::Zero(n, n);

  // Fill diagonal
  for (int i = 0; i < n; i++) {
    tri_matrix(i, i) = d[i];
  }

  // Fill super/sub diagonals
  for (int i = 0; i < n - 1; i++) {
    tri_matrix(i, i + 1) = e[i];
    tri_matrix(i + 1, i) = e[i];
  }

  // Compute eigendecomposition
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(tri_matrix);

  if (solver.info() != Eigen::Success) {
    throw_athelas_error("Eigendecomposition failed in tri_sym_diag");
  }

  // Get eigenvalues and eigenvectors
  const Eigen::VectorXd &eigenvalues = solver.eigenvalues();
  const Eigen::MatrixXd &eigenvectors = solver.eigenvectors();

  // Update d with eigenvalues (Eigen returns them in ascending order)
  for (int i = 0; i < n; i++) {
    d[i] = eigenvalues(i);
  }

  // Matrix multiply eigenvectors' * array. Only array[0] is nonzero initially.
  double const k = array[0];
  for (int i = 0; i < n; i++) {
    // First row of eigenvectors matrix (corresponding to first element of
    // array)
    array[i] = k * eigenvectors(0, i);
  }
}
/**
 * @brief Use Eigen to invert a matrix M using LU factorization.
 **/
void invert_matrix(std::vector<double> &M, int n) {
  // Map the std::vector to an Eigen matrix (column-major order)
  Eigen::Map<Eigen::MatrixXd> matrix(M.data(), n, n);

  // Compute the inverse using LU decomposition
  Eigen::MatrixXd inverse = matrix.inverse();

  // Check if the matrix is invertible by verifying determinant is non-zero
  if (std::abs(matrix.determinant()) < std::numeric_limits<double>::epsilon()) {
    throw_athelas_error(" ! Issue occurred in matrix inversion.");
  }

  // Copy the result back to the original vector
  matrix = inverse;
}

} // namespace athelas
