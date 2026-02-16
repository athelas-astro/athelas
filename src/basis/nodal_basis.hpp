#pragma once

#include <functional>

#include "basis/polynomial_basis.hpp"
#include "basic_types.hpp"
#include "kokkos_types.hpp"
#include "geometry/grid.hpp"

namespace athelas::basis {

class NodalBasis {
public:
  /**
   * @brief Constructor
   * @param uPF Primitive variables (nodal representation)
   * @param grid Grid structure
   * @param nN Number of nodes
   * @param nElements Number of elements
   * @param density_weight Include density in mass matrix
   */
  NodalBasis(const AthelasArray3D<double> uPF,
             GridStructure *grid, 
             const int nN,
             const int nElements, 
             const bool density_weight);

  /**
   * @brief Lagrange basis values L_j(eta_i)
   * @return phi_(ix, i_eta, j) where i_eta: 0=left face, 1..nNodes=GL nodes, nNodes+1=right face
   */
  [[nodiscard]] auto phi() const noexcept -> AthelasArray3D<double>;

  /**
   * @brief Derivative of Lagrange basis dL_j/deta(eta_i)
   * @return dphi_(ix, i_eta, j) - contains differentiation matrix at interior nodes
   */
  [[nodiscard]] auto dphi() const noexcept -> AthelasArray3D<double>;

  /** @brief Direct accessor for dphi */
  auto get_d_phi(const int ix, const int i_eta, const int k) const -> double;

  /**
   * @brief Diagonal mass matrix M_jj = w_j * rho_j * J_j * dr
   * @return mass_matrix_(ix, j)
   */
  [[nodiscard]] auto mass_matrix() const noexcept -> AthelasArray2D<double>;

  /** @brief Inverse mass matrix (diagonal) */
  [[nodiscard]] auto inv_mass_matrix() const noexcept -> AthelasArray2D<double>;

  /** @brief Polynomial order (nNodes - 1) */
  auto order() const noexcept -> int;

  /**
   * @brief Project a nodal basis onto a modal representation
   */
   void nodal_to_modal(
    AthelasArray3D<double> u_k,
    AthelasArray3D<double> ucf) const; 

  /**
   * @brief Project a modal basis onto a nodal representation
   */
   void modal_to_nodal(
    AthelasArray3D<double> ucf,
    AthelasArray3D<double> u_k) const; 

  // === Evaluation methods (API compatibility) ===

  /**
   * @brief Evaluate nodal representation at location i_eta
   * @details For interior nodes (i_eta in [1,nNodes]): returns U directly
   *          For faces (i_eta=0 or nNodes+1): computes via Lagrange extrapolation
   */
  auto basis_eval(AthelasArray3D<double> U, const int ix, const int q,
                  const int i_eta) const -> double;

  auto basis_eval(AthelasArray2D<double> U, const int ix, const int q,
                  const int i_eta) const -> double;

  auto basis_eval(AthelasArray1D<double> U, const int ix,
                  const int i_eta) const -> double;

  /**
   * @brief Project nodal function to representation (trivial for nodal DG)
   * @details Just evaluates nodal_func at nodes
   */
  void project_nodal_to_modal_all_cells(
      AthelasArray3D<double> uCF, 
      AthelasArray3D<double> uPF, 
      GridStructure *grid,
      int q, 
      const std::function<double(double, int, int)> &nodal_func) const;

  // === Nodal-specific methods ===

  /**
   * @brief Get differentiation matrix D_ij = dL_j/deta(eta_i)
   * @details Used for computing du/deta|_i = sum_j D_ij u_j
   */
  [[nodiscard]] auto differentiation_matrix() const noexcept 
      -> AthelasArray2D<double>;

  template <typename ViewType>
  static auto lagrange_polynomial(int j, double xi, const ViewType nodes)
      -> double {
    const int n = static_cast<int>(nodes.size());
    double result = 1.0;

    for (int m = 0; m < n; m++) {
      if (m != j) {
        result *= (xi - nodes(m)) / (nodes(j) - nodes(m));
      }
    }

    return result;
  }

private:
  int nX_;
  int nNodes_;
  bool density_weight_;

  AthelasArray1D<double> nodes_;
  AthelasArray1D<double> weights_;
  AthelasArray3D<double> phi_;
  AthelasArray3D<double> dphi_;  // Basis derivatives: dphi_(ix, i_eta, j) = dL_j/deta(eta_i)
  AthelasArray2D<double> mass_matrix_;
  AthelasArray2D<double> inv_mass_matrix_;
  AthelasArray2D<double> differentiation_matrix_;
  AthelasArray2D<double> vandermonde_;
  AthelasArray2D<double> inv_vandermonde_;

  /** @brief Initialize all basis quantities */
  void initialize_basis(AthelasArray3D<double> uPF,
                       const GridStructure *grid);

  /** @brief Build differentiation matrix */
  void build_differentiation_matrix();

  /** @brief Build Vandermonde and inverse Vandermonde matrices */
  void build_vandermonde_matrices();


  /** @brief Compute diagonal mass matrix. */
  void compute_mass_matrix(const GridStructure *grid);

  /** @brief Evaluate Lagrange polynomial L_j at point xi */
  static auto lagrange_polynomial(int j, double xi,
                                 AthelasArray1D<double> nodes) -> double;

  /** @brief Evaluate derivative of Lagrange polynomial dL_j/dxi at point xi */
  static auto d_lagrange_polynomial(int j, double xi,
                                   AthelasArray1D<double> nodes) -> double;

  /** @brief Fill guard cells (mirror interior) */
  void fill_guard_cells(const GridStructure *grid);
};

} // namespace athelas::basis
