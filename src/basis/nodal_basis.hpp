#pragma once

#include <functional>

#include "basic_types.hpp"
#include "geometry/mesh.hpp"
#include "kokkos_types.hpp"

namespace athelas::basis {

class NodalBasis {
 public:
  /**
   * @brief Constructor
   * @param derived Primitive variables (nodal representation)
   * @param mesh Grid structure
   * @param nN Number of nodes
   * @param nElements Number of elements
   */
  NodalBasis(const AthelasArray3D<double> derived, Mesh *mesh, const int nN,
             const int nElements);

  /**
   * @brief Lagrange basis values L_j(eta_i)
   * @return phi_(ix, i_eta, j) where i_eta: 0=left face, 1..nNodes=GL nodes,
   * nNodes+1=right face
   */
  [[nodiscard]] auto phi() const noexcept -> AthelasArray3D<double>;

  /**
   * @brief Derivative of Lagrange basis dL_j/deta(eta_i)
   * @return dphi_(ix, i_eta, j) - contains differentiation matrix at interior
   * nodes
   */
  [[nodiscard]] auto dphi() const noexcept -> AthelasArray3D<double>;

  /** @brief Direct accessor for dphi */
  auto get_d_phi(const int ix, const int i_eta, const int k) const -> double;

  /**
   * @brief Diagonal reference-mass matrix M_jj = w_j * dm/deta_j
   * @return mass_matrix_(ix, j)
   */
  [[nodiscard]] auto mass_matrix() const noexcept -> AthelasArray2D<double>;

  /** @brief Inverse mass matrix (diagonal) */
  [[nodiscard]] auto inv_mass_matrix() const noexcept -> AthelasArray2D<double>;

  /** @brief Polynomial order (nNodes - 1) */
  [[nodiscard]] auto order() const noexcept -> int;

  /**
   * @brief Project a nodal basis onto a modal representation
   * The IndexRange is for the variables in evolved that we are mapping.
   * The modal vector u_k_ loops from 0.
   */
  void nodal_to_modal(AthelasArray3D<double> u_k,
                      AthelasArray3D<double> evolved,
                      const IndexRange &vb) const;

  /**
   * @brief Project a modal basis onto a nodal representation
   * The IndexRange is for the variables in evolved that we are mapping.
   * The modal vector u_k_ loops from 0.
   */
  void modal_to_nodal(AthelasArray3D<double> evolved,
                      AthelasArray3D<double> u_k, const IndexRange &vb) const;

  // --- Evaluation methods (back compatibility) ---

  /**
   * @brief Evaluate nodal representation at location i_eta
   * @details For interior nodes (i_eta in [1,nNodes]): returns U directly
   *          For faces (i_eta=0 or nNodes+1): computes via Lagrange
   * extrapolation
   */
  auto basis_eval(AthelasArray3D<double> U, const int ix, const int q,
                  const int i_eta) const -> double;

  auto basis_eval(AthelasArray2D<double> U, const int ix, const int q,
                  const int i_eta) const -> double;

  auto basis_eval(AthelasArray1D<double> U, const int ix, const int i_eta) const
      -> double;

  /**
   * @brief Project nodal function to representation (trivial for nodal DG)
   * @details Just evaluates nodal_func at nodes
   */
  void project_nodal_to_modal_all_cells(
      AthelasArray3D<double> evolved, AthelasArray3D<double> derived,
      Mesh *mesh, int q,
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

  /**
   * @brief Persson-Peraire modal-decay smoothness indicator for variable `v`
   * in cell `ix`: the fraction of the L2 (modal) energy carried by the highest
   * Legendre mode. ~O(h^{2k}) where the field is smooth, O(1) across a
   * discontinuity. Cheap -- one inverse-Vandermonde row dotted with the nodal
   * data, plus the nodal energy (Parseval, exact for the orthogonal Legendre
   * modes under GL quadrature). Returns 1 for nNodes_ < 2 (no resolvable high
   * mode), so callers fall back to full upwinding there.
   */
  [[nodiscard]] KOKKOS_INLINE_FUNCTION auto
  modal_decay_indicator(const AthelasArray3D<double> U, const int ix,
                        const int v) const -> double {
    double c_top = 0.0;
    double energy = 0.0;
    for (int q = 0; q < nNodes_; ++q) {
      const double u = U(ix, q, v);
      c_top += inv_vandermonde_(nNodes_ - 1, q) * u;
      energy += weights_(q) * u * u;
    }
    const double top_norm = 1.0 / (2.0 * nNodes_ - 1.0); // ||P_k||^2
    return (c_top * c_top * top_norm) / (energy + 1.0e-300);
  }

 private:
  int nX_;
  int nNodes_;

  AthelasArray1D<double> nodes_;
  AthelasArray1D<double> weights_;
  AthelasArray3D<double> phi_;
  AthelasArray3D<double>
      dphi_; // Basis derivatives: dphi_(ix, i_eta, j) = dL_j/deta(eta_i)
  AthelasArray2D<double> mass_matrix_;
  AthelasArray2D<double> inv_mass_matrix_;
  AthelasArray2D<double> differentiation_matrix_;
  AthelasArray2D<double> vandermonde_;
  AthelasArray2D<double> inv_vandermonde_;

  /** @brief Initialize all basis quantities */
  void initialize_basis(AthelasArray3D<double> derived, const Mesh *mesh);

  /** @brief Build differentiation matrix */
  void build_differentiation_matrix();

  /** @brief Build Vandermonde and inverse Vandermonde matrices */
  void build_vandermonde_matrices();

  /** @brief Compute diagonal mass matrix. */
  void compute_mass_matrix(const Mesh *mesh);

  /** @brief Evaluate Lagrange polynomial L_j at point xi */
  static auto lagrange_polynomial(int j, double xi,
                                  AthelasArray1D<double> nodes) -> double;

  /** @brief Evaluate derivative of Lagrange polynomial dL_j/dxi at point xi */
  static auto d_lagrange_polynomial(int j, double xi,
                                    AthelasArray1D<double> nodes) -> double;

  /** @brief Fill guard cells (mirror interior) */
  void fill_guard_cells(const Mesh *mesh);
};

template <Interface Face>
KOKKOS_FUNCTION auto basis_eval(AthelasArray3D<double> phi,
                                AthelasArray3D<double> u, const int i,
                                const int v) -> double {
  static const int nq = static_cast<int>(phi.extent(2));
  static const IndexRange qb(nq);

  // offset accounts for the design that primitive/aux variables have interface
  // storage while the evolved/conserved variables do not. It ensures
  // that the indexing is over interior collocation points.
  static const int nq_p_i = static_cast<int>(phi.extent(2)); // size of nodes
  const int nq_u = static_cast<int>(u.extent(1));
  const int offset = (nq_u == nq_p_i) ? 0 : 1;
  if constexpr (Face == Interface::Left) {
    double result = 0.0;
    for (int p = 0; p < nq; p++) {
      result += phi(i, 0, p) * u(i, p + offset, v);
    }
    return result;
  }

  if constexpr (Face == Interface::Right) {
    double result = 0.0;
    for (int p = 0; p < nq; p++) {
      result += phi(i, nq + 1, p) * u(i, p + offset, v);
    }
    return result;
  }
}

auto d_legendre_n2(int n, int q, double x) -> double;
auto d_legendre_n3(const int n, const int q, const double x) -> double;

} // namespace athelas::basis
