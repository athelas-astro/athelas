/**
 * @file polynomial_basis.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Core polynomial basis functions
 *
 * @details Provides means to construct and evaluate bases
 *            - legendre
 *            - taylor
 *
 * TODO(astrobarker): [GPU] probably form basis on host + copy
 * TODO(astrobarker): need center of mass for some probs?
 * TODO(astrobarker): derivative matrix
 */

#include <cmath>
#include <cstdlib>
#include <print>

#include "Kokkos_Macros.hpp"
#include "basis/polynomial_basis.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"
#include "utils/error.hpp"

namespace athelas::basis {

/**
 * Constructor creates necessary matrices and bases, etc.
 * This has to be called after the problem is initialized.
 **/
ModalBasis::ModalBasis(poly_basis basis, const AthelasArray3D<double> uPF,
                       GridStructure *grid, const int pOrder, const int nN,
                       const int nElements, const bool density_weight)
    : nX_(nElements), order_(pOrder), nNodes_(nN),
      mSize_((nN) * (nN + 2) * (nElements + 2)),
      density_weight_(density_weight),
      mass_matrix_("MassMatrix", nElements + 2, pOrder),
      inv_mass_matrix_("Inverse MassMatrix", nElements + 2, pOrder),
      phi_("phi_", nElements + 2, nN + 2, pOrder),
      dphi_("dphi_", nElements + 2, nN + 2, pOrder) {
  // --- Compute grid quantities ---
  grid->compute_mass(uPF);
  grid->compute_mass_r(uPF); // Weird place for this to be but works
  grid->compute_center_of_mass(uPF);

  if (basis == poly_basis::legendre) {
    func_ = legendre;
    dfunc_ = d_legendre;
  } else if (basis == poly_basis::taylor) {
    func_ = taylor;
    dfunc_ = d_taylor;
  } else {
    THROW_ATHELAS_ERROR(" ! Bad behavior in ModalBasis constructor !");
  }

  initialize_basis(uPF, grid);
}

/* --- taylor Methods --- */

/**
 * Return taylor polynomial of given order
 *
 * Parameters:
 * -----------
 * eta   : coordinate
 * eta_c : center of mass
 **/
auto ModalBasis::taylor(const int order, const double eta, const double eta_c)
    -> double {

  if (order < 0) {
    THROW_ATHELAS_ERROR(
        "! polynomial basis :: Please enter a valid polynomial order.");
  }

  // Handle constant and linear terms separately -- no need to exponentiate.
  if (order == 0) {
    return 1.0;
  }
  if (order == 1) {
    return eta - eta_c;
  }
  if (order > 1) {
    return std::pow(eta - eta_c, order);
  }
  return 0.0; // should not be reached.
}

/**
 * Return derivative of taylor polynomial of given order
 *
 * Parameters:
 * -----------
 * eta : coordinate
 * eta_c: center of mass
 **/
auto ModalBasis::d_taylor(const int order, const double eta, const double eta_c)
    -> double {

  if (order < 0) {
    THROW_ATHELAS_ERROR(
        " ! polynomial basis :: Please enter a valid polynomial order.");
  }

  // Handle first few terms separately -- no need to call std::pow
  if (order == 0) {
    return 0.0;
  }
  if (order == 1) {
    return 1.0;
  }
  if (order == 2) {
    return 2 * (eta - eta_c);
  }
  if (order > 2) {
    return (order)*std::pow(eta - eta_c, order - 1);
  }
  return 0.0; // should not be reached.
}

/* --- legendre Methods --- */
/* TODO: Make sure that x_c offset for legendre works with COM != 0 */

auto ModalBasis::legendre(const int n, const double x, const double x_c)
    -> double {
  return legendre(n, x - x_c);
}

auto ModalBasis::d_legendre(const int n, double x, const double x_c) -> double {
  return d_legendre(n, x - x_c);
}

// legendre polynomials
auto ModalBasis::legendre(const int n, const double x) -> double {
  return (n == 0)   ? 1.0
         : (n == 1) ? x
                    : (((2 * n) - 1) * x * legendre(n - 1, x) -
                       (n - 1) * legendre(n - 2, x)) /
                          n;
}

// Derivative of legendre polynomials
auto ModalBasis::d_legendre(const int order, const double x) -> double {

  double dPn = 0.0; // P_n

  for (int i = 0; i < order; i++) {
    dPn = (i + 1) * legendre(i, x) + x * dPn;
  }

  return dPn;
}

auto ModalBasis::d_legendre_n(const int poly_order, const int deriv_order,
                              const double x) -> double {
  if (deriv_order == 0) {
    return legendre(poly_order, x);
  }
  if (poly_order < deriv_order) {
    return 0.0;
  }
  if (deriv_order == 1) {
    return d_legendre(poly_order, x);
  }
  return (poly_order * d_legendre_n(poly_order - 1, deriv_order - 1, x)) +
         (x * d_legendre_n(poly_order - 1, deriv_order, x));
}

/* TODO: the following 2 inner product functions need to be cleaned */

/**
 * Lagrangian inner product of functions f and g
 * Used in orthogonalization.
 * Computes < Psi_m, phi__n >
 * <f,g> = \sum_q \rho_q f_Q g_q j^0 w_q
 * TODO: Make inner_product functions cleaner????
 **/
auto ModalBasis::inner_product(const int m, const int n, const int ix,
                               const double eta_c,
                               const AthelasArray3D<double> uPF,
                               const GridStructure *grid) const -> double {
  const auto weights = grid->weights();
  const auto nodes = grid->nodes();
  const auto dr = grid->widths();
  const auto sqrt_gm = grid->sqrt_gm();

  double result = 0.0;
  for (int iN = 0; iN < nNodes_; iN++) {
    // include rho in integrand if necessary
    const double rho = density_weight_ ? uPF(ix, iN + 1, vars::prim::Rho) : 1.0;
    const double eta_q = nodes(iN);
    result += func_(n, eta_q, eta_c) * phi_(ix, iN + 1, m) * weights(iN) * rho *
              dr(ix) * sqrt_gm(ix, iN + 1);
  }

  return result;
}

/**
 * Lagrangian inner product of functions f and g
 * Used in orthogonalization.
 * Computes < phi__m, phi__n >
 * <f,g> = \sum_q \rho_q f_q g_q j^0 w_q
 **/
auto ModalBasis::inner_product(const int n, const int ix,
                               const double /*eta_c*/,
                               const AthelasArray3D<double> uPF,
                               const GridStructure *grid) const -> double {
  const auto weights = grid->weights();
  const auto nodes = grid->nodes();
  const auto dr = grid->widths();
  const auto sqrt_gm = grid->sqrt_gm();

  double result = 0.0;
  for (int iN = 0; iN < nNodes_; iN++) {
    // include rho in integrand if necessary
    const double rho = density_weight_ ? uPF(ix, iN + 1, vars::prim::Rho) : 1.0;
    result += phi_(ix, iN + 1, n) * phi_(ix, iN + 1, n) * weights(iN) * rho *
              dr(ix) * sqrt_gm(ix, iN + 1);
  }

  return result;
}

// Gram-Schmidt orthogonalization of basis
auto ModalBasis::ortho(const int order, const int ix, const int i_eta,
                       const double eta, const double eta_c,
                       const AthelasArray3D<double> uPF,
                       const GridStructure *grid, bool const derivative_option)
    -> double {

  double result = 0.0;

  // TODO(astrobarker): Can this be cleaned up?
  if (not derivative_option) {
    result = func_(order, eta, eta_c);
  } else {
    result = dfunc_(order, eta, eta_c);
  }

  // if ( order == 0 ) return result;

  double phi_n = 0.0;
  for (int k = 0; k < order; k++) {
    const double numerator = inner_product(k, order, ix, eta_c, uPF, grid);
    const double denominator = inner_product(k, ix, eta_c, uPF, grid);
    // ? Can this be cleaned up?
    if (!derivative_option) {
      phi_n = phi_(ix, i_eta, k);
    }
    if (derivative_option) {
      phi_n = dphi_(ix, i_eta, k);
    }
    result -= (numerator / denominator) * phi_n;
  }

  return result;
}

/**
 * Pre-compute the orthogonal basis terms. phi_(ix,k,eta) will store
 * the expansion terms for each order k, stored at various points eta.
 * We store: (-0.5, {GL nodes}, 0.5) for a total of nNodes+2
 * TODO: Incorporate COM centering?
 **/
void ModalBasis::initialize_basis(const AthelasArray3D<double> uPF,
                                  const GridStructure *grid) {
  static const int n_eta = nNodes_ + 2;
  static const int ilo = 1;
  static const int ihi = grid->get_ihi();

  double eta = 0.5;
  for (int ix = ilo; ix <= ihi; ix++) {
    for (int k = 0; k < order_; k++) {
      for (int i_eta = 0; i_eta < n_eta; i_eta++) {
        // face values
        if (i_eta == 0) {
          eta = -0.5;
        } else if (i_eta == nNodes_ + 1) {
          eta = +0.5;
        } else if (i_eta > 0 && i_eta < nNodes_ + 1) { // GL nodes
          eta = grid->get_nodes(i_eta - 1);
        }

        phi_(ix, i_eta, k) = ortho(k, ix, i_eta, eta, 0.0, uPF, grid, false);
        dphi_(ix, i_eta, k) = ortho(k, ix, i_eta, eta, 0.0, uPF, grid, true);
      }
    }
  }
  // check_orthogonality(uPF, grid);
  compute_mass_matrix(uPF, grid);

  // === Fill Guard cells ===

  // ? Using identical basis in guard cells as boundaries ?
  for (int ix = 0; ix < ilo; ix++) {
    for (int i_eta = 0; i_eta < n_eta; i_eta++) {
      for (int k = 0; k < order_; k++) {
        phi_(ilo - 1 - ix, i_eta, k) = phi_(ilo + ix, i_eta, k);
        phi_(ihi + 1 + ix, i_eta, k) = phi_(ihi - ix, i_eta, k);

        dphi_(ilo - 1 - ix, i_eta, k) = dphi_(ilo + ix, i_eta, k);
        dphi_(ihi + 1 + ix, i_eta, k) = dphi_(ihi - ix, i_eta, k);
      }
    }
  }

  for (int ix = 0; ix < ilo; ix++) {
    for (int k = 0; k < order_; k++) {
      mass_matrix_(ilo - 1 - ix, k) = mass_matrix_(ilo + ix, k);
      mass_matrix_(ihi + 1 + ix, k) = mass_matrix_(ihi - ix, k);
    }
  }
}

/**
 * The following checks orthogonality of basis functions on each cell.
 * Returns error if orthogonality is not met.
 **/
void ModalBasis::check_orthogonality(const AthelasArray3D<double> uPF,
                                     const GridStructure *grid) const {

  static const int ilo = 1;
  static const int ihi = grid->get_ihi();

  const auto dr = grid->widths();
  const auto weights = grid->weights();
  const auto sqrt_gm = grid->sqrt_gm();

  for (int ix = ilo; ix <= ihi; ix++) {
    for (int k1 = 0; k1 < order_; k1++) {
      for (int k2 = 0; k2 < order_; k2++) {
        double result = 0.0;
        for (int i_eta = 1; i_eta <= nNodes_; i_eta++) // loop over quadratures
        {
          const double rho =
              density_weight_ ? uPF(ix, i_eta, vars::prim::Rho) : 1.0;
          // Not using an inner_product function because their API is odd..
          result += phi_(ix, i_eta, k1) * phi_(ix, i_eta, k2) * rho *
                    weights(i_eta - 1) * dr(ix) * sqrt_gm(ix, i_eta);
        }

        if (k1 == k2 && result == 0.0) {
          THROW_ATHELAS_ERROR(
              " ! Basis not orthogonal: Diagonal term equal to zero.\n");
        }
        if (k1 != k2 && std::abs(result) > 1e-10) {
          std::println("{} {} {:.3e}", k1, k2, result);
          THROW_ATHELAS_ERROR(
              " ! Basis not orthogonal: Off diagonal term non-zero.\n");
        }
      }
    }
  }
}

/**
 * Computes \int \rho \phi_m \phi_m dw on each cell
 * TODO: Extend mass matrix to more nodes
 * ? Do I need more integration nodes for the mass matrix? ?
 * ? If so, how do I expand this ?
 * ? I would need to compute and store more GL nodes, weights ?
 **/
void ModalBasis::compute_mass_matrix(const AthelasArray3D<double> uPF,
                                     const GridStructure *grid) {
  static const int ilo = 1;
  static const int ihi = grid->get_ihi();
  const int nNodes_ = grid->n_nodes();

  const auto dr = grid->widths();
  const auto sqrt_gm = grid->sqrt_gm();
  const auto weights = grid->weights();

  for (int ix = ilo; ix <= ihi; ix++) {
    for (int k = 0; k < order_; k++) {
      double result = 0.0;
      for (int iN = 0; iN < nNodes_; iN++) {
        // include rho in integrand if necessary
        const double rho =
            density_weight_ ? uPF(ix, iN + 1, vars::prim::Rho) : 1.0;
        result += phi_(ix, iN + 1, k) * phi_(ix, iN + 1, k) * weights(iN) *
                  dr(ix) * sqrt_gm(ix, iN + 1) * rho;
      }
      mass_matrix_(ix, k) = result;
      inv_mass_matrix_(ix, k) = 1.0 / result;
    }
  }
}

/**
 * Evaluate (modal) basis on element ix for quantity q.
 **/
auto ModalBasis::basis_eval(AthelasArray3D<double> U, const int ix, const int q,
                            const int i_eta) const -> double {
  double result = 0.0;
  for (int k = 0; k < order_; k++) {
    result += phi_(ix, i_eta, k) * U(ix, k, q);
  }
  return result;
}

// Same as above, for a 2D vector U_k on a given cell and quantity
// e.g., U(:, ix, :)
auto ModalBasis::basis_eval(AthelasArray2D<double> U, const int ix, const int q,
                            const int i_eta) const -> double {
  double result = 0.0;
  for (int k = 0; k < order_; k++) {
    result += phi_(ix, i_eta, k) * U(k, q);
  }
  return result;
}

// Same as above, for a 1D vector U_k on a given cell and quantity
// e.g., U(ix, :, q)
auto ModalBasis::basis_eval(AthelasArray1D<double> U, const int ix,
                            const int i_eta) const -> double {
  double result = 0.0;
  for (int k = 0; k < order_; k++) {
    result += phi_(ix, i_eta, k) * U(k);
  }
  return result;
}

[[nodiscard]] auto ModalBasis::phi() const noexcept -> AthelasArray3D<double> {
  return phi_;
}

// Accessor for dphi_
auto ModalBasis::get_d_phi(const int ix, const int i_eta, const int k) const
    -> double {
  return dphi_(ix, i_eta, k);
}

[[nodiscard]] auto ModalBasis::dphi() const noexcept -> AthelasArray3D<double> {
  return dphi_;
}

[[nodiscard]] auto ModalBasis::mass_matrix() const noexcept
    -> AthelasArray2D<double> {
  return mass_matrix_;
}
[[nodiscard]] auto ModalBasis::inv_mass_matrix() const noexcept
    -> AthelasArray2D<double> {
  return inv_mass_matrix_;
}

// Accessor for Order
auto ModalBasis::order() const noexcept -> int { return order_; }

/**
 * L2 projection from nodal function to modal representation for all cells
 * Projects nodal_func(x) onto the modal basis for all cells, quantity q
 *
 * Parameters:
 * -----------
 * uCF: conserved fields (modal representation)
 * uPF: primitive fields (nodal representation)
 * grid: grid structure
 * q: conserved field index
 * nodal_func: function that takes x coordinate and returns nodal value
 **/
void ModalBasis::project_nodal_to_modal_all_cells(
    AthelasArray3D<double> uCF, AthelasArray3D<double> uPF, GridStructure *grid,
    int q, const std::function<double(double, int, int)> &nodal_func) const {
  static const IndexRange ib(grid->domain<Domain::Interior>());

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Basis :: Project modal to nodal (all)",
      DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
        project_nodal_to_modal(uCF, uPF, grid, q, i, nodal_func);
      });
}

} // namespace athelas::basis
