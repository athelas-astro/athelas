#include "basis/nodal_basis.hpp"
#include "basis/polynomial_basis.hpp"
#include "kokkos_abstraction.hpp"

namespace athelas::basis {

NodalBasis::NodalBasis(AthelasArray3D<double> uPF, GridStructure *grid,
                       const int nN, const int nElements)
    : nX_(nElements), nNodes_(nN), nodes_("quadrature nodes", nN),
      weights_("quadrature weights", nN),
      phi_("phi_", nElements + 2, nN + 2, nN),
      dphi_("dphi_", nElements + 2, nN + 2, nN),
      mass_matrix_("MassMatrix", nElements + 2, nN),
      inv_mass_matrix_("InvMassMatrix", nElements + 2, nN),
      differentiation_matrix_("DiffMatrix", nN, nN),
      vandermonde_("vandermonde", nN, nN),
      inv_vandermonde_("inverse vandermonde", nN, nN) {

  Kokkos::deep_copy(nodes_, grid->nodes());
  Kokkos::deep_copy(weights_, grid->weights());

  initialize_basis(uPF, grid);
}

[[nodiscard]] auto NodalBasis::phi() const noexcept -> AthelasArray3D<double> {
  return phi_;
}

[[nodiscard]] auto NodalBasis::dphi() const noexcept -> AthelasArray3D<double> {
  return dphi_;
}

auto NodalBasis::get_d_phi(const int ix, const int i_eta, const int k) const
    -> double {
  return dphi_(ix, i_eta, k);
}

[[nodiscard]] auto NodalBasis::mass_matrix() const noexcept
    -> AthelasArray2D<double> {
  return mass_matrix_;
}

[[nodiscard]] auto NodalBasis::inv_mass_matrix() const noexcept
    -> AthelasArray2D<double> {
  return inv_mass_matrix_;
}

auto NodalBasis::order() const noexcept -> int { return nNodes_; }

[[nodiscard]] auto NodalBasis::differentiation_matrix() const noexcept
    -> AthelasArray2D<double> {
  return differentiation_matrix_;
}

/**
 * @brief Evaluate nodal representation at i_eta
 */
auto NodalBasis::basis_eval(AthelasArray3D<double> U, const int ix, const int q,
                            const int i_eta) const -> double {
  double result = 0.0;
  for (int j = 0; j < nNodes_; j++) {
    result += phi_(ix, i_eta, j) * U(ix, j, q);
  }
  return result;
}

auto NodalBasis::basis_eval(AthelasArray2D<double> U, const int ix, const int q,
                            const int i_eta) const -> double {
  double result = 0.0;
  for (int j = 0; j < nNodes_; j++) {
    result += phi_(ix, i_eta, j) * U(j, q);
  }
  return result;
}

auto NodalBasis::basis_eval(AthelasArray1D<double> U, const int ix,
                            const int i_eta) const -> double {
  double result = 0.0;
  for (int j = 0; j < nNodes_; j++) {
    result += phi_(ix, i_eta, j) * U(j);
  }
  return result;
}

auto NodalBasis::lagrange_polynomial(const int j, const double xi,
                                     const AthelasArray1D<double> nodes)
    -> double {
  auto nodes_h = Kokkos::create_mirror_view(nodes);
  Kokkos::deep_copy(nodes_h, nodes);

  const int nnodes = static_cast<int>(nodes_h.extent(0));

  double result = 1.0;
  for (int k = 0; k < nnodes; k++) {
    if (k != j) {
      result *= (xi - nodes_h(k)) / (nodes_h(j) - nodes_h(k));
    }
  }
  return result;
}

auto NodalBasis::d_lagrange_polynomial(const int j, const double xi,
                                       const AthelasArray1D<double> nodes)
    -> double {

  auto nodes_h = Kokkos::create_mirror_view(nodes);
  Kokkos::deep_copy(nodes_h, nodes);

  const int n = static_cast<int>(nodes_h.extent(0));
  double result = 0.0;

  // Product rule: d/dx[prod f_k] = sum_m [ f_m' * prod_{k!=m} f_k ]
  for (int m = 0; m < n; m++) {
    if (m != j) {
      double term = 1.0 / (nodes_h(j) - nodes_h(m));
      for (int k = 0; k < n; k++) {
        if (k != j && k != m) {
          term *= (xi - nodes_h(k)) / (nodes_h(j) - nodes_h(k));
        }
      }
      result += term;
    }
  }

  return result;
}

void NodalBasis::build_differentiation_matrix() {
  auto D_h = Kokkos::create_mirror_view(differentiation_matrix_);
  auto nodes_h = Kokkos::create_mirror_view(nodes_);
  Kokkos::deep_copy(nodes_h, nodes_);

  // Compute barycentric weights, store temporarily in diagonal
  for (int j = 0; j < nNodes_; j++) {
    double w = 1.0;
    for (int k = 0; k < nNodes_; k++) {
      if (k != j) {
        w /= (nodes_h(j) - nodes_h(k));
      }
    }
    D_h(j, j) = w;
  }

  // Fill off-diagonal: D_ij = (w_j / w_i) / (xi_i - xi_j) for i != j
  for (int i = 0; i < nNodes_; i++) {
    for (int j = 0; j < nNodes_; j++) {
      if (i != j) {
        D_h(i, j) = D_h(j, j) / D_h(i, i) / (nodes_h(i) - nodes_h(j));
      }
    }
  }

  // Diagonal via row sum: D_ii = -sum_{j!=i} D_ij
  for (int i = 0; i < nNodes_; i++) {
    double sum = 0.0;
    for (int j = 0; j < nNodes_; j++) {
      if (i != j) {
        sum += D_h(i, j);
      }
    }
    D_h(i, i) = -sum;
  }

  Kokkos::deep_copy(differentiation_matrix_, D_h);
}

/**
 * @brief Build the Vandermonde and inverse Vandermonde matrices
 * We make use of the relation
 *   M = V^T W V
 * Where M is the mass matrix, V the Vandermonde, and W = diag{w_i}
 * contains quadrature weights.
 */
void NodalBasis::build_vandermonde_matrices() {
  auto V_h = Kokkos::create_mirror_view(vandermonde_);
  auto invV_h = Kokkos::create_mirror_view(inv_vandermonde_);
  auto nodes_h = Kokkos::create_mirror_view(nodes_);
  Kokkos::deep_copy(nodes_h, nodes_);
  auto weights_h = Kokkos::create_mirror_view(weights_);
  Kokkos::deep_copy(weights_h, weights_);

  // Build V_{qk} = P_k(x_q)
  for (int q = 0; q < nNodes_; ++q) {
    const double xq = 2.0 * nodes_h(q);
    for (int k = 0; k < nNodes_; ++k) {
      V_h(q, k) = legendre(k, xq);
    }
  }

  // Build inverse Vandermonde (projection matrix)
  // invV_{kq} = (2k+1) * w_q * P_k(x_q)
  for (int k = 0; k < nNodes_; ++k) {
    const double scale = (2.0 * static_cast<double>(k) + 1.0);
    for (int q = 0; q < nNodes_; ++q) {
      invV_h(k, q) = scale * weights_h(q) * V_h(q, k);
    }
  }

  Kokkos::deep_copy(vandermonde_, V_h);
  Kokkos::deep_copy(inv_vandermonde_, invV_h);
}

/**
 * @brief Use the inverse Vandermonde to map a nodal basis to a modal one.
 * The IndexRange is for the variables in ucf that we are mapping.
 * The modal vector u_k_ loops from 0.
 */
void NodalBasis::nodal_to_modal(AthelasArray3D<double> u_k,
                                AthelasArray3D<double> ucf,
                                const IndexRange &vb) const {
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "nodal_to_modal", DevExecSpace(), 0,
      u_k.extent(0) - 1, KOKKOS_CLASS_LAMBDA(const int i) {
        for (int v = vb.s; v <= vb.e; ++v) {
          for (int k = 0; k < nNodes_; ++k) {
            double sum = 0.0;
            for (int q = 0; q < nNodes_; ++q) {
              sum += inv_vandermonde_(k, q) * ucf(i, q, v);
            }
            u_k(i, k, v - vb.s) = sum;
          }
        }
      });
}

/**
 * @brief Use the Vandermonde to map a modal basis to a nodal one.
 * The IndexRange is for the variables in ucf that we are mapping.
 * The modal vector u_k_ loops from 0.
 */
void NodalBasis::modal_to_nodal(AthelasArray3D<double> ucf,
                                AthelasArray3D<double> u_k,
                                const IndexRange &vb) const {
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "modal_to_nodal", DevExecSpace(), 0,
      u_k.extent(0) - 1, KOKKOS_CLASS_LAMBDA(const int i) {
        for (int v = vb.s; v <= vb.e; ++v) {
          for (int q = 0; q < nNodes_; ++q) {
            double sum = 0.0;
            for (int k = 0; k < nNodes_; ++k) {
              sum += vandermonde_(q, k) * u_k(i, k, v - vb.s);
            }
            ucf(i, q, v) = sum;
          }
        }
      });
}

/**
 * @brief Initialize datastructures for the basis.
 */
void NodalBasis::initialize_basis(const AthelasArray3D<double> uPF,
                                  const GridStructure *grid) {

  const int ilo = 1;
  const int ihi = grid->get_ihi();
  const auto nodes = grid->nodes();

  build_differentiation_matrix();
  build_vandermonde_matrices();

  auto phi_h = Kokkos::create_mirror_view(phi_);
  auto dphi_h = Kokkos::create_mirror_view(dphi_);
  auto nodes_h = Kokkos::create_mirror_view(nodes);
  Kokkos::deep_copy(nodes_h, nodes);

  // Evaluate L_j at: i_eta=0 (left face), i_eta=1..nNodes (GL nodes),
  // i_eta=nNodes+1 (right face)
  for (int ix = ilo; ix <= ihi; ix++) {
    for (int j = 0; j < nNodes_; j++) {

      // Left face (i_eta = 0, eta = -0.5)
      phi_h(ix, 0, j) = lagrange_polynomial(j, -0.5, nodes);
      dphi_h(ix, 0, j) = d_lagrange_polynomial(j, -0.5, nodes);

      // Interior GL nodes: L_j(eta_i) = delta_ij
      for (int i = 0; i < nNodes_; i++) {
        const int i_eta = i + 1;
        phi_h(ix, i_eta, j) = (i == j) ? 1.0 : 0.0;

        // dphi from differentiation matrix
        auto D_h = Kokkos::create_mirror_view(differentiation_matrix_);
        Kokkos::deep_copy(D_h, differentiation_matrix_);
        dphi_h(ix, i_eta, j) = D_h(i, j);
      }

      // Right face (i_eta = nNodes+1, eta = 0.5)
      phi_h(ix, nNodes_ + 1, j) = lagrange_polynomial(j, 0.5, nodes);
      dphi_h(ix, nNodes_ + 1, j) = d_lagrange_polynomial(j, 0.5, nodes);
    }
  }

  Kokkos::deep_copy(phi_, phi_h);
  Kokkos::deep_copy(dphi_, dphi_h);

  compute_mass_matrix(grid);
  fill_guard_cells(grid);
}

/**
 * @brief Compute diagonal mass matrix
 * M_qq = w_q dm
 * NOTE: This is constant in time on an element.
 */
void NodalBasis::compute_mass_matrix(const GridStructure *grid) {

  const int ilo = 1;
  const int ihi = grid->get_ihi();

  auto weights = grid->weights();
  auto mass = grid->mass();

  auto mass_h = Kokkos::create_mirror_view(mass_matrix_);
  auto inv_mass_h = Kokkos::create_mirror_view(inv_mass_matrix_);

  for (int i = ilo; i <= ihi; ++i) {
    for (int j = 0; j < nNodes_; j++) {
      const double M_jj = weights(j) * mass(i);

      mass_h(i, j) = M_jj;
      inv_mass_h(i, j) = 1.0 / M_jj;
    }
  }

  Kokkos::deep_copy(mass_matrix_, mass_h);
  Kokkos::deep_copy(inv_mass_matrix_, inv_mass_h);
}

void NodalBasis::fill_guard_cells(const GridStructure *grid) {

  const int ilo = 1;
  const int ihi = grid->get_ihi();
  const int n_eta = nNodes_ + 2;

  auto phi_h = Kokkos::create_mirror_view(phi_);
  auto dphi_h = Kokkos::create_mirror_view(dphi_);
  auto mass_h = Kokkos::create_mirror_view(mass_matrix_);

  Kokkos::deep_copy(phi_h, phi_);
  Kokkos::deep_copy(dphi_h, dphi_);
  Kokkos::deep_copy(mass_h, mass_matrix_);

  // Mirror boundary values into guard cells
  for (int ix = 0; ix < ilo; ix++) {
    for (int i_eta = 0; i_eta < n_eta; i_eta++) {
      for (int j = 0; j < nNodes_; j++) {
        phi_h(ilo - 1 - ix, i_eta, j) = phi_h(ilo + ix, i_eta, j);
        phi_h(ihi + 1 + ix, i_eta, j) = phi_h(ihi - ix, i_eta, j);

        dphi_h(ilo - 1 - ix, i_eta, j) = dphi_h(ilo + ix, i_eta, j);
        dphi_h(ihi + 1 + ix, i_eta, j) = dphi_h(ihi - ix, i_eta, j);
      }
    }

    for (int j = 0; j < nNodes_; j++) {
      mass_h(ilo - 1 - ix, j) = mass_h(ilo + ix, j);
      mass_h(ihi + 1 + ix, j) = mass_h(ihi - ix, j);
    }
  }

  Kokkos::deep_copy(phi_, phi_h);
  Kokkos::deep_copy(dphi_, dphi_h);
  Kokkos::deep_copy(mass_matrix_, mass_h);
}

auto d_legendre_n3(const int n, const int q, const double x) -> double {
  if (q > n || q < 0) {
    return 0.0;
  }

  // Start by finding the constant q-th derivative of P_q
  double p_prev = 1.0;
  for (int i = 1; i <= q; ++i) {
    p_prev *= (2.0 * i - 1.0); // This is (2q-1)!!
  }

  if (n == q) {
    return p_prev;
  }

  // Next is the derivative of P_{q+1}
  // P_{q+1}^(q) = (2q+1) * x * P_q^(q)
  double p_curr = (2.0 * q + 1.0) * x * p_prev;

  if (n == q + 1) {
    return p_curr;
  }

  // Use the three-term recurrence for derivatives
  for (int i = q + 2; i <= n; ++i) {
    double p_next =
        ((2.0 * i - 1.0) * x * p_curr - (i + q - 1.0) * p_prev) / (i - q);
    p_prev = p_curr;
    p_curr = p_next;
  }

  return p_curr;
}

auto d_legendre_n2(const int n, const int q, const double x) -> double {
  if (q > n || q < 0) {
    return 0.0;
  }

  // Base case: q-th derivative of P_q is a constant
  // Constant value: (2q)! / (2^q * q!) = (2q-1)!!
  double dfact = 1.0;
  for (int i = 1; i <= q; ++i) {
    dfact *= (q + i) * 0.5;
  }

  if (n == q) {
    return dfact;
  }

  // Use the Three-Term Recurrence for Jacobi Polynomials P_m^(alpha, beta)
  // Here m = n - q, and alpha = beta = q
  const int alpha = q;
  const double a_plus_b = 2.0 * q;

  double p_prev = dfact; // This is J_0
  // J_1 = 0.5 * (alpha - beta + (alpha + beta + 2) * x) * J_0
  double p_curr = 0.5 * (a_plus_b + 2.0) * x * p_prev;

  if (n == q + 1) {
    return p_curr;
  }

  for (int m = 1; m < (n - q); ++m) {
    const auto m_d = static_cast<double>(m);

    // Coefficients for the Jacobi recurrence J_{m+1}
    const double a1 =
        2.0 * (m_d + 1.0) * (m_d + a_plus_b + 1.0) * (2.0 * m_d + a_plus_b);
    const double a3 = (2.0 * m_d + a_plus_b) * (2.0 * m_d + a_plus_b + 1.0) *
                      (2.0 * m_d + a_plus_b + 2.0);
    const double a4 =
        2.0 * (m_d + alpha) * (m_d + alpha) * (2.0 * m_d + a_plus_b + 2.0);

    const double p_next = (a3 * x * p_curr - a4 * p_prev) / a1;

    p_prev = p_curr;
    p_curr = p_next;
  }

  return p_curr;
}

} // namespace athelas::basis
