/**
 * @file NodalBasis.cpp
 * @brief Implementation of nodal DG basis using Lagrange polynomials
 */

#include "basis/nodal_basis.hpp"

namespace athelas::basis {

NodalBasis::NodalBasis(const AthelasArray3D<double> uPF, GridStructure *grid,
                       const int nN, const int nElements,
                       const bool density_weight)
    : nX_(nElements), nNodes_(nN), mSize_((nN) * (nN + 2) * (nElements + 2)),
      density_weight_(density_weight),
      mass_matrix_("MassMatrix", nElements + 2, nN),
      inv_mass_matrix_("InvMassMatrix", nElements + 2, nN),
      phi_("phi_", nElements + 2, nN + 2, nN),
      dphi_("dphi_", nElements + 2, nN + 2, nN),
      differentiation_matrix_("DiffMatrix", nN, nN), 
      legendre_phi_("noda::legendre_phi", nN, 5) {

  grid->compute_mass(uPF);
  grid->compute_mass_r(uPF);
  grid->compute_center_of_mass(uPF);

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
 * @details Interior nodes: direct return. Faces: Lagrange extrapolation via
 * phi_
 */
auto NodalBasis::basis_eval(AthelasArray3D<double> U, const int ix, const int q,
                            const int i_eta) const -> double {

  // Left face: extrapolate using phi_(ix, 0, j) = L_j(-0.5)
  if (i_eta == 0) {
    double result = 0.0;
    for (int j = 0; j < nNodes_; j++) {
      result += phi_(ix, 0, j) * U(ix, j, q);
    }
    return result;
  }

  // Right face: extrapolate using phi_(ix, nNodes+1, j) = L_j(0.5)
  if (i_eta == nNodes_ + 1) {
    double result = 0.0;
    for (int j = 0; j < nNodes_; j++) {
      result += phi_(ix, nNodes_ + 1, j) * U(ix, j, q);
    }
    return result;
  }

  // Interior nodes: direct return
  const int j = i_eta - 1;
  return U(ix, j, q);
}

auto NodalBasis::basis_eval(AthelasArray2D<double> U, const int ix, const int q,
                            const int i_eta) const -> double {

  if (i_eta == 0) {
    double result = 0.0;
    for (int j = 0; j < nNodes_; j++) {
      result += phi_(ix, 0, j) * U(j, q);
    }
    return result;
  }

  if (i_eta == nNodes_ + 1) {
    double result = 0.0;
    for (int j = 0; j < nNodes_; j++) {
      result += phi_(ix, nNodes_ + 1, j) * U(j, q);
    }
    return result;
  }

  const int j = i_eta - 1;
  return U(j, q);
}

auto NodalBasis::basis_eval(AthelasArray1D<double> U, const int ix,
                            const int i_eta) const -> double {

  if (i_eta == 0) {
    double result = 0.0;
    for (int j = 0; j < nNodes_; j++) {
      result += phi_(ix, 0, j) * U(j);
    }
    return result;
  }

  if (i_eta == nNodes_ + 1) {
    double result = 0.0;
    for (int j = 0; j < nNodes_; j++) {
      result += phi_(ix, nNodes_ + 1, j) * U(j);
    }
    return result;
  }

  const int j = i_eta - 1;
  return U(j);
}

auto NodalBasis::lagrange_polynomial(const int j, const double xi,
                                     const AthelasArray1D<double> &nodes)
    -> double {

  auto nodes_h = Kokkos::create_mirror_view(nodes);
  Kokkos::deep_copy(nodes_h, nodes);

  double result = 1.0;
  for (int k = 0; k < nodes_h.extent(0); k++) {
    if (k != j) {
      result *= (xi - nodes_h(k)) / (nodes_h(j) - nodes_h(k));
    }
  }
  return result;
}

auto NodalBasis::d_lagrange_polynomial(const int j, const double xi,
                                       const AthelasArray1D<double> &nodes)
    -> double {

  auto nodes_h = Kokkos::create_mirror_view(nodes);
  Kokkos::deep_copy(nodes_h, nodes);

  const int n = nodes_h.extent(0);
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

void NodalBasis::build_differentiation_matrix(
    const AthelasArray1D<double> &nodes) {

  auto D_h = Kokkos::create_mirror_view(differentiation_matrix_);
  auto nodes_h = Kokkos::create_mirror_view(nodes);
  Kokkos::deep_copy(nodes_h, nodes);

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

void NodalBasis::initialize_basis(const AthelasArray3D<double> uPF,
                                  const GridStructure *grid) {

  const int ilo = 1;
  const int ihi = grid->get_ihi();
  const auto nodes = grid->nodes();

  build_differentiation_matrix(nodes);

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

  compute_mass_matrix(uPF, grid);
  fill_guard_cells(grid);
}

void NodalBasis::compute_mass_matrix(const AthelasArray3D<double> uPF,
                                     const GridStructure *grid) {

  const int ilo = 1;
  const int ihi = grid->get_ihi();

  const auto dr = grid->widths();
  const auto sqrt_gm = grid->sqrt_gm();
  const auto weights = grid->weights();

  auto mass_h = Kokkos::create_mirror_view(mass_matrix_);
  auto inv_mass_h = Kokkos::create_mirror_view(inv_mass_matrix_);

  // Diagonal mass matrix: M_jj = w_j * rho_j * sqrt_gm_j * dr
  for (int ix = ilo; ix <= ihi; ix++) {
    for (int j = 0; j < nNodes_; j++) {
      const int iN = j;

      const double rho =
          density_weight_ ? uPF(ix, iN + 1, vars::prim::Rho) : 1.0;

      const double M_jj = weights(iN) * rho * sqrt_gm(ix, iN + 1) * dr(ix);

      mass_h(ix, j) = M_jj;
      inv_mass_h(ix, j) = 1.0 / M_jj;
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

} // namespace athelas::basis
