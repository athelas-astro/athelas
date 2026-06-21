/**
 * @file mesh.cpp
 * --------------
 *
 * @brief Class for holding the spatial grid.
 *
 * @details This class Mesh holds key pieces of the grid:
 *          - nx
 *          - nnodes
 *          - weights
 *
 *          For a loop over real zones, loop from ilo to ihi (inclusive).
 *          ilo = 1 (one ghost cell)
 *          ihi = nElements_
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <vector>

#include "basic_types.hpp"
#include "geometry/mesh.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"
#include "math/quadrature.hpp"
#include "math/utils.hpp"
#include "utils/utilities.hpp"

namespace athelas {

Mesh::Mesh(const ProblemIn *pin)
    : nElements_(pin->param()->get<int>("problem.nx")),
      nNodes_(pin->param()->get<int>("basis.nnodes")), mSize_(nElements_ + 2),
      xL_(pin->param()->get<double>("problem.xl")),
      xR_(pin->param()->get<double>("problem.xr")),
      geometry_(pin->param()->get<std::string>("problem.geometry")),
      grid_type_(pin->param()->get<std::string>("problem.grid_type")),
      nodes_("Nodes", nNodes_), weights_("weights_", nNodes_),
      centers_("Centers", mSize_), widths_("widths_", mSize_),
      x_l_("Left Interface", mSize_ + 1), mass_("Cell mass_", mSize_),
      dm_deta_("Reference mass density", mSize_, nNodes_),
      mass_r_("Enclosed mass", mSize_, nNodes_ + 2),
      center_of_mass_("Center of mass_", mSize_),
      sqrt_gm_("Sqrt Gamma", mSize_, nNodes_ + 2),
      integration_matrix_("Integration matrix", nNodes_, nNodes_),
      grid_("Mesh", mSize_, nNodes_ + 2) {
  std::vector<double> tmp_nodes(nNodes_);
  std::vector<double> tmp_weights(nNodes_);

  for (int q = 0; q < nNodes_; q++) {
    tmp_nodes[q] = 0.0;
    tmp_weights[q] = 0.0;
  }

  math::quadrature::lg_quadrature(nNodes_, tmp_nodes, tmp_weights);

  // TODO(astrobarker): use host copies for this.
  for (int q = 0; q < nNodes_; q++) {
    nodes_(q) = tmp_nodes[q];
    weights_(q) = tmp_weights[q];
  }

  build_integration_matrix();
  create_grid(pin);
}

// Return cell center
auto Mesh::centers(int iC) const -> double { return centers_(iC); }

// Return given quadrature node
auto Mesh::get_nodes(int nN) const -> double { return nodes_(nN); }

// Accessor for xL
auto Mesh::get_x_l() const noexcept -> double { return xL_; }

// Accessor for xR
auto Mesh::get_x_r() const noexcept -> double { return xR_; }

// Accessor for SqrtGm
KOKKOS_INLINE_FUNCTION
auto Mesh::get_sqrt_gm(const double X) const -> double {
  if (geometry_ == "spherical") [[likely]] {
    return X * X;
  }
  return 1.0;
}

KOKKOS_INLINE_FUNCTION
auto Mesh::coordinate_volume(const double X) const -> double {
  if (geometry_ == "spherical") [[likely]] {
    return X * X * X / 3.0;
  }
  return X;
}

KOKKOS_INLINE_FUNCTION
auto Mesh::coordinate_from_volume(const double X) const -> double {
  if (geometry_ == "spherical") [[likely]] {
    return std::cbrt(3.0 * X);
  }
  return X;
}

// Return nNodes_
KOKKOS_FUNCTION
auto Mesh::n_nodes() const noexcept -> int { return nNodes_; }

// Return nElements_
KOKKOS_FUNCTION
auto Mesh::n_elements() const noexcept -> int { return nElements_; }

// Return first physical zone
KOKKOS_FUNCTION
auto Mesh::get_ilo() noexcept -> int { return 1; }

// Return last physical zone
KOKKOS_FUNCTION
auto Mesh::get_ihi() const noexcept -> int { return nElements_; }

// Return true if in spherical symmetry
KOKKOS_FUNCTION
auto Mesh::do_geometry() const noexcept -> bool {
  return geometry_ == "spherical";
}

// grid creation logic
void Mesh::create_grid(const ProblemIn *pin) {
  if (utilities::to_lower(grid_type_) == "uniform") {
    create_uniform_grid();
  } else if (utilities::to_lower(grid_type_) == "logarithmic") {
    // Need to be careful of coordinates with log grid!
    if (xL_ < 0.0 || xR_ < 0.0) {
      throw_athelas_error(
          "Negative coordinates are not supported with logarithmic gridding!");
    }
    if (xL_ == 0.0) {
      // We cannot have a boundary at 0 with a log grid. We replace the inner
      // boundary arbitrarily with 1.0e-4 * xR and warn the user to perhaps
      // consider asetting a better inner boundary.
      // This logic is not very general and might be made smarter.
      // It also is not an important edge case.
      const double new_xl = 1.0e-4 * xR_;
      std::stringstream ss;
      ss << std::scientific << std::setprecision(3) << new_xl;
      athelas_warning("Logarithmic grid requestion with XL = 0.0. This does "
                      "not work. Setting xL = 10^-4 * xR = " +
                      ss.str() +
                      ". You might consider setting a more appropriate value.");

      // update outer boundary in params and in class
      auto &xl_pin = pin->param()->get_mutable_ref<double>("problem.xl");
      xl_pin = new_xl;
      xL_ = new_xl;
    }
    create_log_grid();
  } else {
    throw_athelas_error("Unknown grid type '" + grid_type_ + "' provided!");
  }
}

void Mesh::copy_from(const Mesh &other) {
  nElements_ = other.nElements_;
  nNodes_ = other.nNodes_;
  mSize_ = other.mSize_;
  xL_ = other.xL_;
  xR_ = other.xR_;
  geometry_ = other.geometry_;
  grid_type_ = other.grid_type_;

  Kokkos::deep_copy(nodes_, other.nodes_);
  Kokkos::deep_copy(weights_, other.weights_);
  Kokkos::deep_copy(centers_, other.centers_);
  Kokkos::deep_copy(widths_, other.widths_);
  Kokkos::deep_copy(x_l_, other.x_l_);
  Kokkos::deep_copy(mass_, other.mass_);
  Kokkos::deep_copy(dm_deta_, other.dm_deta_);
  Kokkos::deep_copy(mass_r_, other.mass_r_);
  Kokkos::deep_copy(center_of_mass_, other.center_of_mass_);
  Kokkos::deep_copy(sqrt_gm_, other.sqrt_gm_);
  Kokkos::deep_copy(integration_matrix_, other.integration_matrix_);
  Kokkos::deep_copy(grid_, other.grid_);
}

/**
 * @brief Build the reference integration matrix
 *   I(i, q) = \int_{-1/2}^{eta_i} L_q(eta) deta,
 * where L_q is the q-th Lagrange polynomial on the reference element
 * [-1/2, 1/2]. Writing L_q in monomial form, L_q = (sum_k c_k eta^k) / denom,
 * each entry is integrated analytically term by term:
 *   I(i, q) = sum_k (c_k / denom) (eta_i^{k+1} - (-1/2)^{k+1}) / (k + 1).
 * Contracting a field's nodal values against row i gives its partial integral
 * up to node i:
 *   \int_{-1/2}^{eta_i} f_h deta = sum_q I(i, q) f_q.
 * Used to place interior nodes from cumulative mass/volume integrals in
 * reconstruct_mesh and compute_mass_r.
 * NOTE: Conceptually this belongs to the basis but is only used for the Mesh.
 **/
void Mesh::build_integration_matrix() {
  auto nodes_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), nodes_);
  auto integration_matrix_h = Kokkos::create_mirror_view(integration_matrix_);

  for (int q = 0; q < nNodes_; ++q) {
    std::vector<double> coeffs(1, 1.0);
    double denom = 1.0;

    for (int m = 0; m < nNodes_; ++m) {
      if (m == q) {
        continue;
      }

      std::vector<double> next(coeffs.size() + 1, 0.0);
      for (std::size_t k = 0; k < coeffs.size(); ++k) {
        next[k] -= nodes_h(m) * coeffs[k];
        next[k + 1] += coeffs[k];
      }
      coeffs = next;
      denom *= nodes_h(q) - nodes_h(m);
    }

    for (int i = 0; i < nNodes_; ++i) {
      double integral = 0.0;
      const double eta = nodes_h(i);
      for (std::size_t k = 0; k < coeffs.size(); ++k) {
        const auto power = static_cast<double>(k + 1);
        integral += (coeffs[k] / denom) *
                    (std::pow(eta, power) - std::pow(-0.5, power)) / power;
      }
      integration_matrix_h(i, q) = integral;
    }
  }

  Kokkos::deep_copy(integration_matrix_, integration_matrix_h);
}

/**
 * @brief uniform mesh
 */
void Mesh::create_uniform_grid() {

  const int ilo = 1; // first real zone
  const int ihi = nElements_; // last real zone

  auto widths_h = Kokkos::create_mirror_view(widths_);
  auto centers_h = Kokkos::create_mirror_view(centers_);
  auto x_l_h = Kokkos::create_mirror_view(x_l_);

  const double dx = (xR_ - xL_) / static_cast<double>(nElements_);

  x_l_h(0) = xL_ - dx;
  for (int i = ilo; i <= ihi + 1; i++) {
    x_l_h(i) = xL_ + static_cast<double>(i - ilo) * dx;
  }
  x_l_h(ihi + 1) = xR_;
  x_l_h(ihi + 2) = xR_ + dx;

  for (int i = ilo - 1; i <= ihi + 1; i++) {
    if (!std::isfinite(x_l_h(i)) || !std::isfinite(x_l_h(i + 1)) ||
        x_l_h(i + 1) <= x_l_h(i)) {
      throw_athelas_error("Invalid uniform grid construction!");
    }
    widths_h(i) = x_l_h(i + 1) - x_l_h(i);
    centers_h(i) = 0.5 * (x_l_h(i) + x_l_h(i + 1));
  }

  // copy back to device mirrors
  Kokkos::deep_copy(widths_, widths_h);
  Kokkos::deep_copy(centers_, centers_h);
  Kokkos::deep_copy(x_l_, x_l_h);

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Mesh :: Create uniform grid", DevExecSpace(),
      ilo - 1, ihi + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        grid_(i, 0) = x_l_(i);
        sqrt_gm_(i, 0) = get_sqrt_gm(x_l_(i));
        for (int q = 1; q < nNodes_ + 1; q++) {
          grid_(i, q) = node_coordinate(i, q - 1);
          sqrt_gm_(i, q) = get_sqrt_gm(grid_(i, q));
        }
        grid_(i, nNodes_ + 1) = x_l_(i + 1);
        sqrt_gm_(i, nNodes_ + 1) = get_sqrt_gm(x_l_(i + 1));
      });
}

/**
 * @brief logarithmic radial mesh generation
 *
 * Sets up logarithmically spaced cell edges and derives centers/widths from
 * those edges.
 */
void Mesh::create_log_grid() {

  const int ilo = 1; // first real zone
  const int ihi = nElements_; // last real zone

  auto widths_h = Kokkos::create_mirror_view(widths_);
  auto centers_h = Kokkos::create_mirror_view(centers_);
  auto x_l_h = Kokkos::create_mirror_view(x_l_);

  const double log_ratio = std::log(math::utils::ratio(xR_, xL_));
  const double dlog = log_ratio / static_cast<double>(nElements_);
  const double spacing_ratio = std::exp(dlog);

  x_l_h(0) = xL_ / spacing_ratio;
  for (int i = ilo; i <= ihi + 1; i++) {
    x_l_h(i) = xL_ * std::exp(static_cast<double>(i - ilo) * dlog);
  }
  x_l_h(ihi + 1) = xR_;
  x_l_h(ihi + 2) = xR_ * spacing_ratio;

  for (int i = ilo - 1; i <= ihi + 1; i++) {
    if (!std::isfinite(x_l_h(i)) || !std::isfinite(x_l_h(i + 1)) ||
        x_l_h(i + 1) <= x_l_h(i)) {
      throw_athelas_error("Invalid logarithmic grid construction!");
    }
    widths_h(i) = x_l_h(i + 1) - x_l_h(i);
    centers_h(i) = 0.5 * (x_l_h(i) + x_l_h(i + 1));
  }

  // copy back to device mirrors
  Kokkos::deep_copy(widths_, widths_h);
  Kokkos::deep_copy(centers_, centers_h);
  Kokkos::deep_copy(x_l_, x_l_h);

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Mesh :: Create log grid", DevExecSpace(),
      ilo - 1, ihi + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        grid_(i, 0) = x_l_(i);
        sqrt_gm_(i, 0) = get_sqrt_gm(x_l_(i));
        for (int q = 1; q < nNodes_ + 1; q++) {
          grid_(i, q) = node_coordinate(i, q - 1);
          sqrt_gm_(i, q) = get_sqrt_gm(grid_(i, q));
        }
        grid_(i, nNodes_ + 1) = x_l_(i + 1);
        sqrt_gm_(i, nNodes_ + 1) = get_sqrt_gm(x_l_(i + 1));
      });
}

/**
 * Compute the fixed reference mass measure from the initial geometry and
 * density.
 **/
KOKKOS_FUNCTION
void Mesh::compute_mass_measure(AthelasArray3D<double> evolved) {
  const int nNodes_ = n_nodes();
  const int ilo = get_ilo();
  const int ihi = get_ihi();
  constexpr int idx_tau = 0;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Mesh :: Compute mass", DevExecSpace(), ilo,
      ihi, KOKKOS_CLASS_LAMBDA(const int i) {
        double mass = 0.0;
        for (int q = 0; q < nNodes_; q++) {
          const double rho = 1.0 / evolved(i, q, idx_tau);
          dm_deta_(i, q) = sqrt_gm_(i, q + 1) * rho * widths_(i);
          mass += weights_(q) * dm_deta_(i, q);
        }
        mass_(i) = mass;
        // if (do_geometry()) {
        //   mass_(i) *= constants::FOURPI;
        // }
      });

  // Guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Mesh :: Compute mass (ghosts)",
      DevExecSpace(), 0, ilo - 1, KOKKOS_CLASS_LAMBDA(const int i) {
        mass_(ilo - 1 - i) = mass_(ilo + i);
        mass_(ihi + 1 + i) = mass_(ihi - i);
        for (int q = 0; q < nNodes_; q++) {
          dm_deta_(ilo - 1 - i, q) = dm_deta_(ilo + i, q);
          dm_deta_(ihi + 1 + i, q) = dm_deta_(ihi - i, q);
        }
      });
}

/**
 * @brief Compute enclosed mass at the interfaces and nodes.
 *
 * Interface masses are a cross-cell prefix sum of the cell masses; interior
 * nodes use the cumulative reference-mass partial integral, with a whole-cell
 * linear fallback when the high-order partials are non-monotone.
 **/
KOKKOS_FUNCTION
void Mesh::compute_mass_r(AthelasArray3D<double> /*evolved*/) {
  const int nNodes_ = n_nodes();
  const int ilo = get_ilo();
  const int ihi = get_ihi();

  static const double geom_fac = (do_geometry()) ? 4.0 * constants::PI : 1.0;

  athelas::par_scan(
      DEFAULT_FLAT_LOOP_PATTERN, "Mesh :: enclosed mass faces", DevExecSpace(),
      ilo, ihi,
      KOKKOS_CLASS_LAMBDA(const int i, double &partial_sum,
                          const bool is_final) {
        double dm_cell = 0.0;
        for (int q = 0; q < nNodes_; ++q) {
          dm_cell += weights_(q) * dm_deta_(i, q);
        }

        if (is_final) {
          mass_r_(i, 0) = geom_fac * partial_sum;
        }
        partial_sum += dm_cell;
        if (is_final) {
          mass_r_(i, nNodes_ + 1) = geom_fac * partial_sum;
        }
      });

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Mesh :: enclosed mass nodes", DevExecSpace(),
      ilo, ihi, KOKKOS_CLASS_LAMBDA(const int i) {
        const double mass_left = mass_r_(i, 0);
        const double mass_right = mass_r_(i, nNodes_ + 1);

        Kokkos::Array<double, MAX_ORDER> dm;
        for (int p = 0; p < nNodes_; ++p) {
          dm[p] = dm_deta_(i, p);
        }

        // High-order enclosed mass at each node from the cumulative partial
        // integral. Accept the whole cell only if the partials are monotone
        // (increasing, within the face masses); otherwise fall back to the
        // linear volume-fraction interpolation for the whole cell.
        Kokkos::Array<double, MAX_ORDER> m_node;
        bool monotone = true;
        double m_prev = mass_left;
        for (int q = 0; q < nNodes_; ++q) {
          double partial_mass = 0.0;
          for (int p = 0; p < nNodes_; ++p) {
            partial_mass += integration_matrix_(q, p) * dm[p];
          }
          m_node[q] = mass_left + geom_fac * partial_mass;
          if (m_node[q] <= m_prev || m_node[q] >= mass_right) {
            monotone = false;
          }
          m_prev = m_node[q];
        }

        if (monotone) {
          for (int q = 0; q < nNodes_; ++q) {
            mass_r_(i, q + 1) = m_node[q];
          }
        } else {
          const double x_left = coordinate_volume(grid_(i, 0));
          const double dx = coordinate_volume(grid_(i, nNodes_ + 1)) - x_left;
          for (int q = 0; q < nNodes_; ++q) {
            const double x_node = coordinate_volume(grid_(i, q + 1));
            const double theta_raw =
                (dx > 0.0) ? (x_node - x_left) / dx : (nodes_(q) + 0.5);
            mass_r_(i, q + 1) = mass_left + std::clamp(theta_raw, 0.0, 1.0) *
                                                (mass_right - mass_left);
          }
        }
      });

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Mesh :: enclosed mass ghosts", DevExecSpace(),
      0, ilo - 1, KOKKOS_CLASS_LAMBDA(const int i) {
        for (int q = 0; q < nNodes_ + 2; ++q) {
          mass_r_(ilo - 1 - i, q) = mass_r_(ilo + i, q);
          mass_r_(ihi + 1 + i, q) = mass_r_(ihi - i, q);
        }
      });
}

KOKKOS_FUNCTION
auto Mesh::enclosed_mass(const int ix, const int q) const noexcept -> double {
  return mass_r_(ix, q);
}

/**
 * Compute cell centers of masses reference coordinates
 **/
KOKKOS_FUNCTION
void Mesh::compute_center_of_mass(AthelasArray3D<double> /*evolved*/) {
  const int nNodes_ = n_nodes();
  const int ilo = get_ilo();
  const int ihi = get_ihi();

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Mesh :: center of mass", DevExecSpace(), ilo,
      ihi, KOKKOS_CLASS_LAMBDA(const int i) {
        double com = 0.0;

        for (int q = 0; q < nNodes_; q++) {
          com += nodes_(q) * weights_(q) * dm_deta_(i, q);
        }

        center_of_mass_(i) = com / mass_(i);
      });

  // Guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Mesh :: center of mass (ghosts))",
      DevExecSpace(), 0, ilo - 1, KOKKOS_CLASS_LAMBDA(const int i) {
        center_of_mass_(ilo - 1 - i) = center_of_mass_(ilo + i);
        center_of_mass_(ihi + 1 + i) = center_of_mass_(ihi - i);
      });
}

// The interior grid is recovered entirely from the cumulative reference-mass
// integral of tau; reconstruction only needs a single fixed point for the
// absolute position. That point is the inner-face radius `x_inner`, which the
// caller advances by the inner interface velocity v*(ilo).
void Mesh::reconstruct_mesh(AthelasArray3D<double> evolved,
                            const double x_inner) {
  const int ilo = get_ilo();
  const int ihi = get_ihi();
  constexpr int idx_tau = 0;

  // In the below, "X" is a generalized volume coordinate.
  // X = r or (R^3)/3 for Cartesian and spherical, respectively.
  const double X_inner = coordinate_volume(x_inner);

  athelas::par_scan(
      DEFAULT_FLAT_LOOP_PATTERN, "Mesh :: Reconstruct from tau", DevExecSpace(),
      ilo, ihi,
      KOKKOS_CLASS_LAMBDA(const int i, double &partial_sum,
                          const bool is_final) {
        // Per-node reference-mass increment mu*tau, evaluated once and reused
        // for the cell total (dX) and the within-cell partial integrals below.
        Kokkos::Array<double, MAX_ORDER> mu_tau;
        for (int q = 0; q < nNodes_; ++q) {
          mu_tau[q] = dm_deta_(i, q) * evolved(i, q, idx_tau);
        }

        double dX = 0.0;
        for (int q = 0; q < nNodes_; ++q) {
          dX += weights_(q) * mu_tau[q];
        }

        const double X_left = X_inner + partial_sum;
        partial_sum += dX;
        const double X_right = X_inner + partial_sum;

        if (is_final) {
          const double r_left = coordinate_from_volume(X_left);
          const double r_right = coordinate_from_volume(X_right);

          x_l_(i) = r_left;
          if (i == ihi) {
            x_l_(i + 1) = r_right;
          }
          centers_(i) = 0.5 * (r_left + r_right);
          widths_(i) = r_right - r_left;

          grid_(i, 0) = r_left;
          sqrt_gm_(i, 0) = get_sqrt_gm(r_left);

          // High-order node placement from the cumulative reference-mass volume
          // integral X_q = X_left + \int_{-1/2}^{eta_q} mu*tau deta. Decide
          // monotonicity for the whole cell first, then commit to a single
          // placement: all high-order if every partial integral is strictly
          // increasing and between the faces, otherwise use the fixed enclosed
          // mass fraction so gravity and reconstruction share the same nodal
          // mass coordinate.
          Kokkos::Array<double, MAX_ORDER> X_node;
          bool monotone = true;
          double X_prev = X_left;
          for (int q = 0; q < nNodes_; ++q) {
            double X_q = X_left;
            for (int p = 0; p < nNodes_; ++p) {
              X_q += integration_matrix_(q, p) * mu_tau[p];
            }
            X_node[q] = X_q;
            if (Kokkos::isnan(X_q) || X_q <= X_prev || X_q >= X_right) {
              monotone = false;
            }
            X_prev = X_q;
          }

          const double mass_left = mass_r_(i, 0);
          const double mass_width = mass_r_(i, nNodes_ + 1) - mass_left;
          for (int q = 0; q < nNodes_; ++q) {
            const double theta_m =
                (mass_width > 0.0)
                    ? std::clamp((mass_r_(i, q + 1) - mass_left) / mass_width,
                                 0.0, 1.0)
                    : nodes_(q) + 0.5;
            const double X_fallback = X_left + theta_m * (X_right - X_left);
            const double r_q = monotone ? coordinate_from_volume(X_node[q])
                                        : coordinate_from_volume(X_fallback);
            grid_(i, q + 1) = r_q;
            sqrt_gm_(i, q + 1) = get_sqrt_gm(r_q);
          }
          grid_(i, nNodes_ + 1) = r_right;
          sqrt_gm_(i, nNodes_ + 1) = get_sqrt_gm(r_right);
        }
      });

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Mesh :: Reconstruct ghosts", DevExecSpace(),
      0, 1, KOKKOS_CLASS_LAMBDA(const int side) {
        const int i = (side == 0) ? ilo - 1 : ihi + 1;
        const int interior = (side == 0) ? ilo : ihi;
        const double dr = widths_(interior);

        if (side == 0) {
          x_l_(i) = x_l_(ilo) - dr;
          x_l_(i + 1) = x_l_(ilo);
        } else {
          x_l_(i) = x_l_(ihi + 1);
          x_l_(i + 1) = x_l_(ihi + 1) + dr;
        }

        centers_(i) = 0.5 * (x_l_(i) + x_l_(i + 1));
        widths_(i) = x_l_(i + 1) - x_l_(i);
        grid_(i, 0) = x_l_(i);
        sqrt_gm_(i, 0) = get_sqrt_gm(grid_(i, 0));
        for (int q = 1; q <= nNodes_; ++q) {
          grid_(i, q) = node_coordinate(i, q - 1);
          sqrt_gm_(i, q) = get_sqrt_gm(grid_(i, q));
        }
        grid_(i, nNodes_ + 1) = x_l_(i + 1);
        sqrt_gm_(i, nNodes_ + 1) = get_sqrt_gm(grid_(i, nNodes_ + 1));
      });
}

// Access by (element, node)
KOKKOS_FUNCTION
auto Mesh::operator()(int i, int j) -> double & { return grid_(i, j); }

KOKKOS_FUNCTION
auto Mesh::operator()(int i, int j) const -> double { return grid_(i, j); }

[[nodiscard]] auto Mesh::widths() const -> AthelasArray1D<double> {
  return widths_;
}
[[nodiscard]] auto Mesh::weights() const -> AthelasArray1D<double> {
  return weights_;
}
[[nodiscard]] auto Mesh::nodes() const -> AthelasArray1D<double> {
  return nodes_;
}
[[nodiscard]] auto Mesh::x_l() const -> AthelasArray1D<double> { return x_l_; }
[[nodiscard]] auto Mesh::mass() const -> AthelasArray1D<double> {
  return mass_;
}
[[nodiscard]] auto Mesh::dm_deta() const -> AthelasArray2D<double> {
  return dm_deta_;
}
[[nodiscard]] auto Mesh::enclosed_mass() const -> AthelasArray2D<double> {
  return mass_r_;
}
[[nodiscard]] auto Mesh::centers() const -> AthelasArray1D<double> {
  return centers_;
}
[[nodiscard]] auto Mesh::centers() -> AthelasArray1D<double> {
  return centers_;
}
[[nodiscard]] auto Mesh::nodal_grid() -> AthelasArray2D<double> {
  return grid_;
}
[[nodiscard]] auto Mesh::nodal_grid() const -> AthelasArray2D<double> {
  return grid_;
}
[[nodiscard]] auto Mesh::sqrt_gm() const -> AthelasArray2D<double> {
  return sqrt_gm_;
}

} // namespace athelas
