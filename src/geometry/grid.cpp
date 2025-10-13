/**
 * @file grid.cpp
 * --------------
 *
 * @brief Class for holding the spatial grid.
 *
 * @details This class GridStructure holds key pieces of the grid:
 *          - nx
 *          - nnodes
 *          - weights
 *
 *          For a loop over real zones, loop from ilo to ihi (inclusive).
 *          ilo = 1 (one ghost cell)
 *          ihi = nElements_
 */

#include <vector>

#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"
#include "quadrature/quadrature.hpp"
#include "utils/utilities.hpp"

namespace athelas {

GridStructure::GridStructure(const ProblemIn *pin)
    : nElements_(pin->param()->get<int>("problem.nx")),
      nNodes_(pin->param()->get<int>("fluid.nnodes")), mSize_(nElements_ + 2),
      xL_(pin->param()->get<double>("problem.xl")),
      xR_(pin->param()->get<double>("problem.xr")),
      geometry_(pin->param()->get<Geometry>("problem.geometry_model")),
      grid_type_(pin->param()->get<std::string>("problem.grid_type")),
      nodes_("Nodes", nNodes_), weights_("weights_", nNodes_),
      centers_("Centers", mSize_), widths_("widths_", mSize_),
      x_l_("Left Interface", mSize_ + 1), mass_("Cell mass_", mSize_),
      mass_r_("Enclosed mass", mSize_, nNodes_),
      center_of_mass_("Center of mass_", mSize_),
      sqrt_gm_("Sqrt Gamma", mSize_, nNodes_ + 2),
      grid_("Grid", mSize_, nNodes_) {
  std::vector<double> tmp_nodes(nNodes_);
  std::vector<double> tmp_weights(nNodes_);

  for (int q = 0; q < nNodes_; q++) {
    tmp_nodes[q] = 0.0;
    tmp_weights[q] = 0.0;
  }

  quadrature::lg_quadrature(nNodes_, tmp_nodes, tmp_weights);

  // TODO(astrobarker): use host copies for this.
  for (int q = 0; q < nNodes_; q++) {
    nodes_(q) = tmp_nodes[q];
    weights_(q) = tmp_weights[q];
  }

  create_grid(pin);
}

// Return cell center
auto GridStructure::centers(int iC) const -> double { return centers_(iC); }

// Return given quadrature node
auto GridStructure::get_nodes(int nN) const -> double { return nodes_(nN); }

// Accessor for xL
auto GridStructure::get_x_l() const noexcept -> double { return xL_; }

// Accessor for xR
auto GridStructure::get_x_r() const noexcept -> double { return xR_; }

// Accessor for SqrtGm
auto GridStructure::get_sqrt_gm(const double X) const -> double {
  if (geometry_ == Geometry::Spherical) [[likely]] {
    return X * X;
  }
  return 1.0;
}

// Accessor for x_l_
KOKKOS_FUNCTION
auto GridStructure::get_left_interface(int ix) const -> double {
  return x_l_(ix);
}

// Return nNodes_
KOKKOS_FUNCTION
auto GridStructure::n_nodes() const noexcept -> int { return nNodes_; }

// Return nElements_
KOKKOS_FUNCTION
auto GridStructure::n_elements() const noexcept -> int { return nElements_; }

// Return first physical zone
KOKKOS_FUNCTION
auto GridStructure::get_ilo() noexcept -> int { return 1; }

// Return last physical zone
KOKKOS_FUNCTION
auto GridStructure::get_ihi() const noexcept -> int { return nElements_; }

// Return true if in spherical symmetry
KOKKOS_FUNCTION
auto GridStructure::do_geometry() const noexcept -> bool {
  return geometry_ == Geometry::Spherical;
}

// grid creation logic
void GridStructure::create_grid(const ProblemIn *pin) {
  if (utilities::to_lower(grid_type_) == "uniform") {
    create_uniform_grid();
  } else if (utilities::to_lower(grid_type_) == "logarithmic") {
    // Need to be careful of coordinates with log grid!
    if (xL_ < 0.0 || xR_ < 0.0) {
      THROW_ATHELAS_ERROR(
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
      WARNING_ATHELAS("Logarithmic grid requestion with XL = 0.0. This does "
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
    THROW_ATHELAS_ERROR("Unknown grid type '" + grid_type_ + "' provided!");
  }
}

/**
 * @brief uniform mesh
 */
void GridStructure::create_uniform_grid() {

  const int ilo = 1; // first real zone
  const int ihi = nElements_; // last real zone

  auto widths_h = Kokkos::create_mirror_view(widths_);
  auto centers_h = Kokkos::create_mirror_view(centers_);
  auto x_l_h = Kokkos::create_mirror_view(x_l_);

  for (int i = 0; i < nElements_ + 2; i++) {
    widths_h(i) = (xR_ - xL_) / nElements_;
  }

  x_l_h(1) = xL_;
  for (int ix = 2; ix < nElements_ + 2; ix++) {
    x_l_h(ix) = x_l_h(ix - 1) + widths_h(ix - 1);
  }

  centers_h(ilo) = xL_ + 0.5 * widths_h(ilo);
  for (int i = ilo + 1; i <= ihi; i++) {
    centers_h(i) = centers_h(i - 1) + widths_h(i - 1);
  }

  for (int i = ilo - 1; i >= 0; i--) {
    centers_h(i) = centers_h(i + 1) - widths_h(i + 1);
  }
  for (int i = ihi + 1; i < nElements_ + 1 + 1; i++) {
    centers_h(i) = centers_h(i - 1) + widths_h(i - 1);
  }

  // copy back to device mirrors
  Kokkos::deep_copy(widths_, widths_h);
  Kokkos::deep_copy(centers_, centers_h);
  Kokkos::deep_copy(x_l_, x_l_h);

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Grid :: Create uniform grid", DevExecSpace(),
      ilo, ihi, KOKKOS_CLASS_LAMBDA(const int i) {
        sqrt_gm_(i, 0) = get_sqrt_gm(x_l_(i));
        for (int q = 1; q < nNodes_ + 1; q++) {
          grid_(i, q - 1) = node_coordinate(i, q - 1);
          sqrt_gm_(i, q) = get_sqrt_gm(grid_(i, q - 1));
        }
        sqrt_gm_(i, nNodes_ + 1) = get_sqrt_gm(x_l_(i));
      });
}

/**
 * @brief logarithmic radial mesh generation
 *
 * Sets up logarithmic mesh with cell centers:
 * x_i = x_l * (x_r / x_l)^(i/(nx - 1))
 */
void GridStructure::create_log_grid() {

  const int ilo = 1; // first real zone
  const int ihi = nElements_; // last real zone

  auto widths_h = Kokkos::create_mirror_view(widths_);
  auto centers_h = Kokkos::create_mirror_view(centers_);
  auto x_l_h = Kokkos::create_mirror_view(x_l_);

  const double log_xl = std::log10(xL_);
  const double log_ratio = std::log10(utilities::ratio(xR_, xL_));
  const double dx = log_ratio / (nElements_ - 1);

  // Set up cell centers
  for (int i = ilo; i <= ihi; i++) {
    const double log_xi = log_xl + i * dx;
    centers_h(i) = std::pow(10.0, log_xi);
  }

  // Handle ghost cells
  for (int i = ilo - 1; i >= 0; i--) {
    centers_h(i) = centers_h(i + 1) - widths_h(i + 1);
  }
  for (int i = ihi + 1; i < nElements_ + 1 + 1; i++) {
    centers_h(i) = centers_h(i - 1) + widths_h(i - 1);
  }

  // Set up left edges
  x_l_h(1) = xL_;
  for (int i = 2; i < nElements_ + 2; i++) {
    x_l_h(i) = 0.5 * (centers_h(i - 1) + centers_h(i));
  }

  // Calculate remaining widths with geometric progression
  for (int i = 0; i < nElements_ + 2; i++) {
    widths_h(i) = x_l_h(i + 1) - x_l_h(i);
  }

  // copy back to device mirrors
  Kokkos::deep_copy(widths_, widths_h);
  Kokkos::deep_copy(centers_, centers_h);
  Kokkos::deep_copy(x_l_, x_l_h);

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Grid :: Create log grid", DevExecSpace(), ilo,
      ihi, KOKKOS_CLASS_LAMBDA(const int i) {
        sqrt_gm_(i, 0) = get_sqrt_gm(x_l_(i));
        for (int q = 1; q < nNodes_ + 1; q++) {
          grid_(i, q - 1) = node_coordinate(i, q - 1);
          sqrt_gm_(i, q) = get_sqrt_gm(grid_(i, q - 1));
        }
        sqrt_gm_(i, nNodes_ + 1) = get_sqrt_gm(x_l_(i));
      });
}

/**
 * Compute cell masses
 **/
KOKKOS_FUNCTION
void GridStructure::compute_mass(const AthelasArray3D<double> uPF) {
  const int nNodes_ = n_nodes();
  const int ilo = get_ilo();
  const int ihi = get_ihi();

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Grid :: Compute mass", DevExecSpace(), ilo,
      ihi, KOKKOS_CLASS_LAMBDA(const int i) {
        double mass = 0.0;
        for (int q = 0; q < nNodes_; q++) {
          const double X = node_coordinate(i, q);
          mass += weights_(q) * get_sqrt_gm(X) * uPF(i, q, 0);
        }
        mass *= widths_(i);
        mass_(i) = mass;
      });

  // Guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Grid :: Compute mass (ghosts)",
      DevExecSpace(), 0, ilo - 1, KOKKOS_CLASS_LAMBDA(const int i) {
        mass_(ilo - 1 - i) = mass_(ilo + i);
        mass_(ihi + 1 + i) = mass_(ihi - i);
      });
}

/**
 * @brief Compute enclosed masses
 *
 * NOTE: This function is intended only to be called once, as it allocated.
 * If we find ourselves for any reason calling this repeatedly,
 * it should be refactored.
 **/
KOKKOS_FUNCTION
void GridStructure::compute_mass_r(const AthelasArray3D<double> uPF) {
  const int nNodes_ = n_nodes();
  const int ilo = get_ilo();
  const int ihi = get_ihi();

  static const double geom_fac = (do_geometry()) ? 4.0 * constants::PI : 1.0;

  const int total_points = (ihi - ilo + 1) * nNodes_;
  AthelasArray1D<double> mass_contrib("mass_contrib", total_points);
  AthelasArray1D<double> cumulative_mass("cumulative_mass", total_points);

  // 1: Compute individual mass contributions in parallel
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Grid :: compute mass contributions",
      DevExecSpace(), 0, total_points - 1, KOKKOS_CLASS_LAMBDA(const int idx) {
        const int ix = ilo + idx / nNodes_;
        const int q = idx % nNodes_;
        const double X = node_coordinate(ix, q);
        mass_contrib(idx) = weights_(q) * get_sqrt_gm(X) * uPF(ix, q, 0);
      });

  // 2: Perform parallel inclusive scan (cumulative sum)
  athelas::par_scan(
      DEFAULT_FLAT_LOOP_PATTERN, "Grid :: enclosed mass", DevExecSpace(), 0,
      total_points - 1,
      KOKKOS_LAMBDA(const int idx, double &partial_sum, const bool is_final) {
        partial_sum += mass_contrib(idx);
        if (is_final) {
          cumulative_mass(idx) = partial_sum;
        }
      });

  // 3: sort into mass_r_
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Grid :: store enclosed mass", DevExecSpace(),
      0, total_points - 1, KOKKOS_CLASS_LAMBDA(const int idx) {
        const int ix = ilo + idx / nNodes_;
        const int q = idx % nNodes_;
        mass_r_(ix, q) = cumulative_mass(idx) * widths_(ix) * geom_fac;
      });

  // Get total mass
  // auto h_cumulative =
  // Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cumulative_mass);
  // mass = h_cumulative(total_points - 1);
}

KOKKOS_FUNCTION
auto GridStructure::enclosed_mass(const int ix, const int q) const noexcept
    -> double {
  return mass_r_(ix, q);
}

/**
 * Compute cell centers of masses reference coordinates
 **/
KOKKOS_FUNCTION
void GridStructure::compute_center_of_mass(const AthelasArray3D<double> uPF) {
  const int nNodes_ = n_nodes();
  const int ilo = get_ilo();
  const int ihi = get_ihi();

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Grid :: center of mass", DevExecSpace(), ilo,
      ihi, KOKKOS_CLASS_LAMBDA(const int i) {
        double com = 0.0;

        for (int q = 0; q < nNodes_; q++) {
          const double X = node_coordinate(i, q);
          com += nodes_(q) * weights_(q) * get_sqrt_gm(X) * uPF(i, q, 0);
        }

        com *= widths_(i);
        center_of_mass_(i) = com / mass_(i);
      });

  // Guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Grid :: center of mass (ghosts))",
      DevExecSpace(), 0, ilo - 1, KOKKOS_CLASS_LAMBDA(const int i) {
        center_of_mass_(ilo - 1 - i) = center_of_mass_(ilo + i);
        center_of_mass_(ihi + 1 + i) = center_of_mass_(ihi - i);
      });
}

/**
 * Update grid coordinates using interface velocities.
 **/
KOKKOS_FUNCTION
void GridStructure::update_grid(const AthelasArray1D<double> SData) {

  const int ilo = get_ilo();
  const int ihi = get_ihi();

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Grid :: Update (1)", DevExecSpace(), ilo,
      ihi + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        x_l_(i) = SData(i);
        widths_(i) = SData(i + 1) - SData(i);
        centers_(i) = 0.5 * (SData(i + 1) + SData(i));
      });

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Grid :: Update (2)", DevExecSpace(), ilo,
      ihi + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        for (int q = 0; q < nNodes_; q++) {
          grid_(i, q) = node_coordinate(i, q);
        }
      });

  if (do_geometry()) {
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Grid :: Update (Sqrt Gm)", DevExecSpace(),
        ilo, ihi + 1, KOKKOS_CLASS_LAMBDA(const int i) {
          sqrt_gm_(i, 0) = get_sqrt_gm(x_l_(i));
          for (int q = 1; q < nNodes_ + 1; q++) {
            sqrt_gm_(i, q) = grid_(i, q - 1) * grid_(i, q - 1);
          }
          sqrt_gm_(i, nNodes_ + 1) = get_sqrt_gm(x_l_(i + 1));
        });
  }
}

// Access by (element, node)
KOKKOS_FUNCTION
auto GridStructure::operator()(int i, int j) -> double & { return grid_(i, j); }

KOKKOS_FUNCTION
auto GridStructure::operator()(int i, int j) const -> double {
  return grid_(i, j);
}

[[nodiscard]] auto GridStructure::widths() const -> AthelasArray1D<double> {
  return widths_;
}
[[nodiscard]] auto GridStructure::weights() const -> AthelasArray1D<double> {
  return weights_;
}
[[nodiscard]] auto GridStructure::nodes() const -> AthelasArray1D<double> {
  return nodes_;
}
[[nodiscard]] auto GridStructure::x_l() const -> AthelasArray1D<double> {
  return x_l_;
}
[[nodiscard]] auto GridStructure::mass() const -> AthelasArray1D<double> {
  return mass_;
}
[[nodiscard]] auto GridStructure::enclosed_mass() const
    -> AthelasArray2D<double> {
  return mass_r_;
}
[[nodiscard]] auto GridStructure::centers() const -> AthelasArray1D<double> {
  return centers_;
}
[[nodiscard]] auto GridStructure::centers() -> AthelasArray1D<double> {
  return centers_;
}
[[nodiscard]] auto GridStructure::nodal_grid() -> AthelasArray2D<double> {
  return grid_;
}
[[nodiscard]] auto GridStructure::nodal_grid() const -> AthelasArray2D<double> {
  return grid_;
}
[[nodiscard]] auto GridStructure::sqrt_gm() const -> AthelasArray2D<double> {
  return sqrt_gm_;
}

} // namespace athelas
