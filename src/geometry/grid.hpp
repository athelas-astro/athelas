#pragma once

#include "Kokkos_Macros.hpp"
#include "kokkos_types.hpp"
#include "pgen/problem_in.hpp"

namespace athelas {

enum class Geometry { Planar, Spherical };

enum class Domain { Interior, Entire };

enum class Interface { Left, Right };

class GridStructure {
 public:
  explicit GridStructure(const ProblemIn *pin);
  GridStructure() = default;
  [[nodiscard]] auto centers(int iC) const -> double;
  [[nodiscard]] auto get_nodes(int nN) const -> double;
  [[nodiscard]] auto get_x_l() const noexcept -> double;
  [[nodiscard]] auto get_x_r() const noexcept -> double;
  [[nodiscard]] auto get_sqrt_gm(double X) const -> double;

  [[nodiscard]] auto do_geometry() const noexcept -> bool;

  [[nodiscard]] static auto get_ilo() noexcept -> int;
  [[nodiscard]] auto get_ihi() const noexcept -> int;
  [[nodiscard]] auto n_nodes() const noexcept -> int;
  [[nodiscard]] auto n_elements() const noexcept -> int;

  void create_grid(const ProblemIn *pin);
  void create_uniform_grid();
  void create_log_grid();

  void update_grid(AthelasArray1D<double> SData);
  void compute_mass(AthelasArray3D<double> uPF);
  void compute_mass_r(AthelasArray3D<double> uPF);
  auto enclosed_mass(int ix, int iN) const noexcept -> double;
  void compute_center_of_mass(AthelasArray3D<double> uPF);
  void compute_center_of_mass_radius(AthelasArray3D<double> uPF);

  [[nodiscard]] auto x_l() const -> AthelasArray1D<double>;
  [[nodiscard]] auto widths() const -> AthelasArray1D<double>;
  [[nodiscard]] auto weights() const -> AthelasArray1D<double>;
  [[nodiscard]] auto nodes() const -> AthelasArray1D<double>;
  [[nodiscard]] auto mass() const -> AthelasArray1D<double>;
  [[nodiscard]] auto enclosed_mass() const -> AthelasArray2D<double>;
  [[nodiscard]] auto centers() const -> AthelasArray1D<double>;
  [[nodiscard]] auto centers() -> AthelasArray1D<double>;
  [[nodiscard]] auto nodal_grid() -> AthelasArray2D<double>;
  [[nodiscard]] auto nodal_grid() const -> AthelasArray2D<double>;
  [[nodiscard]] auto sqrt_gm() const -> AthelasArray2D<double>;

  // domain
  template <Domain D>
  [[nodiscard]] auto domain() const noexcept -> std::pair<int, int> {
    if constexpr (D == Domain::Interior) {
      return {1, nElements_};
    } else if constexpr (D == Domain::Entire) {
      return {0, nElements_ + 1};
    }
  }

  template <Domain D>
  [[nodiscard]] auto nodal_domain() const noexcept -> std::pair<int, int> {
    if constexpr (D == Domain::Interior) {
      return {1, nNodes_};
    } else if constexpr (D == Domain::Entire) {
      return {0, nNodes_ + 1};
    }
  }

  // Give physical grid coordinate from a node.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION auto node_coordinate(const int iC,
                                                            const int q) const
      -> double {
    return x_l_(iC) * shape_function(0, nodes_(q)) +
           x_l_(iC + 1) * shape_function(1, nodes_(q));
  }

  KOKKOS_FUNCTION
  auto operator()(int i, int j) -> double &;
  KOKKOS_FUNCTION
  auto operator()(int i, int j) const -> double;

 private:
  int nElements_;
  int nNodes_;
  int mSize_;

  double xL_;
  double xR_;

  Geometry geometry_;
  std::string grid_type_; // uniform or logarithmic

  AthelasArray1D<double> nodes_;
  AthelasArray1D<double> weights_;

  AthelasArray1D<double> centers_;
  AthelasArray1D<double> widths_;
  AthelasArray1D<double> x_l_; // left interface coordinate

  AthelasArray1D<double> mass_; // cell mass
  AthelasArray2D<double> mass_r_; // enclosed mass
  AthelasArray1D<double> center_of_mass_;

  AthelasArray2D<double> sqrt_gm_;
  AthelasArray2D<double> grid_;

  // linear shape function on the reference element
  KOKKOS_INLINE_FUNCTION
  auto static shape_function(const int interface, const double eta) -> double {
    if (interface == 0) {
      return 1.0 * (0.5 - eta);
    }
    if (interface == 1) {
      return 1.0 * (0.5 + eta);
    }
    return 0.0; // unreachable, but silences warnings
  }
};

} // namespace athelas
