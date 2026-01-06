#pragma once

#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "geometry/grid.hpp"
#include "pgen/problem_in.hpp"
#include "state/state.hpp"

namespace athelas::geometry {

namespace pkg_vars {
constexpr int Velocity = 0;
} // namespace pkg_vars

class GeometryPackage {
 public:
  GeometryPackage(const ProblemIn *pin, basis::ModalBasis *basis, int n_stages,
                  bool active = true);

  void update_explicit(const StageData &stage_data, const GridStructure &grid,
                       const TimeStepInfo &dt_info);

  void apply_delta(AthelasArray3D<double> lhs,
                   const TimeStepInfo &dt_info) const;

  void zero_delta() const noexcept;

  [[nodiscard]] auto min_timestep(const StageData & /*stage_data*/,
                                  const GridStructure & /*grid*/,
                                  const TimeStepInfo & /*dt_info*/) const
      -> double;

  [[nodiscard]] auto name() const noexcept -> std::string_view;

  [[nodiscard]] auto is_active() const noexcept -> bool;

  void fill_derived(StageData &stage_data, const GridStructure &grid,
                    const TimeStepInfo &dt_info) const;

  void set_active(bool active);

 private:
  bool active_;
  int order_;
  basis::ModalBasis *basis_;
  AthelasArray4D<double> delta_; // [nstages, nx, order, nvars]
};

} // namespace athelas::geometry
