#pragma once

#include <memory>

#include "atom/atom.hpp"
#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "history/history.hpp"
#include "interface/packages_base.hpp"
#include "limiters/slope_limiter_utilities.hpp"
#include "pgen/problem_in.hpp"
#include "timestepper/operator_split_stepper.hpp"
#include "timestepper/timestepper.hpp"

namespace athelas {

using atom::AtomicData;
using bc::BoundaryConditions;

/**
 * @class Driver
 * @brief the primary executor of the simulation.
 * Owns key data and calls timestepper, IO.
 */
class Driver {
 public:
  //  explicit Driver(std::shared_ptr<ProblemIn> pin);
  // Driver
  explicit Driver(std::shared_ptr<ProblemIn> pin) // NOLINT
      : pin_(pin), manager_(std::make_unique<PackageManager>()),
        split_manager_(std::make_unique<PackageManager>()),
        restart_(pin->param()->get<bool>("problem.restart")),
        bcs_(std::make_unique<BoundaryConditions>(
            bc::make_boundary_conditions(pin.get()))),
        time_(0.0), dt_(pin_->param()->get<double>("output.dt_init")),
        t_end_(pin->param()->get<double>("problem.tf")),
        eos_(std::make_unique<eos::EOS>(eos::initialize_eos(pin.get()))),
        opac_(std::make_unique<Opacity>(initialize_opacity(pin.get()))),
        grid_(pin.get()), sl_hydro_(initialize_slope_limiter(
                              "fluid", &grid_, pin.get(), {0, 1, 2}, 3)),
        sl_rad_(initialize_slope_limiter("radiation", &grid_, pin.get(), {3, 4},
                                         2)), // update
        ssprk_(pin.get(), &grid_, eos_.get()),
        history_(std::make_unique<HistoryOutput>(
            pin->param()->get<std::string>("output.hist_fn"),
            pin->param()->get<std::string>("output.dir"),
            pin->param()->get<bool>("output.history_enabled"))),
        mesh_state_(pin.get(), ssprk_.n_stages()) {
    initialize(pin.get());
  }

  auto execute() -> int;

 private:
  // init
  void initialize(ProblemIn *pin);

  void post_init_work();
  void post_step_work();

  std::shared_ptr<ProblemIn> pin_;

  std::unique_ptr<PackageManager> manager_;
  std::unique_ptr<PackageManager> split_manager_;

  // TODO(astrobarker): thread in run_id_
  // std::string run_id_;
  bool restart_;

  std::unique_ptr<BoundaryConditions> bcs_;

  double time_{};
  double dt_;
  double t_end_;

  // core bits
  // TODO(astrobarker): keep eos_, opac_ in packages.
  std::unique_ptr<eos::EOS> eos_;
  std::unique_ptr<Opacity> opac_;
  GridStructure grid_;

  // slope limiters
  SlopeLimiter sl_hydro_;
  SlopeLimiter sl_rad_;

  // timesteppers
  TimeStepper ssprk_;
  std::unique_ptr<OperatorSplitStepper> split_stepper_;
  bool operator_split_physics_ = false;

  // history
  std::unique_ptr<HistoryOutput> history_;

  // bases
  std::unique_ptr<basis::ModalBasis> fluid_basis_; // init in constr body
  std::unique_ptr<basis::ModalBasis> radiation_basis_; // init in constr body

  MeshState mesh_state_;
}; // class Driver

} // namespace athelas

namespace {

/**
 * Compute the CFL timestep restriction.
 **/

inline auto compute_cfl(const double CFL, const int order) -> double {
  double c = 1.0;

  const double max_cfl = 0.95;
  return std::min(c * CFL / ((2.0 * (order)-1.0)), max_cfl);
}
} // namespace
