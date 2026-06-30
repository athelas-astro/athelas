#pragma once

#include <memory>

#include "atom/atom.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/mesh.hpp"
#include "history/history.hpp"
#include "interface/packages_base.hpp"
#include "io/io.hpp"
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
  explicit Driver(std::shared_ptr<ProblemIn> pin,
                  std::string restart_filename = "") // NOLINT
      : pin_(pin), manager_(std::make_unique<PackageManager>()),
        split_manager_(std::make_unique<PackageManager>()),
        restart_filename_(std::move(restart_filename)),
        restart_(!restart_filename_.empty()),
        bcs_(std::make_unique<BoundaryConditions>(
            bc::make_boundary_conditions(pin.get()))),
        time_(0.0), dt_(pin_->param()->get<double>("output.dt_init")),
        t_end_(pin->param()->get<double>("problem.tf")), ssprk_(pin.get()),
        mesh_state_(pin.get(), ssprk_.n_stages()),
        sl_hydro_(initialize_slope_limiter("fluid", &mesh_state_.mesh(),
                                           pin.get(), {0, 2})),
        sl_rad_(initialize_slope_limiter("radiation", &mesh_state_.mesh(),
                                         pin.get(), {3, 4})), // update
        history_(std::make_unique<HistoryOutput>(
            pin->param()->get<std::string>("output.hist_fn"),
            pin->param()->get<std::string>("output.dir"),
            pin->param()->get<bool>("output.history_enabled"))) {
    initialize(pin.get());
  }

  auto execute() -> int;

 private:
  // init
  void initialize(ProblemIn *pin);

  void post_init_work();
  void post_step_work();

  // Refresh output-only diagnostics (e.g. the optical-depth profile field).
  // Called before HDF5 / history IO, never on the per-substep path.
  void pre_output_work();

  std::shared_ptr<ProblemIn> pin_;

  std::unique_ptr<PackageManager> manager_;
  std::unique_ptr<PackageManager> split_manager_;

  // TODO(astrobarker): thread in run_id_
  // std::string run_id_;
  std::string restart_filename_;
  bool restart_;
  io::SimInfo restart_info_{}; // populated by restart load; zero otherwise

  std::unique_ptr<BoundaryConditions> bcs_;

  double time_{};
  double dt_;
  double t_end_;

  // timesteppers
  TimeStepper ssprk_;
  std::unique_ptr<OperatorSplitStepper> split_stepper_;
  bool operator_split_physics_ = false;

  // core simulation state (owns the mesh)
  MeshState mesh_state_;

  // slope limiters
  SlopeLimiter sl_hydro_;
  SlopeLimiter sl_rad_;

  // history
  std::unique_ptr<HistoryOutput> history_;

  // diagnostics: optical-depth profile field is filled at output cadence only
  bool diag_optical_depth_enabled_{};
}; // class Driver

} // namespace athelas

namespace {

/**
 * Compute the CFL timestep restriction.
 **/

inline auto compute_cfl(const double CFL, const int nq) -> double {
  double c = 1.0;

  const double max_cfl = 0.95;
  return std::min(c * CFL / ((2.0 * (nq)-1.0)), max_cfl);
}
} // namespace
