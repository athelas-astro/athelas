#include "timestepper/operator_split_stepper.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"

namespace athelas {

using eos::EOS;

OperatorSplitStepper::OperatorSplitStepper(const GridStructure &grid,
                                           const int nvars)
    : nvars_evolved_(nvars),
      dU_("OperatorSplit::dU", grid.n_elements() + 2, grid.n_nodes(), nvars) {}

void OperatorSplitStepper::step(PackageManager *pkgs, State *state,
                                const GridStructure &grid, const double t,
                                const double dt) {

  auto U = state->u_cf();

  TimeStepInfo dt_info{.t = t, .dt = dt, .dt_a = dt, .dt_coef = dt, .stage = 0};

  pkgs->update_explicit(state, dU_, grid, dt_info);
  pkgs->update_implicit_iterative(state, dU_, grid, dt_info);
  pkgs->apply_delta(U, dt_info);
}

} // namespace athelas
