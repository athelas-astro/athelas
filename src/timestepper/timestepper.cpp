/**
 * @file timestepper.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Primary time marching routine
 */

#include <vector>

#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "timestepper/tableau.hpp"
#include "timestepper/timestepper.hpp"

namespace athelas {

using eos::EOS;

/**
 * The constructor creates the necessary data structures for time evolution.
 * Lots of structures used in discretizations live here.
 **/
TimeStepper::TimeStepper(const ProblemIn *pin, GridStructure *grid, EOS *eos)
    : nvars_evolved_(nvars_evolved(pin)), mSize_(grid->n_elements() + 2),
      integrator_(
          create_tableau(pin->param()->get<MethodID>("time.integrator"))),
      nStages_(integrator_.num_stages), tOrder_(integrator_.explicit_order),
      SumVar_U_("SumVar_U", mSize_ + 1, pin->param()->get<int>("fluid.porder"),
                nvars_evolved_),
      grid_s_(nStages_ + 1, GridStructure(pin)),
      stage_data_("StageData", nStages_ + 1, mSize_ + 1), eos_(eos) {}

[[nodiscard]] auto TimeStepper::n_stages() const noexcept -> int {
  return integrator_.num_stages;
}

// Computes number of evolved vars.
// Can't be used for mass fractions when mixing is considered
// Will have to remove / change at that point.
[[nodiscard]] auto TimeStepper::nvars_evolved(const ProblemIn *pin) noexcept
    -> int {
  static const int base = 3;
  static const bool rad_enabled = pin->param()->get<bool>("physics.rad_active");
  static const bool nickel_enabled =
      pin->param()->get<bool>("physics.heating.nickel.enabled");

  int additional_vars = 0;
  if (rad_enabled) {
    additional_vars += 2;
  }

  if (nickel_enabled) {
    additional_vars += 3;
  }

  return base + additional_vars;
}

} // namespace athelas
