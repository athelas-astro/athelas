/**
 * @file initialization.hpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Top level problem initialization
 *
 * @details Calls specific problem pgen functions.
 */

#pragma once

#include <string>

#include "geometry/grid.hpp"
#include "pgen/advection.hpp"
#include "pgen/ejecta_csm.hpp"
#include "pgen/hydrostatic_balance.hpp"
#include "pgen/marshak.hpp"
#include "pgen/moving_contact.hpp"
#include "pgen/ni_decay.hpp"
#include "pgen/noh.hpp"
#include "pgen/one_zone_ionization.hpp"
#include "pgen/problem_in.hpp"
#include "pgen/progenitor.hpp"
#include "pgen/rad_advection.hpp"
#include "pgen/rad_equilibrium.hpp"
#include "pgen/rad_shock.hpp"
#include "pgen/rad_shock_steady.hpp"
#include "pgen/sedov.hpp"
#include "pgen/shockless_noh.hpp"
#include "pgen/shu_osher.hpp"
#include "pgen/smooth_flow.hpp"
#include "pgen/sod.hpp"
#include "state/state.hpp"
#include "utils/error.hpp"

namespace athelas {

/**
 * Initialize the mesh_state for various problems.
 **/
void initialize_fields(MeshState &mesh_state, GridStructure *grid,
                       ProblemIn *pin) {
  std::print("# Running problem generator... ");

  const auto problem_name = pin->param()->get<std::string>("problem.problem");

  // This is clunky and not elegant but it works.
  if (problem_name == "supernova") {
    progenitor_init(mesh_state, grid, pin);
  } else if (problem_name == "sod") {
    sod_init(mesh_state, grid, pin);
  } else if (problem_name == "shu_osher") {
    shu_osher_init(mesh_state, grid, pin);
  } else if (problem_name == "moving_contact") {
    moving_contact_init(mesh_state, grid, pin);
  } else if (problem_name == "hydrostatic_balance") {
    hydrostatic_balance_init(mesh_state, grid, pin);
  } else if (problem_name == "smooth_advection") {
    advection_init(mesh_state, grid, pin);
  } else if (problem_name == "sedov") {
    sedov_init(mesh_state, grid, pin);
  } else if (problem_name == "noh") {
    noh_init(mesh_state, grid, pin);
  } else if (problem_name == "shockless_noh") {
    shockless_noh_init(mesh_state, grid, pin);
  } else if (problem_name == "smooth_flow") {
    smooth_flow_init(mesh_state, grid, pin);
  } else if (problem_name == "ejecta_csm") {
    ejecta_csm_init(mesh_state, grid, pin);
  } else if (problem_name == "rad_equilibrium") {
    rad_equilibrium_init(mesh_state, grid, pin);
  } else if (problem_name == "rad_advection") {
    rad_advection_init(mesh_state, grid, pin);
  } else if (problem_name == "rad_shock_steady") {
    rad_shock_steady_init(mesh_state, grid, pin);
  } else if (problem_name == "rad_shock") {
    rad_shock_init(mesh_state, grid, pin);
  } else if (problem_name == "marshak") {
    marshak_init(mesh_state, grid, pin);
  } else if (problem_name == "one_zone_ionization") {
    one_zone_ionization_init(mesh_state, grid, pin);
  } else if (problem_name == "ni_decay") {
    ni_decay_init(mesh_state, grid, pin);
  } else {
    throw_athelas_error("Please choose a valid problem_name!");
  }
  std::println(" .. complete!\n");
}

} // namespace athelas
