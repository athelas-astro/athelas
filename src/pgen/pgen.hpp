/**
 * @file pgen.hpp
 * --------------
 *
 * @brief Top level problem initialization
 *
 * @details Calls specific problem pgen functions.
 */

#pragma once

#include <string>
#include <string_view>
#include <unordered_map>

#include "geometry/mesh.hpp"
#include "interface/state.hpp"
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
#include "pgen/shocktube.hpp"
#include "pgen/shu_osher.hpp"
#include "pgen/smooth_flow.hpp"
#include "utils/error.hpp"

namespace athelas {

/**
 * Initialize the mesh_state for various problems.
 **/
void initialize_fields(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin) {
  std::print("# Running problem generator... ");

  using init_fn = void (*)(MeshState &, Mesh *, ProblemIn *);
  static const std::unordered_map<std::string_view, init_fn> pgen_registry = {
      {"supernova", &pgen::progenitor::init},
      {"shocktube", &pgen::shocktube::init},
      {"shu_osher", &pgen::shu_osher::init},
      {"moving_contact", &pgen::moving_contact::init},
      {"hydrostatic_balance", &pgen::hydrostatic_balance::init},
      {"smooth_advection", &pgen::advection::init},
      {"sedov", &pgen::sedov::init},
      {"noh", &pgen::noh::init},
      {"shockless_noh", &pgen::shockless_noh::init},
      {"smooth_flow", &pgen::smooth_flow::init},
      {"ejecta_csm", &pgen::ejecta_csm::init},
      {"rad_equilibrium", &pgen::rad_equilibrium::init},
      {"rad_advection", &pgen::rad_advection::init},
      {"rad_shock_steady", &pgen::rad_shock_steady::init},
      {"rad_shock", &pgen::rad_shock::init},
      {"marshak", &pgen::marshak::init},
      {"one_zone_ionization", &pgen::one_zone_ionization::init},
      {"ni_decay", &pgen::ni_decay::init},
  };

  const auto problem_name = pin->param()->get<std::string>("problem.name");
  const auto it = pgen_registry.find(problem_name);
  if (it == pgen_registry.end()) {
    throw_athelas_error("Please choose a valid problem_name!");
  }
  it->second(mesh_state, mesh, pin);

  std::println(" .. complete!\n");
}

} // namespace athelas
