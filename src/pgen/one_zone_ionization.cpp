#include "pgen/one_zone_ionization.hpp"

#include "basic_types.hpp"
#include "composition/composition.hpp"
#include "composition/saha.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"

namespace athelas::pgen::one_zone_ionization {

/**
 * Initialize one_zone_ionization test
 **/
void init(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin) {
  const bool ionization_active =
      pin->param()->get<bool>("physics.ionization.enabled");
  const int saha_ncomps =
      pin->param()->get<int>("ionization.ncomps"); // for ionization
  const auto ncomps =
      pin->param()->get<int>("composition.ncomps", 3); // mass fractions
  athelas_requires(ncomps == 3, "One zone ionization requires ncomps = 3.");
  athelas_requires(ionization_active,
                   "One zone ionization requires ionization enabled!");
  const auto eos_type = pin->param()->get<std::string>("eos.type");
  // athelas_requires(eos_type == "ideal", "One zone ionization requires ideal
  // gas eos!");
  //  Don't try to track ionization for more species than we use.
  //  We will track ionization for the first saha_ncomps species
  athelas_requires(saha_ncomps == 3,
                   "One zone ionization requires [ionization.ncomps] = 3.");

  auto evolved = mesh_state(0).get_field("evolved");
  auto derived = mesh_state(0).get_field("derived");

  const int idx_tau = mesh_state(0).var_index("evolved", "specific_volume");
  const int idx_ener =
      mesh_state(0).var_index("evolved", "specific_total_fluid_energy");
  const int idx_density = mesh_state(0).var_index("derived", "density");
  const int idx_pressure = mesh_state(0).var_index("derived", "pressure");
  const int idx_tgas = mesh_state(0).var_index("derived", "gas_temperature");
  const int idx_sie =
      mesh_state(0).var_index("derived", "specific_internal_energy");
  auto sd0 = mesh_state(0);

  static const IndexRange ib(mesh->domain<Domain::Interior>());
  static const int nNodes = mesh->n_nodes();

  const auto temperature =
      pin->param()->get<double>("problem.params.temperature", 5800); // K
  const auto rho =
      pin->param()->get<double>("problem.params.rho", 1000.0); // g/cc

  const auto fn_ionization =
      pin->param()->get<std::string>("ionization.fn_ionization");
  const auto fn_deg =
      pin->param()->get<std::string>("ionization.fn_degeneracy");

  if (temperature <= 0.0 || rho <= 0.0) {
    throw_athelas_error("Temperature and denisty must be positive definite!");
  }

  std::shared_ptr<atom::CompositionData> comps =
      std::make_shared<atom::CompositionData>(mesh->n_elements() + 2, nNodes,
                                              ncomps);
  std::shared_ptr<atom::IonizationState> ionization_state =
      std::make_shared<atom::IonizationState>(
          mesh->n_elements() + 2, nNodes, ncomps, 7, saha_ncomps, fn_ionization,
          fn_deg, pin->param()->get<std::string>("ionization.solver"));
  auto mass_fractions = mesh_state.mass_fractions("evolved");
  auto charges = comps->charge();
  auto neutrons = comps->neutron_number();
  auto inv_atomic_mass = comps->inverse_atomic_mass();
  auto ionization_states = ionization_state->ionization_fractions();
  auto zbar = ionization_state->zbar();

  // Set up ball of hydrogen, helium, carbon.
  constexpr int i_H = 0;
  constexpr int i_He = 1;
  constexpr int i_C = 2;
  constexpr double X_H = 0.75;
  constexpr double X_He = 0.23;
  constexpr double X_C = 0.02;
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: OneZoneIonization :: nodal",
      DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
        for (int q = 0; q < nNodes; q++) {
          evolved(i, q, idx_tau) = 1.0 / rho;
          mass_fractions(i, q, i_H) = X_H;
          mass_fractions(i, q, i_He) = X_He;
          mass_fractions(i, q, i_C) = X_C;
        }
        for (int q = 0; q < nNodes + 2; q++) {
          derived(i, q, idx_density) = rho;
          derived(i, q, idx_tgas) = temperature;

          // Set Zbar assuming full ionization -- used as guess in Saha below.
          zbar(i, q, i_H) = 1;
          zbar(i, q, i_He) = 2;
          zbar(i, q, i_C) = 6;
        }

        charges(i_H) = 1;
        neutrons(i_H) = 0;
        inv_atomic_mass(i_H) = 1.0;

        charges(i_He) = 2;
        neutrons(i_He) = 2;
        inv_atomic_mass(i_He) = 1.0 / 4.0;

        charges(i_C) = 6;
        neutrons(i_C) = 6;
        inv_atomic_mass(i_C) = 1.0 / 12.0;
      });
  mesh_state.setup_composition(comps);
  mesh_state.setup_ionization(ionization_state);

  const auto &eos = mesh_state.eos();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: OneZoneIonization :: Energy",
      DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
        eos::EOSLambda lambda;
        for (int q = 0; q < nNodes + 2; q++) {
          derived(i, q, idx_density) = rho;
          derived(i, q, idx_tgas) = temperature;
          if (eos_type == "paczynski") {
            atom::paczynski_terms(sd0, i, q, lambda.ptr());
          }
          derived(i, q, idx_pressure) = pressure_from_density_temperature(
              eos, rho, temperature, lambda.ptr());
          derived(i, q, idx_sie) = sie_from_density_pressure(
              eos, rho, derived(i, q, idx_pressure), lambda.ptr());
        }
        for (int q = 0; q < nNodes; ++q) {
          evolved(i, q, idx_ener) = derived(i, q + 1, idx_sie);
        }
      });
}

} // namespace athelas::pgen::one_zone_ionization
