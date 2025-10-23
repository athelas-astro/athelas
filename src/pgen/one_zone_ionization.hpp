/**
 * @file one_zone_ionization.hpp
 * --------------
 *
 * @brief One zone ionization test
 */

#pragma once

#include "basis/polynomial_basis.hpp"
#include "composition/saha.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * Initialize one_zone_ionization test
 **/
void one_zone_ionization_init(State *state, GridStructure *grid, ProblemIn *pin,
                              const eos::EOS *eos,
                              basis::ModalBasis *fluid_basis = nullptr) {
  const bool ionization_active =
      pin->param()->get<bool>("physics.ionization_enabled");
  const int saha_ncomps =
      pin->param()->get<int>("ionization.ncomps"); // for ionization
  const auto ncomps =
      pin->param()->get<int>("composition.ncomps", 1); // mass fractions
  if (!ionization_active) {
    THROW_ATHELAS_ERROR("One zone ionization requires ionization enabled!");
  }
  if (pin->param()->get<std::string>("eos.type") != "ideal") {
    THROW_ATHELAS_ERROR("One zone ionization requires ideal gas eos!");
  }
  // Don't try to track ionization for more species than we use.
  // We will track ionization for the first saha_ncomps species
  if (saha_ncomps > ncomps) {
    THROW_ATHELAS_ERROR("One zone ionization requires [ionization.ncomps] >= "
                        "[problem.params.ncomps]!");
  }

  AthelasArray3D<double> uCF = state->u_cf();
  AthelasArray3D<double> uPF = state->u_pf();
  auto uAF = state->u_af();

  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const int nNodes = grid->n_nodes();
  static const int order = nNodes;

  const int q_Tau = 0;
  const int q_V = 1;
  const int q_E = 2;

  const int iPF_D = 0;

  const auto temperature =
      pin->param()->get<double>("problem.params.temperature", 5800); // K
  const auto rho =
      pin->param()->get<double>("problem.params.rho", 1000.0); // g/cc
  const double vel = 0.0;
  const double tau = 1.0 / rho;

  const auto fn_ionization =
      pin->param()->get<std::string>("ionization.fn_ionization");
  const auto fn_deg =
      pin->param()->get<std::string>("ionization.fn_degeneracy");

  if (temperature <= 0.0 || rho <= 0.0) {
    THROW_ATHELAS_ERROR("Temperature and denisty must be positive definite!");
  }

  const double mu = 1.0 + constants::m_e / constants::m_p;
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;
  const double sie = constants::k_B * temperature / (gm1 * mu * constants::m_p);

  std::shared_ptr<atom::CompositionData> comps =
      std::make_shared<atom::CompositionData>(
          grid->n_elements() + 2, order, ncomps,
          state->params()->get<int>("n_stages"));
  std::shared_ptr<atom::IonizationState> ionization_state =
      std::make_shared<atom::IonizationState>(grid->n_elements() + 2, nNodes,
                                              saha_ncomps, saha_ncomps + 1,
                                              fn_ionization, fn_deg);
  auto mass_fractions = state->mass_fractions();
  auto charges = comps->charge();
  auto neutrons = comps->neutron_number();
  auto ionization_states = ionization_state->ionization_fractions();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: OneZoneIonization (1)",
      DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
        const int k = 0;

        uCF(i, k, q_Tau) = tau;
        uCF(i, k, q_V) = vel;
        uCF(i, k, q_E) = sie;

        for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
          uPF(i, iNodeX, iPF_D) = rho;
          uAF(i, iNodeX, 1) = temperature;
        }

        // set up comps
        // For this problem we set up a contiguous list of species
        // form Z = 1 to ncomps. Mass fractions are uniform with no slopes.
        for (int elem = 0; elem < ncomps; ++elem) {
          mass_fractions(i, k, elem) = 1.0 / ncomps;
          charges(elem) = elem + 1;
          neutrons(elem) = elem + 1;
        }

        for (int node = 0; node < nNodes + 2; ++node) {

          // overkill
          if (ionization_active) {
            for (int elem = 0; elem < saha_ncomps; ++elem) {
              const int Z = charges(elem);

              for (int z = 0; z < Z + 1; ++z) {
                ionization_states(i, node, elem, z) = 0.0; // unnecessary
              }
            }
          }
        }
      });

  state->setup_composition(comps);
  state->setup_ionization(ionization_state);
  if (fluid_basis != nullptr) {
    atom::solve_saha_ionization(*state, *grid, *eos, *fluid_basis);
  }

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: OneZoneIonization (ghost)",
      DevExecSpace(), 0, ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          uPF(ib.s - 1 - i, iN, 0) = uPF(ib.s + i, (nNodes + 2) - iN - 1, 0);
          uPF(ib.s + 1 + i, iN, 0) = uPF(ib.s - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas
