#pragma once

#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * Initialize ni_decay test
 **/
void ni_decay_init(MeshState &mesh_state, GridStructure *grid, ProblemIn *pin,
                   const eos::EOS *eos,
                   basis::ModalBasis *fluid_basis = nullptr) {
  const bool composition_active =
      pin->param()->get<bool>("physics.composition_enabled");
  const bool ni_decay_active =
      pin->param()->get<bool>("physics.heating.nickel.enabled");
  const auto ncomps = 3; // Ni, Co, Fe
  athelas_requires(composition_active,
                   "Ni decay requires composition enabled!");
  athelas_requires(ni_decay_active,
                   "Ni decay requires nickel heating enabled!");
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Ni decay requires ideal gas eos!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");
  auto uAF = mesh_state(0).get_field("u_af");

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

  if (temperature <= 0.0 || rho <= 0.0) {
    throw_athelas_error("Temperature and density must be positive definite!");
  }

  const double mu = 1.0 + constants::m_e / constants::m_p;
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;
  const double sie = constants::k_B * temperature / (gm1 * mu * constants::m_p);

  std::shared_ptr<atom::CompositionData> comps =
      std::make_shared<atom::CompositionData>(grid->n_elements() + 2, order,
                                              ncomps);
  auto mass_fractions = mesh_state.mass_fractions("u_cf");
  auto charges = comps->charge();
  auto neutrons = comps->neutron_number();
  auto ye = comps->ye();
  auto inv_atomic_mass = comps->inverse_atomic_mass();
  auto *species_indexer = comps->species_indexer();
  species_indexer->add("ni56", 3);
  species_indexer->add("co56", 4);
  species_indexer->add("fe56", 5);
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: NiDecay (1)", DevExecSpace(), ib.s,
      ib.e, KOKKOS_LAMBDA(const int i) {
        const int k = 0;

        uCF(i, k, q_Tau) = tau;
        uCF(i, k, q_V) = vel;
        uCF(i, k, q_E) = sie;

        for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
          uPF(i, iNodeX, iPF_D) = rho;
          uAF(i, iNodeX, 1) = temperature;
          ye(i, iNodeX) = 0.5;
        }

        // set up comps
        // For this problem we set up a contiguous list of species
        // form Z = 1 to ncomps. Mass fractions are uniform with no slopes.
        mass_fractions(i, k, 0) = 1.0; // Pure Ni

        // Ni
        charges(0) = 28;
        neutrons(0) = 28;
        inv_atomic_mass(0) = 1.0 / (56.0);
        // Co
        charges(1) = 27;
        neutrons(1) = 29;
        inv_atomic_mass(1) = 1.0 / (56.0);
        // Co
        charges(2) = 26;
        neutrons(2) = 30;
        inv_atomic_mass(2) = 1.0 / (56.0);
      });

  mesh_state.setup_composition(comps);

  // composition boundary condition
  static const IndexRange vb_comps(std::make_pair(3, 3 + ncomps - 1));
  bc::fill_ghost_zones_composition(uCF, vb_comps);

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: NiDecay (ghost)", DevExecSpace(), 0,
      ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          uPF(ib.s - 1 - i, iN, 0) = uPF(ib.s + i, (nNodes + 2) - iN - 1, 0);
          uPF(ib.s + 1 + i, iN, 0) = uPF(ib.s - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas
