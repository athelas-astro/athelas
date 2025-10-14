#include <cmath>

#include "atom/atom.hpp"
#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "composition/compdata.hpp"
#include "composition/composition.hpp"
#include "composition/saha.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"
#include "state/state.hpp"

/**
 * @brief Functionality for saha ionization
 *
 * Word of warning: the code here is a gold medalist in index gymnastics.
 */

namespace athelas::atom {

using atom::IonLevel;
using basis::ModalBasis, basis::basis_eval;
using eos::EOS;

void solve_saha_ionization(State &state, const GridStructure &grid,
                           const EOS &eos, const ModalBasis &fluid_basis) {
  const auto uCF = state.u_cf();
  const auto uaf = state.u_af();
  const auto *const comps = state.comps();
  auto *const ionization_states = state.ionization_state();
  const auto *const atomic_data = ionization_states->atomic_data();
  const auto mass_fractions = comps->mass_fractions();
  const auto species = comps->charge();
  auto ionization_fractions = ionization_states->ionization_fractions();

  const auto phi = fluid_basis.phi();

  // pull out atomic data containers
  const auto ion_data = atomic_data->ion_data();
  const auto species_offsets = atomic_data->offsets();

  const auto &nNodes = grid.n_nodes();
  assert(ionization_fractions.extent(2) <=
         static_cast<size_t>(std::numeric_limits<int>::max()));
  const auto &ncomps = static_cast<int>(ionization_fractions.extent(2));

  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange nb(nNodes + 2);
  static const IndexRange eb(ncomps);
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Saha :: Solve ionization all", DevExecSpace(),
      ib.s, ib.e, nb.s, nb.e, eb.s, eb.e,
      KOKKOS_LAMBDA(const int i, const int q, const int e) {
        const double rho =
            1.0 / basis_eval(phi, uCF, i, vars::cons::SpecificVolume, q);
        const double temperature = uaf(i, q, 1);

        const double x_e = basis_eval(phi, mass_fractions, i, e, q);

        const int z = e + 1;
        const double nk = element_number_density(x_e, z, rho);

        // pull out element info
        const auto species_atomic_data =
            species_data(ion_data, species_offsets, e);
        auto ionization_fractions_e =
            Kokkos::subview(ionization_fractions, i, q, e, Kokkos::ALL);

        saha_solve(ionization_fractions_e, z, temperature, species_atomic_data,
                   nk);
      });
}

} // namespace athelas::atom
