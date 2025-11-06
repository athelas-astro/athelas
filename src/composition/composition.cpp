#include "composition/composition.hpp"
#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"
#include "utils/constants.hpp"

namespace athelas::atom {

using basis::ModalBasis;

/**
 * @brief Store the extra "lambda" terms for paczynski eos
 * NOTE:: Lambda contents:
 * 0: N (for ion pressure)
 * 1: ye
 * 2: ybar (mean ionization state)
 * 3: sigma1
 * 4: sigma2
 * 5: sigma3
 * 6: e_ioncorr (ionization corrcetion to internal energy)
 * 7: temperature_guess
 *
 * TODO(astrobarker): should inputs to this be subviews?
 * Should this exist?
 */
void paczynski_terms(const State *const state, const int ix, const int node,
                     double *const lambda) {
  const auto ucf = state->u_cf();
  const auto uaf = state->u_af();

  const auto *const comps = state->comps();
  const auto mass_fractions = state->mass_fractions();
  const auto species = comps->charge();
  const auto neutron_number = comps->neutron_number();
  const auto number_density = comps->number_density();
  const auto ye = comps->ye();

  const auto *const ionization_states = state->ionization_state();
  const auto ionization_fractions = ionization_states->ionization_fractions();
  const auto ybar = ionization_states->ybar();
  const auto e_ion_corr = ionization_states->e_ion_corr();
  const auto sigma1 = ionization_states->sigma1();
  const auto sigma2 = ionization_states->sigma2();
  const auto sigma3 = ionization_states->sigma3();

  lambda[0] = number_density(ix, node);
  lambda[1] = ye(ix, node);
  lambda[2] = ybar(ix, node);
  lambda[3] = sigma1(ix, node);
  lambda[4] = sigma2(ix, node);
  lambda[5] = sigma3(ix, node);
  lambda[6] = e_ion_corr(ix, node);
  lambda[7] = uaf(ix, node, vars::aux::Tgas); // temperature
}

// Compute electron number density
auto electron_density(const AthelasArray3D<double> mass_fractions,
                      const AthelasArray4D<double> ion_fractions,
                      const AthelasArray1D<int> charges, const int i,
                      const int q, const double rho) -> double {
  double n_e = 0.0;
  const size_t n_species = charges.size();

  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "Paczynski :: Reduce :: ne", DevExecSpace(), 0,
      n_species - 1,
      KOKKOS_LAMBDA(const int elem, double &ne_local) {
        const double n_elem = element_number_density(mass_fractions(i, q, elem),
                                                     charges(elem), rho);

        // Sum charge * ionization_fraction for each charge state
        const int max_charge = charges(elem);
        for (int charge = 1; charge <= max_charge; ++charge) {
          const double f_ion = ion_fractions(i, q, elem, charge);
          ne_local += charge * f_ion * n_elem;
        }
      },
      Kokkos::Sum<double>(n_e));
  return n_e;
}

} // namespace athelas::atom
