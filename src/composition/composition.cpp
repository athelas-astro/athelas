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
  const auto number_density = comps->number_density();
  const auto ye = comps->ye();

  const auto *const ionization_states = state->ionization_state();
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
  lambda[7] = uaf(ix, node, vars::aux::Tgas);
}

} // namespace athelas::atom
