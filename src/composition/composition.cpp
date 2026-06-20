#include "composition/composition.hpp"
#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "eos/eos.hpp"
#include "geometry/mesh.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"
#include "utils/constants.hpp"

namespace athelas::atom {

using basis::NodalBasis;

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
void paczynski_terms(const StageData &stage_data, const int ix, const int node,
                     double *lambda) {
  const auto evolved = stage_data.get_field("evolved");
  const auto derived = stage_data.get_field("derived");
  const int idx_tgas = stage_data.var_index("derived", "gas_temperature");

  const auto *const comps = stage_data.comps();
  const auto number_density = comps->number_density();
  const auto ye = comps->ye();

  const auto *const ionization_states = stage_data.ionization_state();
  const auto ybar = ionization_states->ybar();
  const auto e_ion_corr = ionization_states->e_ion_corr();
  const auto sigma1 = ionization_states->sigma1();
  const auto sigma2 = ionization_states->sigma2();
  const auto sigma3 = ionization_states->sigma3();

  lambda[eos::paczynski_lambda::number_density] = number_density(ix, node);
  lambda[eos::paczynski_lambda::ye] = ye(ix, node);
  lambda[eos::paczynski_lambda::ybar] = ybar(ix, node);
  lambda[eos::paczynski_lambda::sigma1] = sigma1(ix, node);
  lambda[eos::paczynski_lambda::sigma2] = sigma2(ix, node);
  lambda[eos::paczynski_lambda::sigma3] = sigma3(ix, node);
  lambda[eos::paczynski_lambda::e_ion_corr] = e_ion_corr(ix, node);
  lambda[eos::EOS_LAMBDA_TEMPERATURE] = derived(ix, node, idx_tgas);
}

/**
 * @brief Fill derived composition quantities
 *
 * Currently, fills number densities and electron fraction.
 *
 * TODO(astrobarker): Explore hierarchical parallelism for inner loops.
 */
void fill_derived_comps(StageData &stage_data, const Mesh *const mesh) {
  const auto nnodes = mesh->n_nodes();
  const IndexRange ib(mesh->domain<Domain::Interior>());
  const IndexRange qb(nnodes + 2);

  const auto &basis = stage_data.basis();
  auto phi = basis.phi();

  auto *const comps = stage_data.comps();
  auto mass_fractions = stage_data.mass_fractions("evolved");
  auto mass_fractions_nodal = stage_data.get_field("composition");
  auto species = comps->charge();
  auto inv_atomic_mass = comps->inverse_atomic_mass();
  auto ye = comps->ye();
  auto abar = comps->abar();
  auto number_density = comps->number_density();
  const int num_species = static_cast<int>(comps->n_species());

  auto bulk = stage_data.get_field("bulk_composition");
  static constexpr int idx_x = 0;
  static constexpr int idx_y = 1;
  static constexpr int idx_z = 2;

  static constexpr double inv_m_p = 1.0 / constants::m_p;
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Composition :: fill derived", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_LAMBDA(const int i, const int q) {
        double ye_q = 0.0;
        double sum_y = 0.0;
        for (int e = 0; e < num_species; ++e) {
          const double Z = species(e);
          const double inv_A = inv_atomic_mass(e);
          const double xk = basis::basis_eval(phi, mass_fractions, i, e, q);
          const double xk_invA = xk * inv_A;
          ye_q += Z * xk_invA;
          sum_y += xk_invA;
          mass_fractions_nodal(i, q, e) = xk;

          if (Z == 1) {
            bulk(i, q, idx_x) = xk;
          }
          if (Z == 2) {
            bulk(i, q, idx_y) = xk;
          }
        }
        number_density(i, q) = sum_y * inv_m_p;
        ye(i, q) = ye_q;
        abar(i, q) = 1.0 / sum_y;
        bulk(i, q, idx_z) = 1.0 - (bulk(i, q, idx_x) + bulk(i, q, idx_y));
      });
}

} // namespace athelas::atom
