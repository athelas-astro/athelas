#include "composition/saha.hpp"

#include <limits>

namespace athelas::atom {

template <eos::EOSInversion Inversion, SahaSolver SolverType>
void compute_temperature_with_saha(StageData &stage_data, const Mesh &mesh) {
  using root_finders::RegulaFalsiAlgorithm;
  const auto nnodes = mesh.n_nodes();
  const IndexRange ib(mesh.domain<Domain::Interior>());
  const IndexRange qb(nnodes + 2);

  auto evolved = stage_data.get_field("evolved");
  auto derived = stage_data.get_field("derived");
  const int idx_tau = stage_data.var_index("evolved", "specific_volume");
  const int idx_vel = stage_data.var_index("evolved", "velocity");
  const int idx_ener =
      stage_data.var_index("evolved", "specific_total_fluid_energy");
  const int idx_pressure = stage_data.var_index("derived", "pressure");
  const int idx_tgas = stage_data.var_index("derived", "gas_temperature");
  const auto &eos = stage_data.eos();
  const auto &basis = stage_data.basis();

  const auto *const comps = stage_data.comps();
  auto mass_fractions = stage_data.mass_fractions("evolved");
  auto mass_fractions_nodal = stage_data.get_field("composition");
  auto species = comps->charge();
  auto neutron_number = comps->neutron_number();
  auto inv_atomic_mass = comps->inverse_atomic_mass();
  auto ye = comps->ye();
  auto abar = comps->abar();
  auto n_e = comps->electron_number_density();
  auto number_density = comps->number_density();
  auto *species_indexer = comps->species_indexer();

  auto *const ionization_state = stage_data.ionization_state();
  auto ybar = ionization_state->ybar();
  auto ionization_fractions = ionization_state->ionization_fractions();
  auto ln_i = ionization_state->ln_i();
  auto zbars = ionization_state->zbar();
  auto e_ion_corr = ionization_state->e_ion_corr();
  auto sigma1 = ionization_state->sigma1();
  auto sigma2 = ionization_state->sigma2();
  auto sigma3 = ionization_state->sigma3();

  const auto ncomps_saha = ionization_state->ncomps();

  const bool has_neuts = species_indexer->contains("neut");
  const int start_elem = has_neuts ? 1 : 0;
  const int end_elem = has_neuts ? ncomps_saha : ncomps_saha - 1;
  const IndexRange eb_saha(std::make_pair(start_elem, end_elem));

  const IndexRange eb(std::make_pair(start_elem, comps->n_species() - 1));

  const auto *const atomic_data = ionization_state->atomic_data();
  auto ion_data = atomic_data->ion_data();
  auto species_offsets = atomic_data->offsets();
  auto prefix_sum_pots = atomic_data->sum_pots();

  // Bracketed solve of the root-form coupled residual
  // f(T) = model(rho, T, lambda(T)) - target. abs_tol applies to |f| in
  // model units (erg/g or dyn/cm^2), where no meaningful absolute scale
  // exists; disable it and converge on the relative change in T.
  static root_finders::RootFinder<double, RegulaFalsiAlgorithm<double>> solver(
      {.abs_tol = 0.0, .rel_tol = 1.0e-10, .max_iterations = 64});

  auto phi = basis.phi();
  const std::size_t nstates_total = ionization_fractions.extent(3);
  const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
  const size_t scratch_size = ScratchPad1D<double>::shmem_size(nstates_total);

  athelas::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "EOS :: T/Saha solve", DevExecSpace(),
      scratch_size, scratch_level, ib.s, ib.e, qb.s, qb.e,
      KOKKOS_LAMBDA(athelas::team_mbr_t member, const int i, const int q) {
        n_e(i, q) = 0.0;

        ScratchPad1D<double> saha_factors(member.team_scratch(scratch_level),
                                          nstates_total);

        const double rho = 1.0 / basis::basis_eval(phi, evolved, i, idx_tau, q);
        const double temperature_guess = derived(i, q, idx_tgas);

        double target_var = 0.0;
        if constexpr (Inversion == eos::EOSInversion::Pressure) {
          target_var = derived(i, q, idx_pressure);

        } else {
          const double vel = basis::basis_eval(phi, evolved, i, idx_vel, q);
          const double emt = basis::basis_eval(phi, evolved, i, idx_ener, q);
          target_var = emt - 0.5 * vel * vel;
        }

        const CoupledSolverContent content{evolved,
                                           derived,
                                           mass_fractions,
                                           mass_fractions_nodal,
                                           phi,
                                           zbars,
                                           ye,
                                           n_e,
                                           number_density,
                                           ybar,
                                           abar,
                                           e_ion_corr,
                                           sigma1,
                                           sigma2,
                                           sigma3,
                                           ionization_fractions,
                                           saha_factors,
                                           ln_i,
                                           species,
                                           neutron_number,
                                           inv_atomic_mass,
                                           species_offsets,
                                           ion_data,
                                           prefix_sum_pots,
                                           eb_saha,
                                           eb,
                                           i,
                                           q,
                                           target_var};

        // The coupled residual f(T) = model(rho, T, lambda(T)) - target is
        // monotone increasing, so its root on a sign-changing bracket is the
        // consistent temperature. A bracketed solve cannot be trapped by its
        // starting point: the previous fixed-point iteration was warm-started
        // from the last accepted temperature, so a cell once pinned at the
        // 500 K floor re-seeded itself every stage and could not climb back
        // through the stiff recombination range. A hot interior cell stuck
        // reading 500 K stops emitting, drains its radiation energy to the
        // floor one-way, and freezes the implicit transport solve.
        double res;
        const double f_floor =
            coupled_temperature_residual<Inversion, SolverType>(
                saha_min_temperature, rho, eos, content);
        if (!std::isfinite(f_floor)) {
          // Do not reinterpret a failed thermodynamic evaluation as a cold
          // state. Propagate the failure to the caller.
          res = std::numeric_limits<double>::quiet_NaN();
        } else if (!(f_floor < 0.0)) {
          // The target sits at or below the floor-temperature branch.
          res = saha_min_temperature;
        } else {
          constexpr double bracket_widen = 1.5;
          constexpr double bracket_expand = 8.0;
          double lo = saha_min_temperature;
          double hi = saha_max_temperature;
          bool have_hi = false;

          // Warm probes near the previous temperature tighten the bracket to
          // a few Saha evaluations in the common quasi-static case.
          const double t_probe =
              std::max(temperature_guess / bracket_widen, saha_min_temperature);
          if (t_probe > saha_min_temperature) {
            const double f_probe =
                coupled_temperature_residual<Inversion, SolverType>(
                    t_probe, rho, eos, content);
            if (f_probe < 0.0) {
              lo = t_probe;
            } else if (std::isfinite(f_probe)) {
              hi = t_probe;
              have_hi = true;
            }
          }
          if (!have_hi) {
            double t_hi =
                std::min(std::max(bracket_widen * temperature_guess, 1.0e4),
                         saha_max_temperature);
            double f_hi = coupled_temperature_residual<Inversion, SolverType>(
                t_hi, rho, eos, content);
            while (f_hi < 0.0 && t_hi < saha_max_temperature) {
              lo = t_hi;
              t_hi = std::min(bracket_expand * t_hi, saha_max_temperature);
              f_hi = coupled_temperature_residual<Inversion, SolverType>(
                  t_hi, rho, eos, content);
            }
            hi = t_hi;
          }

          res =
              solver.solve(coupled_temperature_residual<Inversion, SolverType>,
                           lo, hi, temperature_guess, rho, eos, content);
        }

        // Leave the stored ionization state and lambda consistent with the
        // returned temperature; the solver's last internal evaluation is
        // generally not at the root it returns.
        if (std::isfinite(res)) {
          static_cast<void>(coupled_temperature_residual<Inversion, SolverType>(
              res, rho, eos, content));
        }
        derived(i, q, idx_tgas) = res;
      });
}

template void
compute_temperature_with_saha<eos::EOSInversion::Sie, SahaSolver::Linear>(
    StageData &stage_data, const Mesh &mesh);
template void
compute_temperature_with_saha<eos::EOSInversion::Sie, SahaSolver::Log>(
    StageData &stage_data, const Mesh &mesh);
template void
compute_temperature_with_saha<eos::EOSInversion::Pressure, SahaSolver::Linear>(
    StageData &stage_data, const Mesh &mesh);
template void
compute_temperature_with_saha<eos::EOSInversion::Pressure, SahaSolver::Log>(
    StageData &stage_data, const Mesh &mesh);

} // namespace athelas::atom
