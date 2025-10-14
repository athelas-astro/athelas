#include "heating/nickel_package.hpp"
#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "compdata.hpp"
#include "constants.hpp"
#include "geometry/grid.hpp"
#include "geometry/grid_indexer.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"
#include "pgen/problem_in.hpp"
#include "utils/utilities.hpp"

namespace athelas::nickel {
using atom::CompositionData;
using basis::ModalBasis;
using utilities::to_lower;

NickelHeatingPackage::NickelHeatingPackage(const ProblemIn *pin,
                                           ModalBasis *basis, const bool active)
    : active_(active), basis_(basis) {
  // set up heating deposition model
  const auto model_str =
      to_lower(pin->param()->get<std::string>("heating.nickel.model"));
  model_ = parse_model(model_str);

  const int nx = pin->param()->get<int>("problem.nx");
  const int nnodes = pin->param()->get<int>("fluid.nnodes");
  tau_gamma_ = AthelasArray3D<double>("tau_gamma", nx + 2, nnodes,
                                      8); // TODO(astrobarker): make runtime
  int_etau_domega_ =
      AthelasArray2D<double>("int_etau_domega", nx + 2,
                             nnodes); // integration of e^-tau dOmega
}

void NickelHeatingPackage::update_explicit(const State *const state,
                                           AthelasArray3D<double> dU,
                                           const GridStructure &grid,
                                           const TimeStepInfo &dt_info) {
  static const IndexRange ib(grid.domain<Domain::Interior>());
  const auto u_stages = state->u_cf_stages();

  const auto stage = dt_info.stage;
  const auto ucf =
      Kokkos::subview(u_stages, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  auto *comps = state->comps();
  const auto *const species_indexer = comps->species_indexer();
  // index gymnastics. dU holds updates for all quantities including
  // compositional. ind_offset gets us beyond radhydro species.
  static const auto ind_ni = species_indexer->get<int>("ni56");
  static const auto ind_co = species_indexer->get<int>("co56");
  static const auto ind_fe = species_indexer->get<int>("fe56");
  static const auto ind_offset = ucf.extent(2);

  // --- Zero out dU  ---
  // TODO(astrobarker): perhaps care to be taken here once mixing is in.
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "NickelHeating :: Zero delta", DevExecSpace(),
      ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
        dU(i, 0, ind_offset + ind_ni) = 0.0;
        dU(i, 0, ind_offset + ind_co) = 0.0;
        dU(i, 0, ind_offset + ind_fe) = 0.0;
      });

  if (model_ == NiHeatingModel::Swartz) [[likely]] {
    ni_update<NiHeatingModel::Swartz>(ucf, comps, dU, grid, dt_info);
  } else if (model_ == NiHeatingModel::Jeffery) {
    ni_update<NiHeatingModel::Jeffery>(ucf, comps, dU, grid, dt_info);
  } else {
    ni_update<NiHeatingModel::FullTrapping>(ucf, comps, dU, grid, dt_info);
  }
}

template <NiHeatingModel Model>
void NickelHeatingPackage::ni_update(const AthelasArray3D<double> ucf,
                                     CompositionData *comps,
                                     AthelasArray3D<double> dU,
                                     const GridStructure &grid,
                                     const TimeStepInfo &dt_info) const {
  const int &nNodes = grid.n_nodes();
  const int &order = basis_->order();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange kb(order);

  const auto mass_fractions_stages = comps->mass_fractions_stages();
  const auto mass_fractions =
      Kokkos::subview(mass_fractions_stages, dt_info.stage, Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL);
  const auto *const species_indexer = comps->species_indexer();

  // index gymnastics. dU holds updates for all quantities including
  // compositional. ind_offset gets us beyond radhydro species.
  static const auto ind_ni = species_indexer->get<int>("ni56");
  static const auto ind_co = species_indexer->get<int>("co56");
  static const auto ind_fe = species_indexer->get<int>("fe56");

  const auto mass = grid.mass();
  const auto weights = grid.weights();
  const auto phi = basis_->phi();
  const auto inv_mkk = basis_->inv_mass_matrix();

  // NOTE: This source term uses a mass integral instead of a volumetric one.
  // It's just simpler and natural here.
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "NickelHeating :: Update", DevExecSpace(), ib.s,
      ib.e, kb.s, kb.e, KOKKOS_CLASS_LAMBDA(const int i, const int k) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          const double f_dep =
              this->template deposition_function<Model>(ucf, comps, phi, i, q);
          local_sum += f_dep * weights(q) * phi(i, q + 1, k);
        }

        const double dx_o_mkk = mass(i) * inv_mkk(i, k);
        dU(i, k, 2) += local_sum * dx_o_mkk;
      });

  // TODO(astrobarker): Should this be an option?
  // NOTE: Nickel decay chain only affects cell averages.
  // Realistically I don't need to integrate X_Fe, but oh well.
  static const auto ind_offset = ucf.extent(2);
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "NickelHeating :: Decay network",
      DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
        const double x_ni = mass_fractions(i, vars::modes::CellAverage, ind_ni);
        const double x_co = mass_fractions(i, vars::modes::CellAverage, ind_co);
        const double rhs_ni = -LAMBDA_NI_ * x_ni;
        const double rhs_co = LAMBDA_NI_ * x_ni - LAMBDA_CO_ * x_co;
        const double rhs_fe = LAMBDA_CO_ * x_co;

        // Decay only alters cell average mass fractions!
        dU(i, vars::modes::CellAverage, ind_offset + ind_ni) += rhs_ni;
        dU(i, vars::modes::CellAverage, ind_offset + ind_co) += rhs_co;
        dU(i, vars::modes::CellAverage, ind_offset + ind_fe) += rhs_fe;
      });
}

/**
 * @brief Nickel 56 heating timestep restriction
 * @note We simply require the timestep to be smaller than the 56Ni mean
 * lifetime / 10. I doubt that this will ever be needed.
 **/
auto NickelHeatingPackage::min_timestep(const State *const /*state*/,
                                        const GridStructure & /*grid*/,
                                        const TimeStepInfo & /*dt_info*/) const
    -> double {
  static constexpr double MAX_DT = TAU_NI_ / 10.0;
  static constexpr double dt_out = MAX_DT;
  return dt_out;
}

void NickelHeatingPackage::fill_derived(State *state, const GridStructure &grid,
                                        const TimeStepInfo &dt_info) const {

  if (model_ != NiHeatingModel::Jeffery) {
    return;
  }
  using utilities::find_closest_cell;
  using utilities::LINTERP;
  // TODO(astrobarker): possibly compute r_min_ni here.
  // fill dtau_gamma, tau_gamma
  // I think we assume that tau = 0 at the outer interface, but
  // don't include that point on the array, so start from
  // outermost quadrature point
  const int stage = dt_info.stage;

  auto u_s = state->u_cf_stages();

  auto uCF = Kokkos::subview(u_s, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  // hacky
  if (stage == -1) {
    uCF = state->u_cf();
  }
  auto uPF = state->u_pf();
  auto uAF = state->u_af();

  const auto ye = state->comps()->ye();

  const int nnodes = grid.n_nodes();
  const int nx = grid.n_elements();
  static const RadialGridIndexer grid_indexer(nx, nnodes);
  const auto coords = grid.nodal_grid();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange nb(nnodes);

  const int nangles = tau_gamma_.extent(2); // TODO(astrobarker): make runtime!
  const int nr = 8; // TODO(astrobarker): make runtime!
  const double inv_nr = 1.0 / nr;
  const double th_max =
      constants::PI; // Perhaps make this not go into the excised region
  const double th_min = th_max / 4.0;
  const double dtheta = (th_max - th_min) / (nangles);
  const double r_outer = grid.get_x_r();
  const double r_outer2 = r_outer * r_outer;
  const auto centers = grid.centers();
  const std::size_t scratch_size = 0;
  const int scratch_level = 1;
  athelas::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN,
      "NickelHeating :: FillDerived :: OpticalDepth", DevExecSpace(),
      scratch_size, scratch_level, ib.s, ib.e, nb.s, nb.e,
      KOKKOS_CLASS_LAMBDA(athelas::team_mbr_t member, const int i,
                          const int q) {
        const double ri = coords(i, q);
        const double ri2 = ri * ri;

        // TODO(astrobarker) Use team shared memory for tau values
        auto *const taugamma = &tau_gamma_(i, q, 0);

        // Inner parallel loop over angles (thread level parallelism)
        athelas::par_for_inner(
            DEFAULT_INNER_LOOP_PATTERN, member, 0, nangles,
            [&](const int iangle) {
              // Angle-specific calculations here
              const double cos_theta = std::cos(th_min + iangle * dtheta);
              const double two_ri_cos = 2.0 * ri * cos_theta;
              const double rmax =
                  std::sqrt(r_outer2 + ri2 - two_ri_cos * r_outer);
              const double dr = rmax * inv_nr;

              // Compute optical depth for this specific (ix, node, iangle)
              double optical_depth = 0.0;
              for (int i = 0; i < nr + 1; ++i) {
                const double rx = i * dr;
                const double rj = std::sqrt(ri2 + rx * rx + two_ri_cos * rx);
                const int index = utilities::find_closest_cell(centers, rj, nx);
                const double rho_interp =
                    LINTERP(centers(index), centers(index + 1),
                            1.0 / uCF(index, vars::modes::CellAverage,
                                      vars::cons::SpecificVolume),
                            1.0 / uCF(index + 1, vars::modes::CellAverage,
                                      vars::modes::Slope),
                            rx);
                const double ye_interp =
                    LINTERP(centers(index), centers(index + 1), ye(index, 0),
                            ye(index + 1, nnodes + 1), rx);
                optical_depth += dtau(rho_interp, kappa_gamma(ye_interp), dr);
              }

              taugamma[iangle] = optical_depth;
            });

        member.team_barrier();

        double angle_integrated_tau = 0.0;
        athelas::par_reduce_inner(
            inner_loop_pattern_ttr_tag, member, 0, nangles,
            [=](const int iangle, double &local_sum) {
              local_sum += std::exp(taugamma[iangle]) *
                           std::sin(th_min + iangle * dtheta) * dtheta;
            },
            Kokkos::Sum<double>(angle_integrated_tau));
        int_etau_domega_(i, q) = angle_integrated_tau;
      });
}

[[nodiscard]] KOKKOS_FUNCTION auto NickelHeatingPackage::name() const noexcept
    -> std::string_view {
  return "NickelHeating";
}

[[nodiscard]] KOKKOS_FUNCTION auto
NickelHeatingPackage::is_active() const noexcept -> bool {
  return active_;
}

KOKKOS_FUNCTION
void NickelHeatingPackage::set_active(const bool active) { active_ = active; }

} // namespace athelas::nickel
