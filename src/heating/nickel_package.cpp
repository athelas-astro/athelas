#include "heating/nickel_package.hpp"
#include "basic_types.hpp"
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
using basis::NodalBasis;
using utilities::to_lower;

NickelHeatingPackage::NickelHeatingPackage(const ProblemIn *pin,
                                           const Params *indexer,
                                           const int n_stages, const int nq,
                                           const bool active)
    : active_(active) {
  // set up heating deposition model
  const auto model_str =
      to_lower(pin->param()->get<std::string>("heating.nickel.model"));
  model_ = parse_model(model_str);

  const int nx = pin->param()->get<int>("problem.nx");
  tau_gamma_ = AthelasArray3D<double>("tau_gamma", nx + 2, nq,
                                      8); // TODO(astrobarker): make runtime
  int_etau_domega_ = AthelasArray2D<double>("int_etau_domega", nx + 2,
                                            nq); // integration of e^-tau dOmega
  delta_ = AthelasArray4D<double>("nickel delta", n_stages, nx + 2, nq, 4);

  ind_ni_ = indexer->get<int>("ni56");
  ind_co_ = indexer->get<int>("co56");
  ind_fe_ = indexer->get<int>("fe56");
}

void NickelHeatingPackage::update_explicit(const StageData &stage_data,
                                           const GridStructure &grid,
                                           const TimeStepInfo &dt_info) {
  auto *comps = stage_data.comps();

  if (model_ == NiHeatingModel::Jeffery) {
    ni_update<NiHeatingModel::Jeffery>(stage_data, comps, grid, dt_info);
  } else {
    ni_update<NiHeatingModel::FullTrapping>(stage_data, comps, grid, dt_info);
  }
}

/**
 * @brief Nickel heating update.
 * Computes updates for heating and evolves the decay network.
 */
template <NiHeatingModel Model>
void NickelHeatingPackage::ni_update(const StageData &stage_data,
                                     CompositionData *comps,
                                     const GridStructure &grid,
                                     const TimeStepInfo &dt_info) const {
  static const int nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange qb(nNodes);

  const int stage = dt_info.stage;

  auto ucf = stage_data.get_field("u_cf");

  auto mass_fractions = stage_data.mass_fractions("u_cf");
  const auto *const species_indexer = comps->species_indexer();
  static const auto ind_ni = species_indexer->get<int>("ni56");
  static const auto ind_co = species_indexer->get<int>("co56");

  auto dm = grid.mass();
  auto weights = grid.weights();
  const auto &basis = stage_data.fluid_basis();
  auto inv_mqq = basis.inv_mass_matrix();
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "NickelHeating :: Update", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        const double x_ni = ucf(i, q, ind_ni);
        const double x_co = ucf(i, q, ind_co);
        const double f_dep = this->template deposition_function<Model>(i, q);
        const double source = ni_source(x_ni, x_co, f_dep);
        const double norm = weights(q) * dm(i) * inv_mqq(i, q);

        delta_(stage, i, q, pkg_vars::Energy) = f_dep * source * norm;
      });

  // Realistically I don't need to integrate X_Fe, but oh well.
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "NickelHeating :: Decay network", DevExecSpace(),
      ib.s, ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        const double x_ni = ucf(i, q, ind_ni);
        const double x_co = ucf(i, q, ind_co);
        const double rhs_ni = -LAMBDA_NI_ * x_ni;
        const double rhs_co = LAMBDA_NI_ * x_ni - LAMBDA_CO_ * x_co;
        const double rhs_fe = LAMBDA_CO_ * x_co;

        delta_(stage, i, q, pkg_vars::Nickel) = rhs_ni;
        delta_(stage, i, q, pkg_vars::Cobalt) = rhs_co;
        delta_(stage, i, q, pkg_vars::Iron) = rhs_fe;
      });
}

/**
 * @brief Apply nickel package delta.
 */
void NickelHeatingPackage::apply_delta(AthelasArray3D<double> lhs,
                                       const TimeStepInfo &dt_info) const {
  static const int nx = static_cast<int>(lhs.extent(0));
  static const int nq = static_cast<int>(lhs.extent(1));
  static const IndexRange ib(std::make_pair(1, nx - 2));
  static const IndexRange qb(nq);

  const int stage = dt_info.stage;

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Nickel :: Apply delta", DevExecSpace(), ib.s, ib.e,
      qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        lhs(i, q, vars::cons::Energy) +=
            dt_info.dt_coef * delta_(stage, i, q, pkg_vars::Energy);
        lhs(i, q, ind_ni_) +=
            dt_info.dt_coef * delta_(stage, i, q, pkg_vars::Nickel);
        lhs(i, q, ind_co_) +=
            dt_info.dt_coef * delta_(stage, i, q, pkg_vars::Cobalt);
        lhs(i, q, ind_fe_) +=
            dt_info.dt_coef * delta_(stage, i, q, pkg_vars::Iron);
      });
}

/**
 * @brief zero delta field
 */
void NickelHeatingPackage::zero_delta() const noexcept {
  static const IndexRange sb(static_cast<int>(delta_.extent(0)));
  static const IndexRange ib(static_cast<int>(delta_.extent(1)));
  static const IndexRange qb(static_cast<int>(delta_.extent(2)));
  static const IndexRange vb(static_cast<int>(delta_.extent(3)));

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Nickel :: Zero delta", DevExecSpace(), sb.s, sb.e,
      ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int s, const int i, const int q) {
        for (int v = vb.s; v <= vb.e; ++v) {
          delta_(s, i, q, v) = 0.0;
        }
      });
}

/**
 * @brief Nickel 56 heating timestep restriction
 * @note We simply require the timestep to be smaller than the 56Ni mean
 * lifetime / 10. I doubt that this will ever be needed.
 **/
auto NickelHeatingPackage::min_timestep(const StageData & /*stage_data*/,
                                        const GridStructure & /*grid*/,
                                        const TimeStepInfo & /*dt_info*/) const
    -> double {
  static constexpr double MAX_DT = TAU_NI_ / 10.0;
  static constexpr double dt_out = MAX_DT;
  return dt_out;
}

void NickelHeatingPackage::fill_derived(StageData &stage_data,
                                        const GridStructure &grid,
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

  auto ucf = stage_data.get_field("u_cf");
  // hacky
  // if (stage == -1) {
  //  ucf = stage_data.get_field("u_cf");
  //}
  auto uPF = stage_data.get_field("u_pf");
  auto uAF = stage_data.get_field("u_af");

  const auto ye = stage_data.comps()->ye();

  const int nnodes = grid.n_nodes();
  const int nx = grid.n_elements();
  static const RadialGridIndexer grid_indexer(nx, nnodes);
  auto coords = grid.nodal_grid();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange qb(nnodes);

  const int nangles = tau_gamma_.extent(2); // TODO(astrobarker): make runtime!
  const int nr = 8; // TODO(astrobarker): make runtime!
  const double inv_nr = 1.0 / nr;
  const double th_max =
      constants::PI; // Perhaps make this not go into the excised region
  const double th_min = th_max / 4.0;
  const double dtheta = (th_max - th_min) / (nangles);
  const double r_outer = grid.get_x_r();
  const double r_outer2 = r_outer * r_outer;
  auto centers = grid.centers();

  const std::size_t scratch_size = 0;
  const int scratch_level = 1;
  athelas::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN,
      "NickelHeating :: FillDerived :: OpticalDepth", DevExecSpace(),
      scratch_size, scratch_level, ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(athelas::team_mbr_t member, const int i,
                          const int q) {
        const double ri = coords(i, q + 1);
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
              for (int l = 0; l < nr; ++l) {
                const double rx = l * dr;
                const double rj = std::sqrt(ri2 + rx * rx + two_ri_cos * rx);
                const int index = utilities::find_closest_cell(centers, rj, nx);
                const double rho_interp = LINTERP(
                    centers(index), centers(index + 1),
                    1.0 / ucf(index, q, vars::cons::SpecificVolume),
                    1.0 / ucf(index + 1, q, vars::cons::SpecificVolume), rj);

                const double ye_interp =
                    LINTERP(centers(index), centers(index + 1), ye(index, 0),
                            ye(index + 1, nnodes + 1), rj);
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
