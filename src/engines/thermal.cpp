#include "engines/thermal.hpp"
#include "Kokkos_Core.hpp"
#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "compdata.hpp"
#include "constants.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"
#include "pgen/problem_in.hpp"
#include "utils/utilities.hpp"

namespace athelas::thermal_engine {
using atom::CompositionData;
using basis::ModalBasis;
using constants::FOURPI;
using utilities::to_lower;

/**
 * @brief ThermalEnginePackage constructor
 * It might be nice to allow to specify instead of a mass
 * extent optionally an index instead.
 *
 * It might be nice to pass in to the constructor t_start
 */
ThermalEnginePackage::ThermalEnginePackage(const ProblemIn *pin,
                                           const State *state,
                                           const GridStructure *grid,
                                           ModalBasis *basis, const bool active)
    : active_(active), basis_(basis), mend_idx_(1) {

  const int nx = pin->param()->get<int>("problem.nx");
  delta_ = AthelasArray3D<double>("nickel delta", nx + 2, basis->order(), 1);

  energy_target_ = pin->param()->get<double>("physics.engine.thermal.energy");
  mode_ =
      to_lower(pin->param()->get<std::string>("physics.engine.thermal.mode"));
  tend_ = pin->param()->get<double>("physics.engine.thermal.tend");
  mstart_ = pin->param()->get<int>("physics.engine.thermal.mstart");

  // I think we may want to divorce from assuming units comparable to Msun
  mend_ = pin->param()->get<double>("physics.engine.thermal.mend") *
          constants::M_sun;

  // Find index of mass spread
  auto mass_enc_h = Kokkos::create_mirror_view(grid->enclosed_mass());
  const double m_start = mass_enc_h(mstart_, 0);
  mend_ += m_start;
  const int nnodes = pin->param()->get<int>("fluid.nnodes");
  for (int i = 1; i <= nx; ++i) {
    for (int q = 0; q < nnodes; ++q) {
      if (mass_enc_h(i, q) <= mend_) {
        mend_idx_++;
      } else {
        break;
      }
    }
  }

  if (mend_idx_ > nx) {
    THROW_ATHELAS_ERROR(
        "ThermalEngine :: mass extent index (mend_idx) is greater than nx!");
  }

  // Now we need to compute the actual deposition energy
  // If we specify an asymptotic explosion energy then offset by
  // the model's total energy
  const auto mcell = grid->mass();
  const auto menc = grid->enclosed_mass();
  if (mode_ == "direct") {
    energy_dep_ = energy_target_;
  } else {
    // integrate total energy on the mesh
    const bool gravity_active =
        pin->param()->get<bool>("physics.gravity_active");
    const int grav_active = gravity_active ? 1 : 0;
    const auto phi = basis_->phi();
    const auto ucf = state->u_cf();
    const auto r = grid->nodal_grid();
    const auto weights = grid->weights();
    double total_energy = 0.0;
    athelas::par_reduce(
        DEFAULT_LOOP_PATTERN, "ThermalEngine :: Total energy", DevExecSpace(),
        1, nx, 0, nnodes - 1,
        KOKKOS_CLASS_LAMBDA(const int i, const int q, double &lenergy) {
          const double e_fluid =
              basis::basis_eval(phi, ucf, i, vars::cons::Energy, q + 1);
          const double e_grav =
              grav_active * constants::G_GRAV * menc(i, q) / r(i, q);
          lenergy += (e_fluid - e_grav) * weights(q) * mcell(i) * FOURPI;
        },
        Kokkos::Sum<double>(total_energy));
    energy_dep_ = energy_target_ - total_energy;
    std::println("etarget emodel edep {:.5e} {:.5e} {:.5e}", energy_target_,
                 total_energy, energy_dep_);
  }

  if (energy_dep_ < 0.0) {
    THROW_ATHELAS_ERROR("The thermal engine energy has become < 0.0!");
  }

  // Below are the a, c, d coefficients for the injection profile
  c_coeff_ = std::log(RATIO_TIME_) / (tend_); // assuming t_start = 0.0
  d_coeff_ = c_coeff_ * energy_dep_ / (1.0 - std::exp(-c_coeff_ * tend_));
  a_coeff_ = std::log(RATIO_MASS_) / (mend_ - m_start);

  // integral for b_coeff_
  double b_int = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "ThermalEngine :: b integral", DevExecSpace(),
      1, mend_idx_,
      KOKKOS_CLASS_LAMBDA(const int i, double &lb) {
        for (int q = 0; q < nnodes; ++q) {
          lb +=
              std::exp(-a_coeff_ * menc(i, q)) * (menc(i + 1, 0) - menc(i, 0));
        }
      },
      Kokkos::Sum<double>(b_int));
  b_int_ = b_int;
}

void ThermalEnginePackage::update_explicit(const State *const state,
                                           const GridStructure &grid,
                                           const TimeStepInfo &dt_info) {
  const int &order = basis_->order();
  static const auto &nnodes = grid.n_nodes();
  static const IndexRange qb(nnodes);
  static const IndexRange kb(order);
  static const IndexRange ib(grid.domain<Domain::Interior>());
  const auto u_stages = state->u_cf_stages();

  const auto stage = dt_info.stage;
  const auto ucf =
      Kokkos::subview(u_stages, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  // --- Zero out delta  ---
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "ThermalEngine :: Zero delta", DevExecSpace(),
      ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
        for (int k = kb.s; k <= kb.e; ++k) {
          delta_(i, k, pkg_vars::Energy) = 0.0;
        }
      });

  const IndexRange ib_dep(std::make_pair(1, mend_idx_));
  const auto weights = grid.weights();
  const auto dr = grid.widths();
  const auto mass = grid.mass();
  const auto menc = grid.enclosed_mass();
  const auto phi = basis_->phi();
  const auto time = dt_info.t;
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "ThermalEngine :: Update", DevExecSpace(),
      ib_dep.s, ib_dep.e, kb.s, kb.e,
      KOKKOS_CLASS_LAMBDA(const int i, const int k) {
        const double b_coeff = d_coeff_ * std::exp(-c_coeff_ * time) / b_int_;
        for (int q = qb.s; q <= qb.e; ++q) {
          delta_(i, k, pkg_vars::Energy) += weights(q) * phi(i, q + 1, k) *
                                            b_coeff *
                                            std::exp(-a_coeff_ * menc(i, q));
        }
        delta_(i, k, pkg_vars::Energy) *= mass(i);
      });

  // --- Divide update by mass matrix ---
  const auto inv_mkk = basis_->inv_mass_matrix();
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "ThermalEngine :: delta / M_kk", DevExecSpace(),
      ib_dep.s, ib_dep.e, kb.s, kb.e,
      KOKKOS_CLASS_LAMBDA(const int i, const int k) {
        const double &imm = inv_mkk(i, k);
        delta_(i, k, pkg_vars::Energy) *= imm;
      });
}

/**
 * @brief apply thermal engine package delta
 */
void ThermalEnginePackage::apply_delta(AthelasArray3D<double> lhs,
                                       const TimeStepInfo &dt_info) const {
  static const int nk = static_cast<int>(lhs.extent(1));
  static const IndexRange ib(std::make_pair(1, mend_idx_));
  static const IndexRange kb(nk);

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Thermal engine :: Apply delta", DevExecSpace(),
      ib.s, ib.e, kb.s, kb.e, KOKKOS_CLASS_LAMBDA(const int i, const int k) {
        lhs(i, k, vars::cons::Energy) +=
            dt_info.dt_coef * delta_(i, k, pkg_vars::Energy);
      });
}

/**
 * @brief Thermal engine timestep restriction
 * @note We require that the engine is spread through 1000 timesteps at least.
 **/
auto ThermalEnginePackage::min_timestep(const State *const /*state*/,
                                        const GridStructure & /*grid*/,
                                        const TimeStepInfo & /*dt_info*/) const
    -> double {
  static const double MAX_DT = tend_ / 1000.0;
  static const double dt_out = MAX_DT;
  return dt_out;
}

/**
 * @brief ThermalEngine fill derived
 */
void ThermalEnginePackage::fill_derived(State *state, const GridStructure &grid,
                                        const TimeStepInfo &dt_info) const {}

[[nodiscard]] KOKKOS_FUNCTION auto ThermalEnginePackage::name() const noexcept
    -> std::string_view {
  return "ThermalEngine";
}

[[nodiscard]] KOKKOS_FUNCTION auto
ThermalEnginePackage::is_active() const noexcept -> bool {
  return active_;
}

KOKKOS_FUNCTION
void ThermalEnginePackage::set_active(const bool active) { active_ = active; }

} // namespace athelas::thermal_engine
