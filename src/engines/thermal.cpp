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
using basis::NodalBasis;
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
                                           const StageData &stage_data,
                                           const GridStructure *grid,
                                           const int n_stages,
                                           const bool active)
    : active_(active), mend_idx_(1) {

  const int nx = pin->param()->get<int>("problem.nx");
  const int nq = grid->n_nodes();
  delta_ = AthelasArray4D<double>("thermal engine delta", n_stages, nx + 2,
                                  nq, 1);

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
  const int nnodes = pin->param()->get<int>("basis.nnodes");
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
    throw_athelas_error(
        "ThermalEngine :: mass extent index (mend_idx) is greater than nx!");
  }

  // Now we need to compute the actual deposition energy
  // If we specify an asymptotic explosion energy then offset by
  // the model's total energy
  auto mcell = grid->mass();
  auto menc = grid->enclosed_mass();
  if (mode_ == "direct") {
    energy_dep_ = energy_target_;
  } else {
    // integrate total energy on the mesh
    const bool gravity_active =
        pin->param()->get<bool>("physics.gravity_active");
    const int grav_active = gravity_active ? 1 : 0;
    const auto &basis = stage_data.fluid_basis();
    auto phi = basis.phi();
    auto ucf = stage_data.get_field("u_cf");
    auto r = grid->nodal_grid();
    auto weights = grid->weights();
    double total_energy = 0.0;
    athelas::par_reduce(
        DEFAULT_LOOP_PATTERN, "ThermalEngine :: Total energy", DevExecSpace(),
        1, nx, 0, nnodes - 1,
        KOKKOS_CLASS_LAMBDA(const int i, const int q, double &lenergy) {
          const double e_fluid =
              basis::basis_eval(phi, ucf, i, vars::cons::Energy, q + 1);
          const double e_grav =
              grav_active * constants::G_GRAV * menc(i, q) / r(i, q + 1);
          lenergy += (e_fluid - e_grav) * weights(q) * mcell(i) * FOURPI;
        },
        Kokkos::Sum<double>(total_energy));
    energy_dep_ = energy_target_ - total_energy;
    std::println("# --- Thermal Engine Parameters --- ");
    std::println("# Model energy   : {:.5e}", total_energy);
    std::println("# Target energy  : {:.5e}", energy_target_);
    std::println("# Engine energy  : {:.5e}", energy_dep_);
  }

  if (energy_dep_ < 0.0) {
    throw_athelas_error("The thermal engine energy has become < 0.0!");
  }

  // Below are the a, c, d coefficients for the injection profile
  c_coeff_ = std::log(RATIO_TIME_) / (tend_); // assuming t_start = 0.0
  d_coeff_ = c_coeff_ * energy_dep_ / (1.0 - std::exp(-c_coeff_ * tend_));
  a_coeff_ = std::log(RATIO_MASS_) / (mend_ - m_start);

  // integral for b_coeff_
  double b_int = 0.0;
  auto weights = grid->weights();
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "ThermalEngine :: b integral", DevExecSpace(),
      1, mend_idx_,
      KOKKOS_CLASS_LAMBDA(const int i, double &lb) {
        /*
          for (int q = 0; q < nnodes; ++q) {
          double dm = FOURPI * mcell(i);
            lb +=
                std::exp(-a_coeff_ * menc(i, q)) * dm;
                //std::exp(-a_coeff_ * menc(i, q)) * (menc(i + 1, q) - menc(i,
          q));
          }
          */
        lb += std::exp(-a_coeff_ * menc(i, 0)) * (menc(i + 1, 0) - menc(i, 0));
      },
      Kokkos::Sum<double>(b_int));
  b_int_ = b_int;
}

void ThermalEnginePackage::update_explicit(const StageData &stage_data,
                                           const GridStructure &grid,
                                           const TimeStepInfo &dt_info) {
  const auto time = dt_info.t;
  const auto &basis = stage_data.fluid_basis();
  static const auto &nnodes = grid.n_nodes();
  static const IndexRange qb(nnodes);
  static const IndexRange ib(grid.domain<Domain::Interior>());

  const auto stage = dt_info.stage;
  auto ucf = stage_data.get_field("u_cf");

  const IndexRange ib_dep(std::make_pair(1, mend_idx_));
  auto weights = grid.weights();
  auto dr = grid.widths();
  auto mass = grid.mass();
  auto menc = grid.enclosed_mass();
  auto phi = basis.phi();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "ThermalEngine :: Update", DevExecSpace(),
      ib_dep.s, ib_dep.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        const double b_coeff = d_coeff_ * std::exp(-c_coeff_ * time) / b_int_;
          delta_(stage, i, q, pkg_vars::Energy) =
              weights(q) * b_coeff *
              std::exp(-a_coeff_ * menc(i, q));
        delta_(stage, i, q, pkg_vars::Energy) *= mass(i);
      });

  // --- Divide update by mass matrix ---
  auto inv_mqq = basis.inv_mass_matrix();
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "ThermalEngine :: delta / M_qq", DevExecSpace(),
      ib_dep.s, ib_dep.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        const double &imm = inv_mqq(i, q);
        delta_(stage, i, q, pkg_vars::Energy) *= imm;
      });
}

/**
 * @brief Apply thermal engine package delta.
 */
void ThermalEnginePackage::apply_delta(AthelasArray3D<double> lhs,
                                       const TimeStepInfo &dt_info) const {
  static const int nq = static_cast<int>(lhs.extent(1));
  static const IndexRange ib(std::make_pair(1, mend_idx_));
  static const IndexRange qb(nq);

  const int stage = dt_info.stage;

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Thermal engine :: Apply delta", DevExecSpace(),
      ib.s, ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        lhs(i, q, vars::cons::Energy) +=
            dt_info.dt_coef * delta_(stage, i, q, pkg_vars::Energy);
      });
}

/**
 * @brief zero delta field
 */
void ThermalEnginePackage::zero_delta() const noexcept {
  static const IndexRange sb(static_cast<int>(delta_.extent(0)));
  static const IndexRange ib(static_cast<int>(delta_.extent(1)));
  static const IndexRange qb(static_cast<int>(delta_.extent(2)));
  static const IndexRange vb(static_cast<int>(delta_.extent(3)));

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Thermal engine :: Zero delta", DevExecSpace(),
      sb.s, sb.e, ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int s, const int i, const int q) {
        for (int v = vb.s; v <= vb.e; ++v) {
          delta_(s, i, q, v) = 0.0;
        }
      });
}

/**
 * @brief Thermal engine timestep restriction
 * @note We require that the engine is spread through 2500 timesteps at least.
 * The actual heating restriction is prohibitively expensive.
 **/
auto ThermalEnginePackage::min_timestep(const StageData & /*stage_data*/,
                                        const GridStructure & /*grid*/,
                                        const TimeStepInfo & /*dt_info*/) const
    -> double {
  return tend_ / 2500.0;
}

/**
 * @brief ThermalEngine fill derived
 * No-op.
 */
void ThermalEnginePackage::fill_derived(StageData &stage_data,
                                        const GridStructure &grid,
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
