#include "timestepper/timestepper.hpp"

#include "geometry/mesh.hpp"
#include "interface/packages_base.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"
#include "limiters/bound_enforcing_limiter.hpp"
#include "limiters/slope_limiter.hpp"
#include "loop_layout.hpp"
#include "problem_in.hpp"
#include "timestepper/tableau.hpp"

namespace athelas {

/**
 * The constructor creates the necessary data structures for time evolution.
 * Lots of structures used in discretizations live here.
 **/
TimeStepper::TimeStepper(const ProblemIn *pin)
    : nvars_evolved_(nvars_evolved(pin)),
      mSize_(pin->param()->get<int>("problem.nx") + 2),
      integrator_(
          create_tableau(pin->param()->get<MethodID>("time.integrator"))),
      nStages_(integrator_.num_stages), tOrder_(integrator_.explicit_order),
      SumVar_U_("SumVar_U", mSize_, pin->param()->get<int>("basis.nnodes"),
                nvars_evolved_),
      x_l_sumvar_("x_l_sumvar_", nStages_, mSize_ + 1) {}

void TimeStepper::reset_stage_sumvar(const Mesh &mesh, int stage,
                                     AthelasArray3D<double> u0,
                                     const IndexRange &ib, const IndexRange &qb,
                                     const IndexRange &vb, const char *label) {
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, label, DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        for (int v = vb.s; v <= vb.e; ++v) {
          SumVar_U_(i, q, v) = u0(i, q, v);
        }
      });

  auto left_interface = mesh.x_l();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Timestepper :: Reset mesh sumvar",
      DevExecSpace(), 0, mSize_, KOKKOS_CLASS_LAMBDA(const int i) {
        x_l_sumvar_(stage, i) = left_interface(i);
      });
}

void TimeStepper::accumulate_grid_motion(MeshState &mesh_state, int sum_stage,
                                         int data_stage, double dt_coef,
                                         const IndexRange &ib,
                                         const char *label) {
  auto interface =
      mesh_state(data_stage).get_field<AthelasArray2D<double>>("interface");
  const int idx_vstar =
      mesh_state(data_stage).var_index("interface", "interface_velocity");
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, label, DevExecSpace(), ib.s, ib.e,
      KOKKOS_CLASS_LAMBDA(const int i) {
        x_l_sumvar_(sum_stage, i) += dt_coef * interface(i, idx_vstar);
      });
}

void TimeStepper::update_stage_mesh(MeshState &mesh_state, int stage) {
  // Stage 0 is the canonical mesh (x_l^n), read directly via StageData::mesh();
  // there is nothing to materialize. Later stages are rebuilt into the single
  // work buffer from the canonical mesh plus this stage's accumulated interface
  // positions. A stage mesh is fully determined by (canonical mesh, x_l for the
  // stage), so one buffer suffices.
  if (stage == 0) {
    return;
  }
  auto x_l_sumvar = Kokkos::subview(x_l_sumvar_, stage, Kokkos::ALL);
  auto &mesh_stage = mesh_state.mesh_stage();
  mesh_stage.copy_from(mesh_state.mesh());
  mesh_stage.update_grid(x_l_sumvar);
}

void TimeStepper::step(PackageManager *pkgs, MeshState &mesh_state,
                       TimeStepInfo &dt_info, SlopeLimiter *sl_hydro) {
  // hydro explicit update
  update_fluid_explicit(pkgs, mesh_state, dt_info, sl_hydro);
}

void TimeStepper::update_fluid_explicit(PackageManager *pkgs,
                                        MeshState &mesh_state,
                                        TimeStepInfo &dt_info,
                                        SlopeLimiter *sl_hydro) {
  auto &mesh = mesh_state.mesh();
  const int nvars = mesh_state.nvars("evolved");
  const IndexRange ib(mesh.domain<Domain::Entire>());
  const IndexRange qb(mesh.n_nodes());
  const IndexRange vb(nvars);

  const double t = dt_info.t;
  const double dt = dt_info.dt;

  const auto &fluid_basis = mesh_state.fluid_basis();
  const auto &eos = mesh_state.eos();

  auto u0 = mesh_state(0).get_field("evolved");
  for (int iS = 0; iS < nStages_; ++iS) {
    dt_info.stage = iS;

    // re-set the summation variables `SumVar`
    auto stage_data = mesh_state.stage(iS);
    auto u = stage_data.get_field("evolved");
    reset_stage_sumvar(mesh, iS, u0, ib, qb, vb,
                       "Timestepper :: EX :: Reset sumvar");

    // --- Inner update loop ---

    for (int j = 0; j < iS; ++j) {
      dt_info.stage = j;
      dt_info.t = t + integrator_.explicit_tableau.c_i(j) * dt;
      const double dt_a_ex = dt * integrator_.explicit_tableau.a_ij(iS, j);
      dt_info.dt_coef = dt_a_ex;

      pkgs->apply_delta(SumVar_U_, dt_info);

      accumulate_grid_motion(mesh_state, iS, j, dt_a_ex, ib,
                             "Timestepper :: EX :: grid");
    } // End inner loop

    // set U_s
    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "Timestepper :: EX :: Set Us", DevExecSpace(),
        ib.s, ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
          for (int v = vb.s; v <= vb.e; ++v) {
            u(i, q, v) = SumVar_U_(i, q, v);
          }
        });

    update_stage_mesh(mesh_state, iS);

    // stage_data.mesh() resolves to this stage's mesh (canonical for stage 0,
    // the work buffer otherwise).
    apply_slope_limiter(sl_hydro, u, stage_data, fluid_basis, eos);
    bel::apply_bound_enforcing_limiter(stage_data);

    dt_info.stage = iS;
    dt_info.t = t + integrator_.explicit_tableau.c_i(iS) * dt;
    pkgs->fill_derived(stage_data, dt_info);
    pkgs->update_explicit(stage_data, dt_info);
  } // end outer loop

  // --- Final U^n update ---

  for (int iS = 0; iS < nStages_; ++iS) {
    dt_info.stage = iS;
    dt_info.t = t + integrator_.explicit_tableau.c_i(iS) * dt;
    const double dt_b_ex = dt * integrator_.explicit_tableau.b_i(iS);
    dt_info.dt_coef = dt_b_ex;

    pkgs->apply_delta(u0, dt_info);

    accumulate_grid_motion(mesh_state, 0, iS, dt_b_ex, ib,
                           "Timestepper :: EX :: Finalize grid");
  }
  mesh.update_grid(Kokkos::subview(x_l_sumvar_, 0, Kokkos::ALL));

  auto sd0 = mesh_state(0);
  apply_slope_limiter(sl_hydro, u0, sd0, sd0.fluid_basis(), sd0.eos());
  bel::apply_bound_enforcing_limiter(sd0);

  pkgs->zero_delta();
}

void TimeStepper::step_imex(PackageManager *pkgs, MeshState &mesh_state,
                            TimeStepInfo &dt_info, SlopeLimiter *sl_hydro,
                            SlopeLimiter *sl_rad) {

  update_rad_hydro_imex(pkgs, mesh_state, dt_info, sl_hydro, sl_rad);
}

void TimeStepper::update_rad_hydro_imex(PackageManager *pkgs,
                                        MeshState &mesh_state,
                                        TimeStepInfo &dt_info,
                                        SlopeLimiter *sl_hydro,
                                        SlopeLimiter *sl_rad) {
  auto &mesh = mesh_state.mesh();
  const int nnodes = mesh.n_nodes();

  const int nvars = mesh_state.nvars("evolved");
  const IndexRange ib(mesh.domain<Domain::Entire>());
  const IndexRange qb(nnodes);
  const IndexRange vb(nvars);

  const double t = dt_info.t;
  const double dt = dt_info.dt;

  const auto &fluid_basis = mesh_state.fluid_basis();
  const auto &rad_basis = mesh_state.rad_basis();
  const auto &eos = mesh_state.eos();

  auto u0 = mesh_state(0).get_field("evolved");
  for (int iS = 0; iS < nStages_; ++iS) {
    dt_info.stage = iS;
    dt_info.t = t + integrator_.explicit_tableau.c_i(iS) * dt;
    auto stage_data = mesh_state.stage(iS);
    auto u = stage_data.get_field("evolved");
    reset_stage_sumvar(mesh, iS, u0, ib, qb, vb,
                       "Timestepper :: IMEX :: Reset sumvar");

    // --- Inner update loop ---

    for (int j = 0; j < iS; ++j) {
      dt_info.stage = j;
      dt_info.t = t + integrator_.explicit_tableau.c_i(j) * dt;
      const double dt_a = dt * integrator_.explicit_tableau.a_ij(iS, j);
      const double dt_a_im = dt * integrator_.implicit_tableau.a_ij(iS, j);
      dt_info.dt_coef = dt_a;
      dt_info.dt_coef_implicit = dt_a_im;

      pkgs->apply_delta(SumVar_U_, dt_info);

      accumulate_grid_motion(mesh_state, iS, j, dt_a, ib,
                             "Timestepper :: IMEX :: Update grid");
    } // End inner loop

    update_stage_mesh(mesh_state, iS);

    // set U_s (stage data)
    Kokkos::deep_copy(u, SumVar_U_);

    // Seems to be necessary when doing explicit transport.
    // stage_data.mesh() resolves to this stage's mesh (canonical for stage 0,
    // the work buffer otherwise).
    apply_slope_limiter(sl_rad, u, stage_data, rad_basis, eos);

    Kokkos::deep_copy(SumVar_U_, u);

    // implicit update
    dt_info.stage = iS;
    dt_info.t = t + integrator_.explicit_tableau.c_i(iS) * dt;
    dt_info.dt_coef = dt * integrator_.implicit_tableau.a_ij(iS, iS);

    // Need a fill derived?
    // pkgs->fill_derived(stage_data, dt_info);
    if (dt_info.dt_coef != 0.0) {
      pkgs->update_implicit(stage_data, SumVar_U_, dt_info);
    }

    apply_slope_limiter(sl_hydro, u, stage_data, fluid_basis, eos);
    apply_slope_limiter(sl_rad, u, stage_data, rad_basis, eos);
    bel::apply_bound_enforcing_limiter(stage_data);
    bel::apply_bound_enforcing_limiter_rad(stage_data);

    dt_info.stage = iS;
    pkgs->fill_derived(stage_data, dt_info);
    pkgs->update_explicit(stage_data, dt_info);
  } // end outer loop

  for (int iS = 0; iS < nStages_; ++iS) {
    dt_info.stage = iS;
    dt_info.t = t + integrator_.explicit_tableau.c_i(iS) * dt;
    const double dt_b = dt * integrator_.explicit_tableau.b_i(iS);
    const double dt_b_im = dt * integrator_.implicit_tableau.b_i(iS);
    dt_info.dt_coef = dt_b;
    dt_info.dt_coef_implicit = dt_b_im;

    pkgs->apply_delta(u0, dt_info);

    accumulate_grid_motion(mesh_state, 0, iS, dt_b, ib,
                           "Timestepper :: IMEX :: Finalize grid");
  }
  mesh.update_grid(Kokkos::subview(x_l_sumvar_, 0, Kokkos::ALL));

  auto sd0 = mesh_state(0);
  apply_slope_limiter(sl_hydro, u0, sd0, fluid_basis, eos);
  apply_slope_limiter(sl_rad, u0, sd0, rad_basis, eos);
  bel::apply_bound_enforcing_limiter(sd0);
  bel::apply_bound_enforcing_limiter_rad(sd0);

  pkgs->zero_delta();
}

[[nodiscard]] auto TimeStepper::n_stages() const noexcept -> int {
  return integrator_.num_stages;
}

// Computes number of evolved vars.
// Can't be used for mass fractions when mixing is considered
// Will have to remove / change at that point.
[[nodiscard]] auto TimeStepper::nvars_evolved(const ProblemIn *pin) noexcept
    -> int {
  static const int base = 3;
  static const bool rad_enabled =
      pin->param()->get<bool>("physics.radiation.enabled");
  static const bool composition_enabled =
      pin->param()->get<bool>("physics.composition.enabled");

  int additional_vars = 0;
  if (rad_enabled) {
    additional_vars += 2;
  }

  if (composition_enabled) {
    const int ncomps = pin->param()->get<int>("composition.ncomps");
    additional_vars += ncomps;
  }

  return base + additional_vars;
}

} // namespace athelas
