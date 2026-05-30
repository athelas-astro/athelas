#include "timestepper/timestepper.hpp"

#include <vector>

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
TimeStepper::TimeStepper(const ProblemIn *pin, Mesh *mesh)
    : nvars_evolved_(nvars_evolved(pin)), mSize_(mesh->n_elements() + 2),
      integrator_(
          create_tableau(pin->param()->get<MethodID>("time.integrator"))),
      nStages_(integrator_.num_stages), tOrder_(integrator_.explicit_order),
      SumVar_U_("SumVar_U", mSize_, pin->param()->get<int>("basis.nnodes"),
                nvars_evolved_),
      mesh_s_(), x_l_sumvar_("x_l_sumvar_", nStages_ + 1, mSize_ + 1) {
  mesh_s_.reserve(nStages_ + 1);
  for (int iS = 0; iS <= nStages_; ++iS) {
    mesh_s_.emplace_back(pin);
  }
}

void TimeStepper::seed_stage_meshes(const Mesh &mesh) {
  for (int iS = 0; iS < nStages_; ++iS) {
    mesh_s_[iS].copy_from(mesh);
  }
}

void TimeStepper::reset_stage_sumvar(int stage, AthelasArray3D<double> u0,
                                     const IndexRange &ib, const IndexRange &qb,
                                     const IndexRange &vb, const char *label) {
  auto left_interface = mesh_s_[stage].x_l();
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, label, DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        for (int v = vb.s; v <= vb.e; ++v) {
          SumVar_U_(i, q, v) = u0(i, q, v);
        }
        x_l_sumvar_(stage, i) = left_interface(i);
      });
}

void TimeStepper::accumulate_grid_motion(MeshState &mesh_state, int sum_stage,
                                         int data_stage, double dt_coef,
                                         const IndexRange &ib,
                                         const char *label) {
  auto facedata =
      mesh_state(data_stage).get_field<AthelasArray2D<double>>("facedata");
  const int idx_vstar = mesh_state(data_stage).var_index("facedata", "vstar");
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, label, DevExecSpace(), ib.s, ib.e,
      KOKKOS_CLASS_LAMBDA(const int i) {
        x_l_sumvar_(sum_stage, i) += dt_coef * facedata(i, idx_vstar);
      });
}

void TimeStepper::update_stage_mesh(const Mesh &mesh, int mesh_stage,
                                    int sum_stage) {
  auto x_l_sumvar = Kokkos::subview(x_l_sumvar_, sum_stage, Kokkos::ALL);
  mesh_s_[mesh_stage].copy_from(mesh);
  mesh_s_[mesh_stage].update_grid(x_l_sumvar);
}

void TimeStepper::step(PackageManager *pkgs, MeshState &mesh_state, Mesh &mesh,
                       TimeStepInfo &dt_info, SlopeLimiter *sl_hydro) {
  // hydro explicit update
  update_fluid_explicit(pkgs, mesh_state, mesh, dt_info, sl_hydro);
}

void TimeStepper::update_fluid_explicit(PackageManager *pkgs,
                                        MeshState &mesh_state, Mesh &mesh,
                                        TimeStepInfo &dt_info,
                                        SlopeLimiter *sl_hydro) {
  const int nvars = mesh_state.nvars("u_cf");
  const IndexRange ib(mesh.domain<Domain::Entire>());
  const IndexRange qb(mesh.n_nodes());
  const IndexRange vb(nvars);

  seed_stage_meshes(mesh);

  const double t = dt_info.t;
  const double dt = dt_info.dt;

  const auto &fluid_basis = mesh_state.fluid_basis();
  const auto &eos = mesh_state.eos();

  auto u0 = mesh_state(0).get_field("u_cf");
  for (int iS = 0; iS < nStages_; ++iS) {
    dt_info.stage = iS;

    // re-set the summation variables `SumVar`
    auto stage_data = mesh_state.stage(iS);
    auto u = stage_data.get_field("u_cf");
    reset_stage_sumvar(iS, u0, ib, qb, vb, "Timestepper :: EX :: Reset sumvar");

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

    update_stage_mesh(mesh, iS, iS);

    apply_slope_limiter(sl_hydro, u, mesh_s_[iS], fluid_basis, eos);
    bel::apply_bound_enforcing_limiter(stage_data, mesh_s_[iS]);

    dt_info.stage = iS;
    dt_info.t = t + integrator_.explicit_tableau.c_i(iS) * dt;
    pkgs->fill_derived(stage_data, mesh_s_[iS], dt_info);
    pkgs->update_explicit(stage_data, mesh_s_[iS], dt_info);
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
  apply_slope_limiter(sl_hydro, u0, mesh, sd0.fluid_basis(), sd0.eos());
  bel::apply_bound_enforcing_limiter(sd0, mesh);

  pkgs->zero_delta();
}

void TimeStepper::step_imex(PackageManager *pkgs, MeshState &mesh_state,
                            Mesh &mesh, TimeStepInfo &dt_info,
                            SlopeLimiter *sl_hydro, SlopeLimiter *sl_rad) {

  update_rad_hydro_imex(pkgs, mesh_state, mesh, dt_info, sl_hydro, sl_rad);
}

void TimeStepper::update_rad_hydro_imex(PackageManager *pkgs,
                                        MeshState &mesh_state, Mesh &mesh,
                                        TimeStepInfo &dt_info,
                                        SlopeLimiter *sl_hydro,
                                        SlopeLimiter *sl_rad) {
  const int nnodes = mesh.n_nodes();

  const int nvars = mesh_state.nvars("u_cf");
  const IndexRange ib(mesh.domain<Domain::Entire>());
  const IndexRange qb(nnodes);
  const IndexRange vb(nvars);

  seed_stage_meshes(mesh);

  const double t = dt_info.t;
  const double dt = dt_info.dt;

  const auto &fluid_basis = mesh_state.fluid_basis();
  const auto &rad_basis = mesh_state.rad_basis();
  const auto &eos = mesh_state.eos();

  auto u0 = mesh_state(0).get_field("u_cf");
  for (int iS = 0; iS < nStages_; ++iS) {
    dt_info.stage = iS;
    dt_info.t = t + integrator_.explicit_tableau.c_i(iS) * dt;
    auto stage_data = mesh_state.stage(iS);
    auto u = stage_data.get_field("u_cf");
    reset_stage_sumvar(iS, u0, ib, qb, vb,
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

    update_stage_mesh(mesh, iS, iS);

    // set U_s (stage data)
    Kokkos::deep_copy(u, SumVar_U_);

    // Seems to be necessary when doing explicit transport.
    apply_slope_limiter(sl_rad, u, mesh_s_[iS], rad_basis, eos);

    Kokkos::deep_copy(SumVar_U_, u);

    // implicit update
    dt_info.stage = iS;
    dt_info.t = t + integrator_.explicit_tableau.c_i(iS) * dt;
    dt_info.dt_coef = dt * integrator_.implicit_tableau.a_ij(iS, iS);

    // Need a fill derived?
    // pkgs->fill_derived(stage_data, mesh_s_[iS], dt_info);
    if (dt_info.dt_coef != 0.0) {
      pkgs->update_implicit(stage_data, SumVar_U_, mesh_s_[iS], dt_info);
    }

    apply_slope_limiter(sl_hydro, u, mesh_s_[iS], fluid_basis, eos);
    apply_slope_limiter(sl_rad, u, mesh_s_[iS], rad_basis, eos);
    bel::apply_bound_enforcing_limiter(stage_data, mesh_s_[iS]);
    bel::apply_bound_enforcing_limiter_rad(stage_data, mesh_s_[iS]);

    dt_info.stage = iS;
    pkgs->fill_derived(stage_data, mesh_s_[iS], dt_info);
    pkgs->update_explicit(stage_data, mesh_s_[iS], dt_info);
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
  apply_slope_limiter(sl_hydro, u0, mesh, fluid_basis, eos);
  apply_slope_limiter(sl_rad, u0, mesh, rad_basis, eos);
  bel::apply_bound_enforcing_limiter(sd0, mesh);
  bel::apply_bound_enforcing_limiter_rad(sd0, mesh);

  pkgs->zero_delta();
}

[[nodiscard]] auto TimeStepper::n_stages() const noexcept -> int {
  return integrator_.num_stages;
}

// TEMPORARY: see header. Remove when mesh_s_ leaves TimeStepper.
[[nodiscard]] auto TimeStepper::compute_n_stages(const ProblemIn *pin) -> int {
  return create_tableau(pin->param()->get<MethodID>("time.integrator"))
      .num_stages;
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
