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
      mSize_(pin->param()->get<int>("mesh.nx") + 2),
      nNodes_(pin->param()->get<int>("basis.nnodes")),
      integrator_(
          create_tableau(pin->param()->get<MethodID>("time.integrator"))),
      nStages_(integrator_.num_stages), tOrder_(integrator_.explicit_order),
      SumVar_U_("SumVar_U", mSize_, nNodes_, nvars_evolved_),
      u0_buffer_("u0_buffer", mSize_, nNodes_, nvars_evolved_),
      x_inner_sumvar_("x_inner_sumvar_", nStages_) {}

void TimeStepper::reset_stage_sumvar(AthelasArray3D<double> u0,
                                     const IndexRange &ib, const IndexRange &qb,
                                     const IndexRange &vb, const char *label) {
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, label, DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        for (int v = vb.s; v <= vb.e; ++v) {
          SumVar_U_(i, q, v) = u0(i, q, v);
        }
      });
}

// Set this stage's buffer to the canonical inner-face radius x_l^n(ilo). The
// inner loop then advances it by v*(ilo) for the stages that contribute.
void TimeStepper::reset_x_inner_buffer(const Mesh &mesh, const int stage) {
  auto x_l = mesh.x_l();
  const int ilo = Mesh::get_ilo();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Timestepper :: Reset x_inner buffer",
      DevExecSpace(), 0, 0,
      KOKKOS_CLASS_LAMBDA(const int) { x_inner_sumvar_(stage) = x_l(ilo); });
}

// Advance the `sum_stage` buffer by dt_coef * v*(ilo) taken from `data_stage`,
// mirroring the RK accumulation applied to the evolved state.
void TimeStepper::accumulate_x_inner_buffer(MeshState &mesh_state,
                                            const int sum_stage,
                                            const int data_stage,
                                            const double dt_coef) {
  auto interface =
      mesh_state(data_stage).get_field<AthelasArray2D<double>>("interface");
  const int idx_vstar =
      mesh_state(data_stage).var_index("interface", "interface_velocity");
  const int ilo = Mesh::get_ilo();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Timestepper :: Accumulate x_inner buffer",
      DevExecSpace(), 0, 0, KOKKOS_CLASS_LAMBDA(const int) {
        x_inner_sumvar_(sum_stage) += dt_coef * interface(ilo, idx_vstar);
      });
}

auto TimeStepper::x_inner_buffer(const int stage) -> double {
  double x_inner = 0.0;
  Kokkos::deep_copy(x_inner, Kokkos::subview(x_inner_sumvar_, stage));
  return x_inner;
}

void TimeStepper::update_stage_mesh(MeshState &mesh_state, int stage,
                                    AthelasArray3D<double> evolved) {
  // Stage 0 is the canonical mesh. Later stages are rebuilt into the single
  // work buffer from the canonical mesh and this stage's tau field. The inner
  // face is positioned by this stage's accumulated x_inner buffer.
  if (stage == 0) {
    mesh_state.mesh().reconstruct_mesh(evolved, x_inner_buffer(0));
    return;
  }
  auto &mesh_stage = mesh_state.mesh_stage();
  mesh_stage.copy_from(mesh_state.mesh());
  mesh_stage.reconstruct_mesh(evolved, x_inner_buffer(stage));
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

  const auto &basis = mesh_state.basis();
  const auto &eos = mesh_state.eos();

  auto u0 = mesh_state(0).get_field("evolved");
  for (int iS = 0; iS < nStages_; ++iS) {
    dt_info.stage = iS;

    // re-set the summation variables `SumVar`
    auto stage_data = mesh_state.stage(iS);
    auto u = stage_data.get_field("evolved");
    reset_stage_sumvar(u0, ib, qb, vb, "Timestepper :: EX :: Reset sumvar");
    reset_x_inner_buffer(mesh, iS);

    // --- Inner update loop ---

    for (int j = 0; j < iS; ++j) {
      dt_info.stage = j;
      dt_info.t = t + integrator_.explicit_tableau.c_i(j) * dt;
      const double dt_a_ex = dt * integrator_.explicit_tableau.a_ij(iS, j);
      dt_info.dt_coef = dt_a_ex;

      pkgs->apply_delta(SumVar_U_, dt_info);
      accumulate_x_inner_buffer(mesh_state, iS, j, dt_a_ex);
    } // End inner loop

    // set U_s
    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "Timestepper :: EX :: Set Us", DevExecSpace(),
        ib.s, ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
          for (int v = vb.s; v <= vb.e; ++v) {
            u(i, q, v) = SumVar_U_(i, q, v);
          }
        });

    update_stage_mesh(mesh_state, iS, u);

    // stage_data.mesh() resolves to this stage's mesh (canonical for stage 0,
    // the work buffer otherwise).
    apply_slope_limiter(sl_hydro, u, stage_data, basis, eos);
    bel::apply_bound_enforcing_limiter(stage_data);
    update_stage_mesh(mesh_state, iS, u);

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
    accumulate_x_inner_buffer(mesh_state, 0, iS, dt_b_ex);
  }
  const double x_inner = x_inner_buffer(0);
  mesh.reconstruct_mesh(u0, x_inner);

  auto sd0 = mesh_state(0);
  apply_slope_limiter(sl_hydro, u0, sd0, sd0.basis(), sd0.eos());
  bel::apply_bound_enforcing_limiter(sd0);
  mesh.reconstruct_mesh(u0, x_inner);

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

  const auto &basis = mesh_state.basis();
  const auto &eos = mesh_state.eos();

  auto u0 = mesh_state(0).get_field("evolved");
  auto initial_state = mesh_state(0);
  bel::apply_bound_enforcing_limiter_rad(initial_state);
  bel::apply_bound_enforcing_limiter(initial_state);
  Kokkos::deep_copy(u0_buffer_, u0);
  for (int iS = 0; iS < nStages_; ++iS) {
    dt_info.stage = iS;
    dt_info.t = t + integrator_.explicit_tableau.c_i(iS) * dt;
    auto stage_data = mesh_state.stage(iS);
    auto u = stage_data.get_field("evolved");
    reset_stage_sumvar(u0_buffer_, ib, qb, vb,
                       "Timestepper :: IMEX :: Reset sumvar");
    reset_x_inner_buffer(mesh, iS);

    // --- Inner update loop ---

    for (int j = 0; j < iS; ++j) {
      dt_info.stage = j;
      dt_info.t = t + integrator_.explicit_tableau.c_i(j) * dt;
      const double dt_a = dt * integrator_.explicit_tableau.a_ij(iS, j);
      const double dt_a_im = dt * integrator_.implicit_tableau.a_ij(iS, j);
      dt_info.dt_coef = dt_a;
      dt_info.dt_coef_implicit = dt_a_im;

      pkgs->apply_delta(SumVar_U_, dt_info);
      accumulate_x_inner_buffer(mesh_state, iS, j, dt_a);
    } // End inner loop

    // set U_s (stage data)
    Kokkos::deep_copy(u, SumVar_U_);
    update_stage_mesh(mesh_state, iS, u);

    // Seems to be necessary when doing explicit transport.
    apply_slope_limiter(sl_rad, u, stage_data, basis, eos);

    Kokkos::deep_copy(SumVar_U_, u);

    // implicit update
    dt_info.stage = iS;
    dt_info.t = t + integrator_.explicit_tableau.c_i(iS) * dt;
    dt_info.dt_coef = dt * integrator_.implicit_tableau.a_ij(iS, iS);

    // The implicit source and EOS Jacobian must use temperature, composition,
    // and ionization data derived from this stage state. In particular,
    // Paczynski/Saha data from the previous stage can define a different
    // Newton system from the one represented by this stage state.
    pkgs->fill_derived(stage_data, dt_info);
    if (dt_info.dt_coef != 0.0) {
      pkgs->update_implicit(stage_data, SumVar_U_, dt_info);
    }
    Kokkos::deep_copy(u, SumVar_U_);

    apply_slope_limiter(sl_hydro, u, stage_data, basis, eos);
    apply_slope_limiter(sl_rad, u, stage_data, basis, eos);
    bel::apply_bound_enforcing_limiter_rad(stage_data);
    bel::apply_bound_enforcing_limiter(stage_data);
    update_stage_mesh(mesh_state, iS, u);

    dt_info.stage = iS;
    pkgs->fill_derived(stage_data, dt_info);
    pkgs->update_explicit(stage_data, dt_info);
  } // end outer loop

  // Stage 0 shares storage with the canonical state. Restore U^n before
  // accumulating the explicit and implicit RK weights from every stage.
  Kokkos::deep_copy(u0, u0_buffer_);
  for (int iS = 0; iS < nStages_; ++iS) {
    dt_info.stage = iS;
    dt_info.t = t + integrator_.explicit_tableau.c_i(iS) * dt;
    const double dt_b = dt * integrator_.explicit_tableau.b_i(iS);
    const double dt_b_im = dt * integrator_.implicit_tableau.b_i(iS);
    dt_info.dt_coef = dt_b;
    dt_info.dt_coef_implicit = dt_b_im;

    pkgs->apply_delta(u0, dt_info);
    accumulate_x_inner_buffer(mesh_state, 0, iS, dt_b);
  }
  const double x_inner = x_inner_buffer(0);
  mesh.reconstruct_mesh(u0, x_inner);

  auto sd0 = mesh_state(0);
  apply_slope_limiter(sl_hydro, u0, sd0, basis, eos);
  apply_slope_limiter(sl_rad, u0, sd0, basis, eos);
  bel::apply_bound_enforcing_limiter_rad(sd0);
  bel::apply_bound_enforcing_limiter(sd0);
  mesh.reconstruct_mesh(u0, x_inner);

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
