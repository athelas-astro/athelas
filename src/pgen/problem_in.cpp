#include "pgen/problem_in.hpp"
#include <cmath>

#include "bc/boundary_conditions_base.hpp"
#include "geometry/mesh.hpp"
#include "io/restart.hpp"
#include "lua_schema.hpp"
#include "pgen/lua_validator.hpp"
#include "timestepper/tableau.hpp"
#include "utils/error.hpp"
#include "utils/utilities.hpp"

namespace athelas {

// Provide access to the underlying params object
auto ProblemIn::param() -> Params * { return params_.get(); }

[[nodiscard]] auto ProblemIn::param() const -> Params * {
  return params_.get();
}

ProblemIn::ProblemIn(
    const std::string &fn, const std::string &output_dir,
    const std::vector<std::pair<std::string, std::string>> &overrides) {

  // ---------------------------------
  // ---------- Load config ----------
  // ---------------------------------
  lua_.open_libraries(sol::lib::base, sol::lib::math);
  try {
    config_ = lua_.script_file(fn);

    // Apply command-line overrides. The input deck returns `config`, so the
    // local table is not visible to subsequent snippets; re-expose it
    // mutate via Lua, and read back.
    if (!overrides.empty()) {
      lua_["config"] = config_;
      for (const auto &[key, expr] : overrides) {
        try {
          lua_.script(std::format("config.{} = {}", key, expr));
        } catch (const sol::error &e) {
          std::println(std::cerr, "Override failed for '{} = {}': {}", key,
                       expr, e.what());
          std::println(std::cerr,
                       "Hint: --key=value is parsed as Lua source; strings "
                       "must be Lua-quoted, e.g. --key='\"value\"'.");
          throw_athelas_error("Invalid command-line override");
        }
        std::println("# Override applied: {} = {}", key, expr);
      }
      config_ = lua_["config"];
    }

    sol::table schema = lua_.script(ATHELAS_SCHEMA_LUA);
    Validator validator(schema);
    validator.validate(config_);
  } catch (const sol::error &e) {
    std::cerr << e.what() << "\n";
    throw_athelas_error("Issue reading input deck!");
  }
  params_ = std::make_unique<Params>();

  std::println("# Loading Input Deck ...");

  // -----------------------------------
  // ---------- problem block ----------
  // -----------------------------------
  sol::optional<sol::table> problem_block = config_["problem"];
  sol::table problem = *problem_block;

  sol::optional<std::string> pname = problem["name"];
  params_->add("problem.name", *pname);

  sol::optional<double> tf = problem["t_end"];
  if (*tf <= 0.0) {
    throw_athelas_error("tf must be > 0.0!");
  }
  params_->add("problem.tf", *tf);

  const double nlim = problem.get_or("nlim", -1.0);
  params_->add("problem.nlim", nlim);

  sol::optional<double> xl = problem["xl"];
  params_->add("problem.xl", *xl);

  sol::optional<double> xr = problem["xr"];
  if (*xr <= *xl) {
    throw_athelas_error("xr must be > xl!");
  }
  params_->add("problem.xr", *xr);

  sol::optional<int> nx = problem["nx"];
  if (*nx <= 0) {
    throw_athelas_error("nx must be > 0!");
  }
  params_->add("problem.nx", *nx);

  // NOTE: It may be worthwhile to have cfl be registered per physics.
  sol::optional<double> cfl = problem["cfl"];
  if (*cfl <= 0.0) {
    throw_athelas_error("cfl must be > 0.0!");
  }
  params_->add("problem.cfl", *cfl);

  sol::optional<std::string> geom = problem["geometry"];
  params_->add("problem.geometry", *geom);

  // TODO(astrobarker): move mesh stuff into own section, not problem
  // Begs the question: what is "problem" and what is "grid"? xl and xr?
  sol::optional<std::string> grid_type = problem["grid_type"];
  params_->add("problem.grid_type", *grid_type);

  // ---------------------------------------------
  // ---------- handle [problem.params] ----------
  // ---------------------------------------------
  sol::optional<sol::table> pparams_block = problem["params"];
  sol::table pparams = *pparams_block;

  for (auto &[key_obj, val_obj] : pparams) {
    const std::string key = key_obj.as<std::string>();
    const std::string out_key = "problem.params." + key;

    // Check int before double: in Lua 5.3+ integers and floats are distinct
    // subtypes. sol2 exposes this — is<int>() is true only for integer-valued
    // numbers, while is<double>() would also match. Check int first.
    if (val_obj.is<bool>()) {
      params_->add(out_key, val_obj.as<bool>());
    } else if (val_obj.is<int>()) {
      params_->add(out_key, val_obj.as<int>());
    } else if (val_obj.is<double>()) {
      params_->add(out_key, val_obj.as<double>());
    } else if (val_obj.is<std::string>()) {
      params_->add(out_key, val_obj.as<std::string>());
    } else {
      throw_athelas_error("Unsupported Lua type for key: " + key);
    }
  }

  // -----------------------------------
  // ---------- physics block ----------
  // -----------------------------------
  sol::optional<sol::table> physics_block = config_["physics"];
  sol::table physics = *physics_block;

  sol::optional<bool> rad = physics["radiation"];
  params_->add("physics.radiation.enabled", *rad);

  sol::optional<bool> grav = physics["gravity"];
  params_->add("physics.gravity.enabled", *grav);

  sol::optional<bool> comps = physics["composition"];
  params_->add("physics.composition.enabled", *comps);

  sol::optional<bool> ion = physics["ionization"];
  params_->add("physics.ionization.enabled", *ion);

  sol::optional<bool> engine = physics["engine"];
  params_->add("physics.engine.enabled", *engine);

  if (*ion && !*comps) {
    throw_athelas_error("Ionization enabled but composition disabled!");
  }

  sol::optional<bool> heating = physics["heating"];
  params_->add("physics.heating.active", *heating);

  // ---------------------------------
  // ---------- basis block ----------
  // ---------------------------------
  sol::optional<sol::table> basis_block = config_["basis"];
  sol::table basis = *basis_block;

  sol::optional<int> nnodes = basis["nnodes"];
  params_->add("basis.nnodes", *nnodes);

  // ---------------------------------
  // ---------- fluid block ----------
  // ---------------------------------
  sol::optional<sol::table> fluid_block = config_["fluid"];
  sol::table fluid = *fluid_block;

  // operator_split is not supported for fluid; error if user set it
  if (fluid["operator_split"] != sol::lua_nil) {
    throw_athelas_error("Operator split not supported for fluid! Remove option "
                        "from [fluid] block.");
  }

  sol::table fluid_limiter_block = fluid["limiter"];
  // Use an empty table as a safe stand-in when the block is absent so all
  // get_or calls below simply return their defaults.
  //  sol::table fluid_lim = fluid_limiter_block.value_or(lua_.create_table());

  const bool limit_fluid = fluid_limiter_block.get_or("enabled", true);
  params_->add("fluid.limiter.enabled", limit_fluid);

  const std::string fluid_limiter_type =
      fluid_limiter_block.get_or<std::string>("type", "minmod");
  params_->add("fluid.limiter.type", fluid_limiter_type);

  if (limit_fluid &&
      (fluid_limiter_type == "minmod" || fluid_limiter_type == "moment")) {
    const double b_tvd = fluid_limiter_block.get_or("b_tvd", 1.0);
    athelas_requires(
        std::isfinite(b_tvd) && b_tvd >= 1.0 && b_tvd <= 2.0,
        "[fluid] block: limiter b_tvd must be finite and in [1, 2].");
    params_->add("fluid.limiter.b_tvd", b_tvd);
    const double m_tvb = fluid_limiter_block.get_or("m_tvb", 0.0);
    athelas_requires(std::isfinite(m_tvb) && m_tvb >= 0.0,
                     "[fluid] block: limiter m_tvb must be finite and >= 0.");
    params_->add("fluid.limiter.m_tvb", m_tvb);
  }

  if (limit_fluid && fluid_limiter_type == "weno") {
    sol::optional<double> gamma_i = fluid_limiter_block["gamma_i"];
    sol::optional<double> gamma_l = fluid_limiter_block["gamma_l"];
    sol::optional<double> gamma_r = fluid_limiter_block["gamma_r"];
    if (gamma_i && !gamma_l && !gamma_r) {
      params_->add("fluid.limiter.gamma_i", *gamma_i);
      params_->add("fluid.limiter.gamma_l", (1.0 - *gamma_i) / 2.0);
      params_->add("fluid.limiter.gamma_r", (1.0 - *gamma_i) / 2.0);
    } else if (gamma_i && gamma_l && gamma_r) {
      params_->add("fluid.limiter.gamma_i", *gamma_i);
      params_->add("fluid.limiter.gamma_l", *gamma_l);
      params_->add("fluid.limiter.gamma_r", *gamma_r);
    } else {
      throw_athelas_error("Error parsing weno gammas in [fluid] block: provide "
                          "only gamma_i, or all gamma_l, gamma_i, gamma_r!");
    }
    const double sum_g = params_->get<double>("fluid.limiter.gamma_i") +
                         params_->get<double>("fluid.limiter.gamma_l") +
                         params_->get<double>("fluid.limiter.gamma_r");
    if (std::abs(sum_g - 1.0) > 1.0e-10) {
      throw_athelas_error(
          " ! Initialization Error: Linear WENO weights must sum to unity.");
    }
    const double weno_p = fluid_limiter_block["weno_p"];
    if (weno_p <= 0.0) {
      throw_athelas_error(
          "[fluid] block: WENO limiter weno_p must be positive!");
    }
    params_->add("fluid.limiter.weno_p", weno_p);
  }

  // tci
  const bool do_fluid_tci = fluid_limiter_block["tci_opt"];
  params_->add("fluid.limiter.tci_enabled", do_fluid_tci);
  if (do_fluid_tci) {
    sol::optional<double> tci_val = fluid_limiter_block["tci_val"];
    params_->add("fluid.limiter.tci_val", *tci_val);
  } else {
    params_->add("fluid.limiter.tci_val", 0.0);
  }

  // characteristic limiting
  const bool fluid_characteristic = fluid_limiter_block["characteristic"];
  params_->add("fluid.limiter.characteristic", fluid_characteristic);

  // --- fluid bc ---
  sol::optional<sol::table> bc_block = config_["bc"];
  sol::optional<sol::table> fluid_bc_block =
      bc_block ? sol::optional<sol::table>((*bc_block)["fluid"]) : sol::nullopt;

  sol::optional<std::string> fluid_bc_i =
      fluid_bc_block ? sol::optional<std::string>((*fluid_bc_block)["bc_i"])
                     : sol::nullopt;
  sol::optional<std::string> fluid_bc_o =
      fluid_bc_block ? sol::optional<std::string>((*fluid_bc_block)["bc_o"])
                     : sol::nullopt;

  const auto fluid_bc_i_lc = utilities::to_lower(*fluid_bc_i);
  const auto fluid_bc_o_lc = utilities::to_lower(*fluid_bc_o);
  params_->add("fluid.bc.i", fluid_bc_i_lc);
  params_->add("fluid.bc.o", fluid_bc_o_lc);

  const auto fluid_i_type = bc::parse_bc_type(fluid_bc_i_lc);
  const auto fluid_o_type = bc::parse_bc_type(fluid_bc_o_lc);
  bc::validate_fluid_bc(fluid_i_type);
  bc::validate_fluid_bc(fluid_o_type);
  if ((fluid_i_type == bc::BcType::Periodic) !=
      (fluid_o_type == bc::BcType::Periodic)) {
    throw_athelas_error("Periodic fluid boundaries must be set on both sides.");
  }

  double fluid_i_surface_pressure = 0.0;
  double fluid_o_surface_pressure = 0.0;

  if (fluid_i_type == bc::BcType::Surface) {
    fluid_i_surface_pressure =
        fluid_bc_block
            ? sol::optional<double>((*fluid_bc_block)["surface_pressure_i"])
                  .value_or(0.0)
            : 0.0;
  }
  if (fluid_o_type == bc::BcType::Surface) {
    fluid_o_surface_pressure =
        fluid_bc_block
            ? sol::optional<double>((*fluid_bc_block)["surface_pressure_o"])
                  .value_or(0.0)
            : 0.0;
  }
  params_->add("fluid.bc.i.surface_pressure", fluid_i_surface_pressure);
  params_->add("fluid.bc.o.surface_pressure", fluid_o_surface_pressure);

  // -------------------------------------
  // ---------- radiation block ----------
  // -------------------------------------
  // I suspect much of this should really go into the individual packages.
  if (*rad) {
    sol::optional<sol::table> rad_block = config_["radiation"];
    sol::table radiation = *rad_block;

    if (radiation["operator_split"] != sol::lua_nil) {
      throw_athelas_error(
          "Operator split not yet supported for radiation! Remove "
          "option from [radiation] block.");
    }

    const std::string discretization_type =
        rad_block->get<std::string>("discretization");
    if (discretization_type != "implicit" &&
        discretization_type != "explicit") {
      throw_athelas_error(
          "radiation.discretization must be 'explicit' or 'implicit'.");
    }
    params_->add("radiation.discretization", discretization_type);

    // AP LLF dissipation damping. The single coefficient C in the damping
    // factor 1 / (1 + C tau) is the only knob: C = 0 recovers standard LLF
    // (no damping), C > 0 enables the optical-depth correction.
    const double ap_coefficient = radiation.get_or("ap_coefficient", 0.0);
    athelas_requires(std::isfinite(ap_coefficient) && ap_coefficient >= 0.0,
                     "[radiation] block: ap_coefficient must be finite and >= "
                     "0.");
    params_->add("radiation.ap_coefficient", ap_coefficient);

    // implicit radiation timestep controls
    if (discretization_type == "implicit") {
      sol::table timestep = radiation["timestep"];
      const double max_frac_change_e =
          timestep.get<double>("max_fractional_change_e");
      const double energy_change_scale =
          timestep.get_or("energy_change_scale", 1.0e-10);
      const double max_change_f = timestep.get<double>("max_change_f");
      athelas_requires(
          std::isfinite(max_frac_change_e) && max_frac_change_e > 0.0,
          "[radiation.timestep] block: max_fractional_change_e must be finite "
          "and > 0.");
      athelas_requires(
          std::isfinite(energy_change_scale) && energy_change_scale >= 0.0,
          "[radiation.timestep] block: energy_change_scale must be finite and "
          ">= 0.");
      athelas_requires(std::isfinite(max_change_f) && max_change_f > 0.0,
                       "[radiation.timestep] block: max_change_f must be "
                       "finite and > 0.");
      params_->add("radiation.timestep.max_fractional_change_e",
                   max_frac_change_e);
      params_->add("radiation.timestep.energy_change_scale",
                   energy_change_scale);
      params_->add("radiation.timestep.max_change_f", max_change_f);

      sol::optional<sol::table> newton_block = radiation["newton"];
      sol::table newton = newton_block.value_or(lua_.create_table());
      params_->add("radiation.newton.max_iter", newton.get_or("max_iter", 10));
      params_->add("radiation.newton.tol", newton.get_or("tol", 1.0e-6));
    }

    sol::optional<sol::table> rad_limiter_block = radiation["limiter"];
    sol::table rad_lim = rad_limiter_block.value_or(lua_.create_table());

    const bool limit_rad = rad_lim.get_or("enabled", true);
    params_->add("radiation.limiter.enabled", limit_rad);

    const std::string rad_limiter_type =
        rad_lim.get_or<std::string>("type", "minmod");
    params_->add("radiation.limiter.type", rad_limiter_type);

    if (limit_rad &&
        (rad_limiter_type == "minmod" || rad_limiter_type == "moment")) {
      const double b_tvd = rad_lim.get_or("b_tvd", 1.0);
      athelas_requires(
          std::isfinite(b_tvd) && b_tvd >= 1.0 && b_tvd <= 2.0,
          "[radiation] block: limiter b_tvd must be finite and in [1, 2].");
      params_->add("radiation.limiter.b_tvd", b_tvd);
      const double m_tvb = rad_lim.get_or("m_tvb", 0.0);
      athelas_requires(
          std::isfinite(m_tvb) && m_tvb >= 0.0,
          "[radiation] block: limiter m_tvb must be finite and >= 0.");
      params_->add("radiation.limiter.m_tvb", m_tvb);
    }

    if (limit_rad && rad_limiter_type == "weno") {
      sol::optional<double> gamma_i = rad_lim["gamma_i"];
      sol::optional<double> gamma_l = rad_lim["gamma_l"];
      sol::optional<double> gamma_r = rad_lim["gamma_r"];
      if (gamma_i && !gamma_l && !gamma_r) {
        params_->add("radiation.limiter.gamma_i", *gamma_i);
        params_->add("radiation.limiter.gamma_l", (1.0 - *gamma_i) / 2.0);
        params_->add("radiation.limiter.gamma_r", (1.0 - *gamma_i) / 2.0);
      } else if (gamma_i && gamma_l && gamma_r) {
        params_->add("radiation.limiter.gamma_i", *gamma_i);
        params_->add("radiation.limiter.gamma_l", *gamma_l);
        params_->add("radiation.limiter.gamma_r", *gamma_r);
      } else {
        throw_athelas_error(
            "Error parsing weno gammas in [radiation] block: provide only "
            "gamma_i, or all gamma_l, gamma_i, gamma_r!");
      }
      const double sum_g = params_->get<double>("radiation.limiter.gamma_i") +
                           params_->get<double>("radiation.limiter.gamma_l") +
                           params_->get<double>("radiation.limiter.gamma_r");
      if (std::abs(sum_g - 1.0) > 1.0e-10) {
        throw_athelas_error(
            " ! Initialization Error: Linear WENO weights must sum to unity.");
      }
      const double weno_p = rad_lim.get_or("weno_p", 2.0);
      if (weno_p <= 0.0) {
        throw_athelas_error(
            "[radiation] block: WENO limiter weno_p must be positive!");
      }
      params_->add("radiation.limiter.weno_p", weno_p);
    }

    // tci
    const bool do_rad_tci = rad_lim.get_or("tci_opt", false);
    params_->add("radiation.limiter.tci_enabled", do_rad_tci);
    if (do_rad_tci) {
      sol::optional<double> tci_val = rad_lim["tci_val"];
      params_->add("radiation.limiter.tci_val", *tci_val);
    } else {
      params_->add("radiation.limiter.tci_val", 0.0);
    }

    // characteristic limiting
    const bool rad_characteristic = rad_lim.get_or("characteristic", false);
    params_->add("radiation.limiter.characteristic", rad_characteristic);

    // --- radiation bc ---
    sol::optional<sol::table> rad_bc_block =
        bc_block ? sol::optional<sol::table>((*bc_block)["radiation"])
                 : sol::nullopt;

    sol::optional<std::string> rad_bc_i =
        rad_bc_block ? sol::optional<std::string>((*rad_bc_block)["bc_i"])
                     : sol::nullopt;
    sol::optional<std::string> rad_bc_o =
        rad_bc_block ? sol::optional<std::string>((*rad_bc_block)["bc_o"])
                     : sol::nullopt;

    const auto rad_bc_i_lc = utilities::to_lower(*rad_bc_i);
    const auto rad_bc_o_lc = utilities::to_lower(*rad_bc_o);
    params_->add("radiation.bc.i", rad_bc_i_lc);
    params_->add("radiation.bc.o", rad_bc_o_lc);

    const auto rad_i_type = bc::parse_bc_type(rad_bc_i_lc);
    const auto rad_o_type = bc::parse_bc_type(rad_bc_o_lc);
    bc::validate_radiation_bc(Boundary::Interior, rad_i_type);
    bc::validate_radiation_bc(Boundary::Exterior, rad_o_type);
    if ((rad_i_type == bc::BcType::Periodic) !=
        (rad_o_type == bc::BcType::Periodic)) {
      throw_athelas_error(
          "Periodic radiation boundaries must be set on both sides.");
    }
    if (discretization_type == "implicit" &&
        rad_i_type == bc::BcType::Periodic) {
      throw_athelas_error(
          "Implicit radiation does not support periodic boundaries.");
    }

    // handle Marshak data
    double rad_i_marshak_incoming_energy = 0.0;
    double rad_o_marshak_incoming_energy = 0.0;

    if (rad_i_type == bc::BcType::Marshak) {
      sol::optional<double> incoming =
          rad_bc_block ? sol::optional<double>(
                             (*rad_bc_block)["marshak_incoming_energy_i"])
                       : sol::nullopt;
      if (!incoming) {
        throw_athelas_error(
            " ! Initialization Error: Marshak radiation boundaries require "
            "marshak_incoming_energy_i.");
      }
      rad_i_marshak_incoming_energy = *incoming;
    }

    params_->add("radiation.bc.i.marshak_incoming_energy",
                 rad_i_marshak_incoming_energy);
    params_->add("radiation.bc.o.marshak_incoming_energy",
                 rad_o_marshak_incoming_energy);
  } // --- radiation block ---

  // -----------------------------------
  // ---------- gravity block ----------
  // -----------------------------------
  if (*grav) {
    sol::optional<sol::table> grav_block = config_["gravity"];
    sol::table gravity = *grav_block;

    const bool split_grav = gravity.get_or("operator_split", false);
    params_->add("physics.gravity.split", split_grav);
    const double gval = gravity.get_or("gval", 1.0);
    params_->add("gravity.gval", gval);
    const std::string gmodel = gravity.get_or<std::string>("model", "constant");
    params_->add("gravity.model", gmodel);
    if (gmodel == "constant" && gval <= 0.0) {
      throw_athelas_error(
          "Constant gravitational potential requested but g <= 0.0!");
    }
  } // gravity block

  // ---------------------------------
  // ---------- composition ----------
  // ---------------------------------
  sol::optional<sol::table> comp_block = config_["composition"];
  sol::optional<int> ncomps = (*comp_block)["ncomps"];
  if (!ncomps && *comps) {
    throw_athelas_error(
        "Composition enabled but no ncomps in composition block!");
  }
  params_->add("composition.ncomps", ncomps.value_or(0));

  // --------------------------------
  // ---------- ionization ----------
  // --------------------------------
  sol::optional<sol::table> ion_block = config_["ionization"];
  if (*ion) {
    sol::table ionization = *ion_block;
    sol::optional<std::string> fn_ion = ionization["fn_ionization"];
    sol::optional<std::string> fn_deg = ionization["fn_degeneracy"];
    sol::optional<int> saha_ncomps = ionization["ncomps"];
    sol::optional<std::string> saha_solver = ionization["solver"];

    if (!fn_ion || !fn_deg) {
      throw_athelas_error("With ionization enabled you must provide paths to "
                          "atomic data (fn_ionization and fn_degeneracy). "
                          "Defaults are in athelas/data/");
    }
    const std::string solver_lc = utilities::to_lower(*saha_solver);
    if (solver_lc != "linear" && solver_lc != "log") {
      throw_athelas_error(
          "[ionization.solver] must be either 'linear' or 'log'!");
    }
    params_->add("ionization.fn_ionization", *fn_ion);
    params_->add("ionization.fn_degeneracy", *fn_deg);
    params_->add("ionization.ncomps", *saha_ncomps);
    params_->add("ionization.solver", solver_lc);
  }

  // -----------------------------------
  // ---------- heating block ----------
  // -----------------------------------
  if (*heating) {
    sol::optional<sol::table> heat_block = config_["heating"];
    sol::optional<sol::table> nickel_block = (*heat_block)["nickel"];
    if (nickel_block) {
      sol::table nickel = *nickel_block;

      sol::optional<bool> nickel_enabled = nickel["enabled"];
      params_->add("physics.heating.nickel.enabled", *nickel_enabled);

      const bool split_nickel = nickel.get_or("operator_split", false);
      params_->add("physics.heating.nickel.split", split_nickel);

      sol::optional<std::string> nickel_model = nickel["model"];
      params_->add("heating.nickel.model", *nickel_model);
    } else {
      params_->add("physics.heating.nickel.enabled", false);
    }
  } else {
    params_->add("physics.heating.nickel.enabled", false);
  } // heating block

  // ----------------------------------
  // ---------- engine block ----------
  // ----------------------------------
  if (*engine) {
    sol::optional<sol::table> eng_block = config_["engine"];
    sol::optional<sol::table> thermal_block = (*eng_block)["thermal"];
    if (thermal_block) {
      sol::table thermal = *thermal_block;

      sol::optional<bool> thermal_engine_enabled = thermal["enabled"];
      params_->add("physics.engine.thermal.enabled", *thermal_engine_enabled);

      sol::optional<double> energy = thermal["energy"];
      params_->add("physics.engine.thermal.energy", *energy);

      // Energy injection mode: direct or asymptotic
      sol::optional<std::string> mode = thermal["mode"];
      const std::string mode_lc = utilities::to_lower(*mode);
      if (mode_lc != "direct" && mode_lc != "asymptotic") {
        throw_athelas_error(
            "[engine.thermal.mode] must be 'direct' or 'asymptotic'!");
      }
      params_->add("physics.engine.thermal.mode", mode_lc);

      // time
      sol::optional<double> tend = thermal["tend"];
      if (*tend <= 0.0) {
        throw_athelas_error("[engine.thermal.tend] must be > 0!");
      }
      params_->add("physics.engine.thermal.tend", *tend);

      // NOTE: currently forcing the start position of the thermal engine to be
      // the left domain.
      params_->add("physics.engine.thermal.mstart", 1); // first real cell

      sol::optional<double> mend = thermal["mend"];
      if (*mend <= 0.0) {
        throw_athelas_error("[engine.thermal.mend] must be > 0!");
      }
      params_->add("physics.engine.thermal.mend", *mend);

      // optional
      const bool split_te = thermal.get_or("operator_split", false);
      params_->add("physics.engine.thermal.split", split_te);
    } else {
      params_->add("physics.engine.thermal.enabled", false);
    }
  } else {
    params_->add("physics.engine.thermal.enabled", false);
  } // engine block

  // ----------------------------
  // ---------- output ----------
  // ----------------------------
  params_->add("output.dir", output_dir);
  sol::optional<sol::table> out_block = config_["output"];
  sol::table output = *out_block;

  const int ncycle_out = output.get_or("ncycle_out", 1);
  const double dt_hdf5 = output.get_or("dt_hdf5", tf.value_or(1.0) / 100.0);
  const double dt_growth_frac = output.get_or("dt_growth_frac", 1.05);
  const double dt_init = output.get_or("dt_init", 1.0e-16);

  sol::optional<sol::table> hist_block = output["history"];
  const bool history_enabled = hist_block.has_value();
  const std::string hist_fn =
      hist_block ? hist_block->get_or<std::string>("fn", "athelas.hst")
                 : "athelas.hst";
  const double hist_dt =
      hist_block ? hist_block->get_or("dt", dt_hdf5 / 10.0) : dt_hdf5 / 10.0;

  sol::optional<double> fixed_dt = output["dt_fixed"];

  if (dt_init <= 0.0) {
    throw_athelas_error("dt_init must be strictly > 0.0\n");
  }
  if (dt_growth_frac <= 1.0) {
    throw_athelas_error("dt_growth_frac must be strictly > 1.0\n");
  }
  if (dt_hdf5 <= 0.0) {
    throw_athelas_error("dt_hdf5 must be strictly > 0.0\n");
  }
  if (hist_dt <= 0.0) {
    throw_athelas_error("hist_dt must be strictly > 0.0\n");
  }
  params_->add("output.ncycle_out", ncycle_out);
  params_->add("output.dt_hdf5", dt_hdf5);
  params_->add("output.dt_growth_frac", dt_growth_frac);
  params_->add("output.dt_init", dt_init);
  params_->add("output.history_enabled", history_enabled);
  params_->add("output.hist_fn", hist_fn);
  params_->add("output.hist_dt", hist_dt);
  if (fixed_dt) {
    params_->add("output.dt_fixed", *fixed_dt);
  }

  // --------------------------
  // ---------- time ----------
  // --------------------------
  sol::optional<sol::table> time_block = config_["time"];
  sol::optional<std::string> integrator = (*time_block)["integrator"];
  const MethodID method_id = string_to_id(utilities::to_lower(*integrator));
  params_->add("time.integrator", method_id);
  params_->add("time.integrator_string", *integrator); // for IO

  // -------------------------
  // ---------- eos ----------
  // -------------------------
  sol::optional<sol::table> eos_block = config_["eos"];
  sol::table eos = *eos_block;

  sol::optional<std::string> eos_type = eos["type"];
  params_->add("eos.type", *eos_type);
  params_->add("eos.gamma", eos.get_or("gamma", 1.4));
  if (*eos_type == "polytropic") {
    sol::optional<double> eos_k = eos["k"];
    sol::optional<double> eos_n = eos["n"];
    if (!eos_k || !eos_n) {
      throw_athelas_error(
          "Polytropic EOS requires 'k' and 'n' in [eos] block!");
    }
    params_->add("eos.k", *eos_k);
    params_->add("eos.n", *eos_n);
  }

  // --------------------------
  // ---------- opac ----------
  // --------------------------
  if (*rad) {
    sol::optional<sol::table> opac_block = config_["opacity"];
    sol::table opacity = *opac_block;

    sol::optional<std::string> opac_type = opacity["type"];
    params_->add("opacity.type", *opac_type);

    if (*opac_type == "tabular") {
      sol::optional<std::string> fn = opacity["filename"];
      if (!fn) {
        throw_athelas_error(
            "Tabular opacity requires 'filename' in [opacity] block!");
      }
      params_->add("opacity.filename", *fn);
    } else if (*opac_type == "constant") {
      sol::optional<double> kr = opacity["kR"];
      sol::optional<double> kp = opacity["kP"];
      if (!kr || !kp) {
        throw_athelas_error(
            "Constant opacity must specify mean opacities kR and kP!");
      }
      params_->add("opacity.kR", *kr);
      params_->add("opacity.kP", *kp);
    } else if (*opac_type == "powerlaw") {
      sol::optional<double> kr = opacity["kR"];
      sol::optional<double> kp = opacity["kP"];
      sol::optional<double> rho_exp = opacity["rho_exp"];
      sol::optional<double> t_exp = opacity["t_exp"];
      sol::optional<double> kp_offset = opacity["kP_offset"];
      sol::optional<double> kr_offset = opacity["kR_offset"];
      if (!kr || !kp || !rho_exp || !t_exp) {
        throw_athelas_error("Powerlaw opacity must specify kR, kP, rho_exp, "
                            "and t_exp!");
      }
      params_->add("opacity.kR", *kr);
      params_->add("opacity.kP", *kp);
      params_->add("opacity.rho_exp", *rho_exp);
      params_->add("opacity.t_exp", *t_exp);
      params_->add("opacity.kR_offset", kr_offset.value_or(0.0));
      params_->add("opacity.kP_offset", kp_offset.value_or(0.0));
    }

    // floors
    sol::optional<sol::table> floors_block = opacity["floors"];
    sol::table floors = floors_block.value_or(lua_.create_table());

    const std::string floor_type =
        floors.get_or<std::string>("type", "core_envelope");
    params_->add("opacity.floors.type", floor_type);
    if (floor_type != "core_envelope" && floor_type != "constant") {
      throw_athelas_error(
          "[opacity.floors.type] must be 'core_envelope' or 'constant'!");
    }
    if (floor_type == "core_envelope") {
      const double core_planck = floors.get_or("core_planck", 0.24);
      const double core_rosseland = floors.get_or("core_rosseland", 0.24);
      const double env_planck = floors.get_or("env_planck", 0.01);
      const double env_rosseland = floors.get_or("env_rosseland", 0.01);
      if ((core_planck < env_planck) || (core_rosseland < env_rosseland)) {
        throw_athelas_error("In the `core_envelope` floor model the core floor "
                            "must be higher than the envelope floor!");
      }
      params_->add("opacity.floors.core_planck", core_planck);
      params_->add("opacity.floors.core_rosseland", core_rosseland);
      params_->add("opacity.floors.env_planck", env_planck);
      params_->add("opacity.floors.env_rosseland", env_rosseland);
    }
    if (floor_type == "constant") {
      const double planck = floors.get_or("planck", 1.0e-3);
      const double rosseland = floors.get_or("rosseland", 1.0e-3);
      params_->add("opacity.floors.planck", planck);
      params_->add("opacity.floors.rosseland", rosseland);
    }
  } // opacity block

  std::println("# Configuration ... Complete\n");
}

// ---------------------------------------------------------------------------
// Restart construction
// ---------------------------------------------------------------------------

ProblemIn::ProblemIn(
    RestartTag /*tag*/, const std::string &h5_filename,
    const std::string &output_dir,
    const std::vector<std::pair<std::string, std::string>> &overrides) {
  // sol::state is still needed to parse --key=expr overrides; no input deck
  // is loaded in restart mode.
  lua_.open_libraries(sol::lib::base, sol::lib::math);
  params_ = std::make_unique<Params>();

  std::println("# Loading Restart File ... {}", h5_filename);

  io::RestartReader reader(h5_filename);
  io::load_params_from_h5(*params_, reader);

  // Always honor the CLI output dir (allows redirecting restart output).
  if (params_->contains("output.dir")) {
    params_->get_mutable_ref<std::string>("output.dir") = output_dir;
  } else {
    params_->add("output.dir", output_dir);
  }

  // MethodID isn't serialized directly — recover it from the string form.
  const auto integrator_name =
      params_->get<std::string>("time.integrator_string");
  const MethodID method_id = string_to_id(utilities::to_lower(integrator_name));
  if (params_->contains("time.integrator")) {
    params_->get_mutable_ref<MethodID>("time.integrator") = method_id;
  } else {
    params_->add("time.integrator", method_id);
  }

  for (const auto &[key, expr] : overrides) {
    apply_restart_override(key, expr);
    std::println("# Override applied: {} = {}", key, expr);
  }

  std::println("# Restart load ... Complete\n");
}

void ProblemIn::apply_restart_override(const std::string &key,
                                       const std::string &expr) {
  sol::object value;
  try {
    value = lua_.script("return " + expr);
  } catch (const sol::error &e) {
    std::println(std::cerr, "Override failed for '{} = {}': {}", key, expr,
                 e.what());
    std::println(std::cerr,
                 "Hint: --key=value is parsed as Lua source; strings must be "
                 "Lua-quoted, e.g. --key='\"value\"'.");
    throw_athelas_error("Invalid command-line override");
  }

  // If the key already exists, preserve its type so consumers see what they
  // expect. Otherwise, infer from the Lua value.
  const bool exists = params_->contains(key);
  const std::type_index existing =
      exists ? params_->get_type(key) : std::type_index(typeid(void));

  auto store = [&](auto v) {
    using T = decltype(v);
    if (exists) {
      params_->get_mutable_ref<T>(key) = std::move(v);
    } else {
      params_->add(key, std::move(v));
    }
  };

  if (exists && existing == typeid(bool)) {
    store(value.as<bool>());
  } else if (exists && existing == typeid(int)) {
    store(value.as<int>());
  } else if (exists && existing == typeid(double)) {
    store(value.as<double>());
  } else if (exists && existing == typeid(std::string)) {
    store(value.as<std::string>());
  } else if (!exists) {
    // No existing type — fall back to Lua's notion. Order matters: bool first
    // (Lua bools also satisfy is<int>() in some sol setups), then int before
    // double (sol distinguishes them).
    if (value.is<bool>()) {
      params_->add(key, value.as<bool>());
    } else if (value.is<int>()) {
      params_->add(key, value.as<int>());
    } else if (value.is<double>()) {
      params_->add(key, value.as<double>());
    } else if (value.is<std::string>()) {
      params_->add(key, value.as<std::string>());
    } else {
      throw_athelas_error("Restart override '" + key +
                          "' has unsupported value type");
    }
  } else {
    throw_athelas_error("Restart override for '" + key +
                        "' targets a param type that is not overridable from "
                        "the CLI");
  }
}

} // namespace athelas
