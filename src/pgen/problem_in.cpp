#include "pgen/problem_in.hpp"
#include "timestepper/tableau.hpp"
#include "utils/error.hpp"
#include "utils/utilities.hpp"

namespace athelas {

// Provide access to the underlying params object
auto ProblemIn::param() -> Params * { return params_.get(); }

[[nodiscard]] auto ProblemIn::param() const -> Params * {
  return params_.get();
}

ProblemIn::ProblemIn(const std::string &fn, const std::string &output_dir) {
  // toml++ wants a string_view
  std::string_view const fn_in{fn};

  // ------------------------------
  // ---------- Load ini ----------
  // ------------------------------
  try {
    config_ = toml::parse_file(fn_in);
  } catch (const toml::parse_error &err) {
    std::cerr << err << "\n";
    THROW_ATHELAS_ERROR("Issue reading input deck!");
  }
  params_ = std::make_unique<Params>();

  std::println("# Loading Input Deck ...");

  // -----------------------------------
  // ---------- problem block ----------
  // -----------------------------------
  if (!config_["problem"].is_table()) {
    THROW_ATHELAS_ERROR("Input deck must have a [problem] block!");
  }

  std::optional<std::string> pname =
      config_["problem"]["problem"].value<std::string>();
  if (!pname.has_value()) {
    THROW_ATHELAS_ERROR("Missing or invalid 'problem' in [problem] block.");
  }
  params_->add("problem.problem", pname.value());

  std::optional<bool> restart = config_["problem"]["restart"].value_or(false);
  params_->add("problem.restart", restart.value_or(false));

  std::optional<double> tf = config_["problem"]["t_end"].value<double>();
  if (!tf.has_value()) {
    THROW_ATHELAS_ERROR("Missing or invalid 'tf' in [problem] block.");
  }
  if (tf.value() <= 0.0) {
    THROW_ATHELAS_ERROR("tf must be > 0.0!");
  }
  params_->add("problem.tf", tf.value());

  const double nlim = config_["problem"]["nlim"].value_or(-1);
  params_->add("problem.nlim", nlim);

  std::optional<double> xl = config_["problem"]["xl"].value<double>();
  if (!xl.has_value()) {
    THROW_ATHELAS_ERROR("Missing or invalid 'xl' in [problem] block.");
  }
  params_->add("problem.xl", xl.value());

  std::optional<double> xr = config_["problem"]["xr"].value<double>();
  if (!xr.has_value()) {
    THROW_ATHELAS_ERROR("Missing or invalid 'xr' in [problem] block.");
  }
  if (xr.value() <= xl.value()) {
    THROW_ATHELAS_ERROR("xr must be > xl!");
  }
  params_->add("problem.xr", xr.value());

  std::optional<int> nx = config_["problem"]["nx"].value<int>();
  if (!nx.has_value()) {
    THROW_ATHELAS_ERROR("Missing nx in [problem] block!");
  }
  if (nx.value() <= 0) {
    THROW_ATHELAS_ERROR("nx must be > 0!");
  }
  params_->add("problem.nx", nx.value());

  std::optional<double> cfl = config_["problem"]["cfl"].value<double>();
  // NOTE: It may be worthwhile to have cfl be registerd per physics.
  if (!cfl.has_value()) {
    THROW_ATHELAS_ERROR("Missing or invalid 'cfl' in [problem] block.");
  }
  if (cfl.value() <= 0.0) {
    THROW_ATHELAS_ERROR("cfl must be > 0.0!");
  }
  params_->add("problem.cfl", cfl.value());

  std::optional<std::string> geom =
      config_["problem"]["geometry"].value<std::string>();
  if (!geom.has_value()) {
    THROW_ATHELAS_ERROR("Missing or invalid 'geom' in [problem] block.");
  }
  params_->add("problem.geometry", geom.value());
  if (geom.value() == "planar") {
    params_->add("problem.geometry_model", Geometry::Planar);
  }
  if (geom.value() == "spherical") {
    params_->add("problem.geometry_model", Geometry::Spherical);
  }
  // TODO(astrobarker): move grid stuff into own section, not problem
  // Begs the question: what is "problem" and what is "grid"? xl and xr?
  std::optional<std::string> grid_type =
      config_["problem"]["grid_type"].value<std::string>();
  if (!grid_type.has_value()) {
    THROW_ATHELAS_ERROR("Missing or invalid 'grid_type' in [problem] block.");
  }
  params_->add("problem.grid_type", grid_type.value());

  // ---------------------------------------------
  // ---------- handle [problem.params] ----------
  // ---------------------------------------------
  if (!config_["problem"]["params"].is_table()) {
    THROW_ATHELAS_ERROR("No [params] block in [problem]!");
  }

  auto *pparams = config_["problem"]["params"].as_table();
  for (auto &&[key, node] : *pparams) {
    std::string out_key = "problem.params." + std::string(key);

    using toml::node_type;
    switch (node.type()) {
    case node_type::integer:
      if (auto val = node.value<int64_t>()) {
        params_->add(out_key, static_cast<int>(*val)); // or int64_t
      }
      break;

    case node_type::floating_point:
      if (auto val = node.value<double>()) {
        params_->add(out_key, *val);
      }
      break;

    case node_type::boolean:
      if (auto val = node.value<bool>()) {
        params_->add(out_key, *val);
      }
      break;

    case node_type::string:
      if (auto val = node.value<std::string>()) {
        params_->add(out_key, *val);
      }
      break;

    default:
      THROW_ATHELAS_ERROR("Unsupported TOML type for key: " + std::string(key));
    }
  }

  // -----------------------------------
  // ---------- physics block ----------
  // -----------------------------------
  if (!config_["physics"].is_table()) {
    THROW_ATHELAS_ERROR("Input deck must have a [physics] block!");
  }
  std::optional<bool> rad = config_["physics"]["radiation"].value<bool>();
  if (!rad) {
    THROW_ATHELAS_ERROR("Missing or invalid 'radiation' in [physics] block.");
  }
  params_->add("physics.rad_active", rad.value());

  std::optional<bool> grav = config_["physics"]["gravity"].value<bool>();
  if (!grav) {
    THROW_ATHELAS_ERROR("Missing or invalid 'gravity' in [physics] block.");
  }
  params_->add("physics.gravity_active", grav.value());

  std::optional<bool> comps = config_["physics"]["composition"].value<bool>();
  if (!comps) {
    THROW_ATHELAS_ERROR("Missing or invalid 'composition' in [physics] block.");
  }
  params_->add("physics.composition_enabled", comps.value());

  std::optional<bool> ion = config_["physics"]["ionization"].value<bool>();
  if (!ion) {
    THROW_ATHELAS_ERROR("Missing or invalid 'ionization' in [physics] block.");
  }
  params_->add("physics.ionization_enabled", ion.value());

  std::optional<bool> engine = config_["physics"]["engine"].value<bool>();
  if (!engine) {
    THROW_ATHELAS_ERROR("Missing or invalid 'engine' in [physics] block.");
  }
  params_->add("physics.engine.enabled", engine.value());

  if (ion.value() && !comps.value()) {
    THROW_ATHELAS_ERROR("Ionization enabled but composition disabled!");
  }

  std::optional<bool> heating = config_["physics"]["heating"].value<bool>();
  if (!heating) {
    THROW_ATHELAS_ERROR("Missing or invalid 'heating' in [physics] block.");
  }
  params_->add("physics.heating.active", heating.value());

  // ---------------------------------
  // ---------- fluid block ----------
  // ---------------------------------
  if (!config_["fluid"].is_table()) {
    THROW_ATHELAS_ERROR("[fluid] block must be provided!");
  }

  if (config_["fluid"]["operator_split"].is_value()) {
    THROW_ATHELAS_ERROR("Operator split not supported for fluid! Remove option "
                        "from [fluid] block.");
  }

  std::optional<int> porder = config_["fluid"]["porder"].value<int>();
  if (!porder) {
    THROW_ATHELAS_ERROR("fluid enabled but 'porder' missing in [fluid] block!");
  }
  params_->add("fluid.porder", porder.value());

  std::optional<int> nnodes = config_["fluid"]["nnodes"].value<int>();
  if (!nnodes) {
    THROW_ATHELAS_ERROR("fluid enabled but 'nnodes' missing in [fluid] block!");
  }
  params_->add("fluid.nnodes", nnodes.value());

  if (!config_["fluid"]["limiter"].is_table()) {
    WARNING_ATHELAS("No [limiter] block in [fluid] - defaulting to minmod with "
                    "standard values!");
  }

  std::optional<bool> limit_fluid =
      config_["fluid"]["limiter"]["do_limiter"].value_or(true);
  params_->add("fluid.limiter.enabled", limit_fluid.value());

  std::optional<std::string> fluid_limiter =
      config_["fluid"]["limiter"]["type"].value_or("minmod");
  params_->add("fluid.limiter.type", fluid_limiter.value());

  if (limit_fluid.value() && fluid_limiter.value() == "minmod") {
    const double b_tvd = config_["fluid"]["limiter"]["b_tvd"].value_or(1.0);
    params_->add("fluid.limiter.b_tvd", b_tvd);
    const double m_tvb = config_["fluid"]["limiter"]["m_tvb"].value_or(0.0);
    params_->add("fluid.limiter.m_tvb", m_tvb);
  }
  if (limit_fluid.value() && fluid_limiter.value() == "weno") {
    std::optional<double> gamma_i =
        config_["fluid"]["limiter"]["gamma_i"].value<double>();
    std::optional<double> gamma_l =
        config_["fluid"]["limiter"]["gamma_l"].value<double>();
    std::optional<double> gamma_r =
        config_["fluid"]["limiter"]["gamma_r"].value<double>();
    if ((gamma_i && !gamma_l) || (gamma_i && !gamma_r)) {
      params_->add("fluid.limiter.gamma_i", gamma_i.value());
      params_->add("fluid.limiter.gamma_l", (1.0 - gamma_i.value()) / 2.0);
      params_->add("fluid.limiter.gamma_r", (1.0 - gamma_i.value()) / 2.0);
    } else if (gamma_i && gamma_r && gamma_l) {
      params_->add("fluid.limiter.gamma_i", gamma_i.value());
      params_->add("fluid.limiter.gamma_r", gamma_r.value());
      params_->add("fluid.limiter.gamma_l", gamma_l.value());
    } else {
      THROW_ATHELAS_ERROR("Error parsing weno gammas in [fluid] block: provide "
                          "only gamma_i, or all gamma_l, gamma_i, gamma_r!");
    }
    const double sum_g = params_->get<double>("fluid.limiter.gamma_i") +
                         params_->get<double>("fluid.limiter.gamma_l") +
                         params_->get<double>("fluid.limiter.gamma_r");
    if (std::abs(sum_g - 1.0) > 1.0e-10) {
      THROW_ATHELAS_ERROR(
          " ! Initialization Error: Linear WENO weights must sum to unity.");
    }
    const double weno_r = config_["fluid"]["limiter"]["weno_r"].value_or(2.0);
    if (weno_r <= 0.0) {
      THROW_ATHELAS_ERROR(
          "[fluid] block: WENO limiter weno_r must be positive!");
    }
    params_->add("fluid.limiter.weno_r", weno_r);
  }

  // tci
  const bool do_tci = config_["fluid"]["limiter"]["tci_opt"].value_or(false);
  params_->add("fluid.limiter.tci_enabled", do_tci);
  if (do_tci) {
    std::optional<double> tci_val =
        config_["fluid"]["limiter"]["tci_val"].value<double>();
    if (!tci_val.has_value()) {
      THROW_ATHELAS_ERROR(
          "[fluid] block: TCI requested but no tci_val provided!");
    }
    params_->add("fluid.limiter.tci_val", tci_val.value());
  } else {
    params_->add("fluid.limiter.tci_val", 0.0);
  }

  // characteristic limiting
  const bool characteristic =
      config_["fluid"]["limiter"]["characteristic"].value_or(false);
  params_->add("fluid.limiter.characteristic", characteristic);

  // fluid bc
  std::optional<std::string> fluid_bc_i =
      config_["bc"]["fluid"]["bc_i"].value<std::string>();
  std::optional<std::string> fluid_bc_o =
      config_["bc"]["fluid"]["bc_o"].value<std::string>();

  // --- fluid bc ---
  if (fluid_bc_i.has_value()) {
    params_->add("fluid.bc.i", utilities::to_lower(fluid_bc_i.value()));
  } else {
    THROW_ATHELAS_ERROR("Inner fluid boundary condition not supplied "
                        "in input deck.");
  }
  if (fluid_bc_o.has_value()) {
    params_->add("fluid.bc.o", utilities::to_lower(fluid_bc_o.value()));
  } else {
    THROW_ATHELAS_ERROR("Outer fluid boundary condition not supplied "
                        "in input deck.");
  }
  check_bc(params_->get<std::string>("fluid.bc.i"));
  check_bc(params_->get<std::string>("fluid.bc.i"));

  // handle dirichlet..
  std::array<double, 3> fluid_i_dirichlet_values = {0.0, 0.0, 0.0};
  std::array<double, 3> fluid_o_dirichlet_values = {0.0, 0.0, 0.0};

  if (fluid_bc_i == "dirichlet") {
    const auto &node = config_["bc"]["fluid"]["dirichlet_values_i"];
    if (node && node.is_array()) {
      const auto *array = node.as_array();
      read_toml_array(array, fluid_i_dirichlet_values);
    } else {
      THROW_ATHELAS_ERROR(" ! Initialization Error: Failed to read fluid "
                          "dirichlet_values_i as array.");
    }
  }

  if (fluid_bc_o == "dirichlet") {
    const auto &node = config_["bc"]["fluid"]["dirichlet_values_o"];
    if (node && node.is_array()) {
      const auto *array = node.as_array();
      read_toml_array(array, fluid_o_dirichlet_values);
    } else {
      THROW_ATHELAS_ERROR(" ! Initialization Error: Failed to read fluid "
                          "dirichlet_values_o as array.");
    }
  }
  params_->add("fluid.bc.i.dirichlet_values", fluid_i_dirichlet_values);
  params_->add("fluid.bc.o.dirichlet_values", fluid_o_dirichlet_values);
  // ---fluid block ---

  // -------------------------------------
  // ---------- radiation block ----------
  // -------------------------------------
  // I suspect much of this should really go into
  // the individual packages.
  if (rad.value()) {
    if (!config_["radiation"].is_table()) {
      THROW_ATHELAS_ERROR(
          "Radiation is active but radiation block is missing!");
    }

    if (config_["radiation"]["operator_split"].is_value()) {
      THROW_ATHELAS_ERROR("Operator split not supported for radiation! Remove "
                          "option from [radiation] block.");
    }

    std::optional<int> porder = config_["radiation"]["porder"].value<int>();
    if (!porder) {
      THROW_ATHELAS_ERROR(
          "radiation enabled but 'porder' missing in [radiation] block!");
    }
    params_->add("radiation.porder", porder.value());

    std::optional<int> nnodes = config_["radiation"]["nnodes"].value<int>();
    if (!nnodes) {
      THROW_ATHELAS_ERROR(
          "radiation enabled but 'nnodes' missing in [radiation] block!");
    }
    params_->add("radiation.nnodes", nnodes.value());

    if (!config_["radiation"]["limiter"].is_table()) {
      WARNING_ATHELAS("No [limiter] block in [radiation] - defaulting to "
                      "minmod with standard values!");
    }

    std::optional<bool> limit_rad =
        config_["radiation"]["limiter"]["do_limiter"].value_or(true);
    params_->add("radiation.limiter.enabled", limit_rad.value());

    std::optional<std::string> rad_limiter =
        config_["radiation"]["limiter"]["type"].value_or("minmod");
    params_->add("radiation.limiter.type", rad_limiter.value());

    if (limit_rad.value() && rad_limiter.value() == "minmod") {
      const double b_tvd =
          config_["radiation"]["limiter"]["b_tvd"].value_or(1.0);
      params_->add("radiation.limiter.b_tvd", b_tvd);
      const double m_tvb =
          config_["radiation"]["limiter"]["m_tvb"].value_or(0.0);
      params_->add("radiation.limiter.m_tvb", m_tvb);
    }
    if (limit_rad.value() && rad_limiter.value() == "weno") {
      std::optional<double> gamma_i =
          config_["radiation"]["limiter"]["gamma_i"].value<double>();
      std::optional<double> gamma_l =
          config_["radiation"]["limiter"]["gamma_l"].value<double>();
      std::optional<double> gamma_r =
          config_["radiation"]["limiter"]["gamma_r"].value<double>();
      if ((gamma_i && !gamma_l) || (gamma_i && !gamma_r)) {
        params_->add("radiation.limiter.gamma_i", gamma_i.value());
        params_->add("radiation.limiter.gamma_l",
                     (1.0 - gamma_i.value()) / 2.0);
        params_->add("radiation.limiter.gamma_r",
                     (1.0 - gamma_i.value()) / 2.0);
      } else if (gamma_i && gamma_r && gamma_l) {
        params_->add("radiation.limiter.gamma_i", gamma_i.value());
        params_->add("radiation.limiter.gamma_r", gamma_r.value());
        params_->add("radiation.limiter.gamma_l", gamma_l.value());
      } else {
        THROW_ATHELAS_ERROR(
            "Error parsing weno gammas in [radiation] block: provide only "
            "gamma_i, or all gamma_l, gamma_i, gamma_r!");
      }
      const double sum_g = params_->get<double>("radiation.limiter.gamma_i") +
                           params_->get<double>("radiation.limiter.gamma_l") +
                           params_->get<double>("radiation.limiter.gamma_r");
      if (std::abs(sum_g - 1.0) > 1.0e-10) {
        THROW_ATHELAS_ERROR(
            " ! Initialization Error: Linear WENO weights must sum to unity.");
      }
      const double weno_r =
          config_["radiation"]["limiter"]["weno_r"].value_or(2.0);
      if (weno_r <= 0.0) {
        THROW_ATHELAS_ERROR(
            "[radiation] block: WENO limiter weno_r must be positive!");
      }
      params_->add("radiation.limiter.weno_r", weno_r);
    }

    // tci
    const bool do_tci =
        config_["radiation"]["limiter"]["tci_opt"].value_or(false);
    params_->add("radiation.limiter.tci_enabled", do_tci);
    if (do_tci) {
      std::optional<double> tci_val =
          config_["radiation"]["limiter"]["tci_val"].value<double>();
      if (!tci_val.has_value()) {
        THROW_ATHELAS_ERROR(
            "[radiation] block: TCI requested but no tci_val provided!");
      }
      params_->add("radiation.limiter.tci_val", tci_val.value());
    } else {
      params_->add("radiation.limiter.tci_val", 0.0);
    }

    // characteristic limiting
    const bool characteristic =
        config_["radiation"]["limiter"]["characteristic"].value_or(false);
    params_->add("radiation.limiter.characteristic", characteristic);

    // --- radiation bc ---
    std::optional<std::string> rad_bc_i =
        config_["bc"]["radiation"]["bc_i"].value<std::string>();
    std::optional<std::string> rad_bc_o =
        config_["bc"]["radiation"]["bc_o"].value<std::string>();

    if (rad_bc_i.has_value()) {
      params_->add("radiation.bc.i", utilities::to_lower(rad_bc_i.value()));
    } else {
      THROW_ATHELAS_ERROR("Inner radiation boundary condition not supplied "
                          "in input deck but radiation is enabled1");
    }
    if (rad_bc_o.has_value()) {
      params_->add("radiation.bc.o", utilities::to_lower(rad_bc_o.value()));
    } else {
      THROW_ATHELAS_ERROR("Outer radiation boundary condition not supplied "
                          "in input deck but radiation is enabled!");
    }
    check_bc(params_->get<std::string>("radiation.bc.i"));
    check_bc(params_->get<std::string>("radiation.bc.i"));

    // handle dirichlet..
    std::array<double, 2> rad_i_dirichlet_values = {0.0, 0.0};
    std::array<double, 2> rad_o_dirichlet_values = {0.0, 0.0};

    if (rad_bc_i == "dirichlet" || rad_bc_i == "marshak") {
      const auto &node = config_["bc"]["radiation"]["dirichlet_values_i"];
      if (node && node.is_array()) {
        const auto *array = node.as_array();
        read_toml_array(array, rad_i_dirichlet_values);
      } else {
        THROW_ATHELAS_ERROR(" ! Initialization Error: Failed to read radiation "
                            "dirichlet_values_i as array.");
      }
    }

    if (rad_bc_o == "dirichlet") {
      const auto &node = config_["bc"]["radiation"]["dirichlet_values_o"];
      if (node && node.is_array()) {
        const auto *array = node.as_array();
        read_toml_array(array, rad_o_dirichlet_values);
      } else {
        THROW_ATHELAS_ERROR(" ! Initialization Error: Failed to read radiation "
                            "dirichlet_values_o as array.");
      }
    }
    params_->add("radiation.bc.i.dirichlet_values", rad_i_dirichlet_values);
    params_->add("radiatio.bc.o.dirichlet_values", rad_o_dirichlet_values);
  } // --- radiation block ---

  // -----------------------------------
  // ---------- gravity block ----------
  // -----------------------------------
  if (grav.value()) {
    if (!config_["gravity"].is_table()) {
      THROW_ATHELAS_ERROR(
          "Gravity is enabled but no [gravity] block exists in input deck!");
    }
    bool split_grav = config_["gravity"]["operator_split"].value_or(false);
    params_->add("physics.gravity.split", split_grav);
    const double gval = config_["gravity"]["gval"].value_or(1.0);
    params_->add("gravity.gval", gval); // Always present
    const std::string gmodel = config_["gravity"]["model"].value_or("constant");
    params_->add("gravity.modelstring", gmodel);
    params_->add("gravity.model", (utilities::to_lower(gmodel) == "spherical")
                                      ? GravityModel::Spherical
                                      : GravityModel::Constant);
    if (params_->get<GravityModel>("gravity.model") == GravityModel::Constant &&
        gval <= 0.0) {
      THROW_ATHELAS_ERROR(
          "Constant gravitational potential requested but g <= 0.0!");
    }
  } // gravity block

  // ---------------------------------
  // ---------- composition ----------
  // ---------------------------------
  if (!config_["composition"].is_table() && comps.value()) {
    THROW_ATHELAS_ERROR(
        "Composition enabled but no [composition] block provided!");
  }
  std::optional<int> ncomps = config_["composition"]["ncomps"].value<int>();
  if (!ncomps && comps.value()) {
    THROW_ATHELAS_ERROR(
        "Composition enabled but no ncomps in composition block!");
  } else {
    params_->add("composition.ncomps", ncomps.value_or(0));
  }

  // --------------------------------
  // ---------- ionization ----------
  // --------------------------------
  if (!config_["ionization"].is_table() && ion.value()) {
    THROW_ATHELAS_ERROR(
        "Ionization enabled but no [ionization] block provided!");
  }
  std::optional<std::string> fn_ion =
      config_["ionization"]["fn_ionization"].value<std::string>();
  std::optional<std::string> fn_deg =
      config_["ionization"]["fn_degeneracy"].value<std::string>();
  std::optional<int> saha_ncomps = config_["ionization"]["ncomps"].value<int>();
  if ((!fn_ion || !fn_deg) && ion.value()) {
    THROW_ATHELAS_ERROR("With ionization enabled you must provide paths to "
                        "atomic data (fn_ionization and fn_degeneracy). "
                        "Defaults are in athelas/data/");
  }
  if ((!saha_ncomps) && ion.value()) {
    THROW_ATHELAS_ERROR(
        "Ionization enabled but ncomps not present in [ionization] block!");
  }
  if (ion.value()) {
    params_->add("ionization.fn_ionization", fn_ion.value());
    params_->add("ionization.fn_degeneracy", fn_deg.value());
    params_->add("ionization.ncomps", saha_ncomps.value());
  }

  // -----------------------------------
  // ---------- heating block ----------
  // -----------------------------------
  if (heating.value()) {
    if (!config_["heating"].is_table()) {
      THROW_ATHELAS_ERROR(
          "Heating is enabled but no [heating] block exists in input deck!");
    }
    if (config_["heating"]["nickel"].is_table()) {
      std::optional<bool> nickel_enabled =
          config_["heating"]["nickel"]["enabled"].value<bool>();
      if (!nickel_enabled) {
        THROW_ATHELAS_ERROR(
            "[heating.nickel] is requested but 'enabled' is missing!");
      }
      params_->add("physics.heating.nickel.enabled", nickel_enabled.value());

      bool split_nickel =
          config_["heating"]["nickel"]["operator_split"].value_or(false);
      params_->add("physics.heating.nickel.split", split_nickel);

      std::optional<std::string> nickel_model =
          config_["heating"]["nickel"]["model"].value<std::string>();
      if (!nickel_model) {
        THROW_ATHELAS_ERROR(
            "[heating.nickel] is requested but 'model' is missing!");
      }
      params_->add("heating.nickel.model", nickel_model.value());
    } else {
      params_->add("physics.heating.nickel.enabled", false);
    }
  } else {
    params_->add("physics.heating.nickel.enabled", false);
  } // heating block

  // ----------------------------------
  // ---------- engine block ----------
  // ----------------------------------
  if (engine.value()) {
    if (!config_["engine"].is_table()) {
      THROW_ATHELAS_ERROR(
          "An engine is enabled but no [engine] block exists in input deck!");
    }
    if (config_["engine"]["thermal"].is_table()) {
      std::optional<bool> thermal_engine_enabled =
          config_["engine"]["thermal"]["enabled"].value<bool>();
      if (!thermal_engine_enabled) {
        THROW_ATHELAS_ERROR(
            "[engine.thermal] is requested but 'enabled' is missing!");
      }
      params_->add("physics.engine.thermal.enabled",
                   thermal_engine_enabled.value());

      // Get the required thermal engine options
      std::optional<double> energy =
          config_["engine"]["thermal"]["energy"].value<double>();
      if (!energy) {
        THROW_ATHELAS_ERROR(
            "[engine.thermal] is requested but 'energy' is missing!");
      }
      params_->add("physics.engine.thermal.energy", energy.value());

      // Energy injection mode: direct or asymptotic
      auto mode = config_["engine"]["thermal"]["mode"].value<std::string>();
      if (!mode) {
        THROW_ATHELAS_ERROR(
            "[engine.thermal] is requested but 'mode' is missing!");
      }
      if (utilities::to_lower(mode.value()) != "direct" &&
          utilities::to_lower(mode.value()) != "asymptotic") {
        THROW_ATHELAS_ERROR(
            "[engine.thermal.mode] must be 'direct' or 'asymptotic'!");
      }
      params_->add("physics.engine.thermal.mode",
                   utilities::to_lower(mode.value()));

      // time
      auto tend = config_["engine"]["thermal"]["tend"].value<double>();
      if (!tend) {
        THROW_ATHELAS_ERROR(
            "[engine.thermal] is requested but 'tend' is missing!");
      }
      if (tend <= 0.0) {
        THROW_ATHELAS_ERROR("[engine.thermal.tend] must be > 0!");
      }
      params_->add("physics.engine.thermal.tend", tend.value());

      // NOTE: currently forcing the start position of the thermal engine to be
      // the left domain.
      params_->add("physics.engine.thermal.mstart", 1); // first real cell

      auto mend = config_["engine"]["thermal"]["mend"].value<double>();
      if (!mend) {
        THROW_ATHELAS_ERROR(
            "[engine.thermal] is requested but 'mend' is missing!");
      }
      if (mend <= 0.0) {
        THROW_ATHELAS_ERROR("[engine.thermal.mend] must be > 0!");
      }
      params_->add("physics.engine.thermal.mend", mend.value());

      // optional
      bool split_te =
          config_["engine"]["thermal"]["operator_split"].value_or(false);
      params_->add("physics.engine.thermal.split", split_te);
    } else {
      params_->add("physics.engine.thermal.enabled", false);
    }
  } else {
    params_->add("physics.engine.thermal.enabled", false);
  } // heating block

  // ----------------------------
  // ---------- output ----------
  // ----------------------------
  params_->add("output.dir", output_dir);
  // In principle everything below can be defaulted, but
  // I still require the block present.
  if (!config_["output"].is_table()) {
    THROW_ATHELAS_ERROR("No [output] block provided!");
  }
  const int ncycle_out = config_["output"]["ncycle_out"].value_or(1);
  const double dt_hdf5 =
      config_["output"]["dt_hdf5"].value_or(tf.value() / 100.0);
  const double dt_init_frac = config_["output"]["dt_init_frac"].value_or(1.05);
  const double initial_dt = config_["output"]["initial_dt"].value_or(1.0e-16);
  const bool history_enabled = config_["output"]["history"].is_table();
  const std::string hist_fn =
      config_["output"]["history"]["fn"].value_or("athelas.hst");
  const double hist_dt =
      config_["output"]["history"]["dt"].value_or(dt_hdf5 / 10);
  if (initial_dt <= 0.0) {
    THROW_ATHELAS_ERROR("initial_dt must be strictly > 0.0\n");
  }
  if (dt_init_frac <= 1.0) {
    THROW_ATHELAS_ERROR("dt_init_frac must be strictly > 1.0\n");
  }
  if (dt_hdf5 <= 0.0) {
    THROW_ATHELAS_ERROR("dt_hdf5 must be strictly > 0.0\n");
  }
  if (hist_dt <= 0.0) {
    THROW_ATHELAS_ERROR("hist_dt must be strictly > 0.0\n");
  }
  params_->add("output.ncycle_out", ncycle_out);
  params_->add("output.dt_hdf5", dt_hdf5);
  params_->add("output.dt_init_frac", dt_init_frac);
  params_->add("output.initial_dt", initial_dt);
  params_->add("output.history_enabled", history_enabled);
  params_->add("output.hist_fn", hist_fn);
  params_->add("output.hist_dt", hist_dt);

  // --------------------------
  // ---------- time ----------
  // --------------------------
  if (!config_["time"].is_table()) {
    THROW_ATHELAS_ERROR("No [time] block provided!");
  }
  std::optional<std::string> integrator =
      config_["time"]["integrator"].value<std::string>();
  if (integrator.has_value()) {
    const MethodID method_id =
        string_to_id(utilities::to_lower(integrator.value()));
    params_->add("time.integrator", method_id);
    params_->add("time.integrator_string", integrator.value()); // for IO
  } else {
    THROW_ATHELAS_ERROR("You must list an integrator in the input deck!");
  }

  // -------------------------
  // ---------- eos ----------
  // -------------------------
  if (!config_["eos"].is_table()) {
    THROW_ATHELAS_ERROR("No [eos] block provided!");
  }
  std::optional<std::string> eos_type =
      config_["eos"]["type"].value<std::string>();
  if (!eos_type.has_value()) {
    THROW_ATHELAS_ERROR("'type' not provided in [eos] block!");
  }
  params_->add("eos.type", eos_type.value());
  params_->add("eos.gamma", config_["eos"]["gamma"].value_or(1.4));
  if (eos_type.value() == "polytropic") {
    params_->add("eos.k", config_["eos"]["k"].value<double>().value());
    params_->add("eos.n", config_["eos"]["n"].value<double>().value());
  }

  // --------------------------
  // ---------- opac ----------
  // --------------------------
  if (rad.value()) {
    if (!config_["opacity"].is_table()) {
      THROW_ATHELAS_ERROR("Radiation abled but no [opac] block provided!");
    }
    std::optional<std::string> opac_type =
        config_["opacity"]["type"].value<std::string>();
    if (!opac_type.has_value()) {
      THROW_ATHELAS_ERROR("'type' not provided in [opac] block!");
    }
    params_->add("opac.type", opac_type.value());

    if (opac_type.value() == "constant") {
      std::optional<double> kr = config_["opacity"]["kR"].value<double>();
      std::optional<double> kp = config_["opacity"]["kP"].value<double>();
      if (!kr.has_value() || !kp.has_value()) {
        THROW_ATHELAS_ERROR(
            "Constant opacity must specify mean opacities kR and kP!");
      }
      params_->add("opac.kR", kr.value());
      params_->add("opac.kP", kp.value());
    }
    if (opac_type.value() == "powerlaw_rho") {
      std::optional<double> kr = config_["opacity"]["kR"].value<double>();
      std::optional<double> kp = config_["opacity"]["kP"].value<double>();
      std::optional<double> exp =
          config_["opacity"]["exp"].value<double>(); // exponent
      if (!kr.has_value() || !kp.has_value() || !exp.has_value()) {
        THROW_ATHELAS_ERROR("Powerlaw rho opacity must specify mean opacities "
                            "kR and kP and an exponent exp!");
      }
      params_->add("opac.kR", kr.value());
      params_->add("opac.kP", kp.value());
      params_->add("opac.exp", exp.value());
    }
  }

  std::println("# Configuration ... Complete\n");
}

auto check_bc(std::string bc) -> bool {
  if (bc != "outflow" && bc != "reflecting" && bc != "dirichlet" &&
      bc != "periodic" && bc != "marshak") {
    THROW_ATHELAS_ERROR(
        " ! Initialization Error: Bad boundary condition choice. Choose: \n"
        " - outflow \n"
        " - reflecting \n"
        " - periodic \n"
        " - dirichlet");
  }
  return false; // should not reach
}

} // namespace athelas
