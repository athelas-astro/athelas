-- schema.lua
-- ----------
-- Athelas input deck schema.
--
-- Leaf nodes are tables containing a `doc` key. All other tables are
-- interior nodes that the validator recurses into.
--
-- Leaf fields:
--   doc      (string, required, may be empty) -- human-readable description
--   type     (string, optional)               -- for documentation only
--   default  (any,    optional)               -- inserted into config if key absent
--   required (bool or table, optional)
--     true                                    -- always required
--     { when = "section.key", equals = val }  -- required when another key equals val
--     { when = "section.key", is_true = true} -- required when another key is true
--   ignore   (bool, optional)                 -- skip this subtable entirely (no validation)

local schema = {}

-- -------------------------
-- problem
-- -------------------------
schema.problem = {
  name = {
    type = "string",
    required = true,
    doc = "Unique identifier for this simulation problem.",
  },
  t_end = {
    type = "double",
    required = true,
    doc = "End time of the simulation.",
  },
  nlim = {
    type = "double",
    default = -1,
    doc = "Maximum double of cycles. -1 for unlimited.",
  },
  geometry = {
    type = "string",
    required = true,
    doc = "Domain geometry. Options: 'planar', 'spherical'.",
  },
  xl = {
    type = "double",
    required = true,
    doc = "Left boundary of the domain.",
  },
  xr = {
    type = "double",
    required = true,
    doc = "Right boundary of the domain. Must be > xl.",
  },
  cfl = {
    type = "double",
    required = true,
    doc = "CFL double for timestep control. Must be > 0.",
  },
  nx = {
    type = "double",
    required = true,
    doc = "Number of grid cells. Must be > 0.",
  },
  grid_type = {
    type = "string",
    required = true,
    doc = "Grid spacing type. Options: 'uniform', 'logarithmic'.",
  },
  params = {
    required = true,
    doc = "Problem-specific parameters. Validated by the problem generator.",
  },
}

-- -------------------------
-- physics
-- -------------------------
-- Should these be required?
schema.physics = {
  radiation = {
    type = "bool",
    default = false,
    doc = "Enable radiation transport.",
  },
  gravity = {
    type = "bool",
    default = false,
    doc = "Enable gravitational source terms.",
  },
  composition = {
    type = "bool",
    default = false,
    doc = "Enable multi-species composition.",
  },
  ionization = {
    type = "bool",
    default = false,
    doc = "Enable Saha ionization. Requires composition = true.",
  },
  heating = {
    type = "bool",
    default = false,
    doc = "Enable nuclear heating sources.",
  },
  engine = {
    type = "bool",
    default = false,
    doc = "Enable energy injection engine.",
  },
}

-- -------------------------
-- basis
-- -------------------------
schema.basis = {
  nnodes = {
    type = "double",
    required = true,
    doc = "Number of DG nodes per cell.",
  },
}

-- -------------------------
-- time
-- -------------------------
schema.time = {
  integrator = {
    type = "string",
    required = true,
    doc = "Time integration method. E.g. 'IMEX_SSPRK11', 'IMEX_ARK32_ESDIRK'.",
  },
}

-- -------------------------
-- eos
-- -------------------------
schema.eos = {
  type = {
    type = "string",
    required = true,
    doc = "Equation of state type. Options:. 'ideal', 'paczynski', 'marshak', 'polytropic'.",
  },
  gamma = {
    type = "double",
    default = 1.4,
    doc = "Adiabatic index. Default: 1.4.",
  },
  k = {
    type = "double",
    required = { when = "eos.type", equals = "polytropic" },
    doc = "Polytropic constant K. Required for polytropic EOS.",
  },
  n = {
    type = "double",
    required = { when = "eos.type", equals = "polytropic" },
    doc = "Polytropic index n. Required for polytropic EOS.",
  },
}

-- -------------------------
-- fluid
-- -------------------------
schema.fluid = {
  limiter = {
    do_limiter = {
      type = "bool",
      default = true,
      doc = "Enable slope limiter for fluid.",
    },
    type = {
      type = "string",
      default = "minmod",
      doc = "Limiter type. Options: 'minmod', 'weno [experimental]'.",
    },
    b_tvd = {
      type = "double",
      default = 1.0,
      doc = "TVD parameter b. Used with minmod limiter.",
    },
    m_tvb = {
      type = "double",
      default = 0.0,
      doc = "TVB parameter M. Used with minmod limiter.",
    },
    tci_opt = {
      type = "bool",
      default = false,
      doc = "Enable troubled-cell indicator.",
    },
    tci_val = {
      type = "double",
      required = { when = "fluid.limiter.tci_opt", is_true = true },
      doc = "Troubled-cell indicator threshold.",
    },
    characteristic = {
      type = "bool",
      default = false,
      doc = "Enable characteristic limiting.",
    },
    gamma_i = {
      type = "double",
      required = { when = "fluid.limiter.type", equals = "weno" },
      doc = "WENO central weight. Required for WENO limiter.",
    },
    gamma_l = {
      type = "double",
      doc = "WENO left weight. Inferred from gamma_i if omitted.",
    },
    gamma_r = {
      type = "double",
      doc = "WENO right weight. Inferred from gamma_i if omitted.",
    },
    weno_r = {
      type = "double",
      default = 2.0,
      doc = "WENO smoothness exponent. Must be > 0.",
    },
  },
}

-- -------------------------
-- radiation
-- -------------------------
schema.radiation = {
  discretization = {
    type = "string",
    required = { when = "physics.radiation", is_true = true },
    doc = "Spatial discretization of the transport term. Options: 'implicit' or 'explicit'.",
  },
  timestep = {
    max_fractional_change_e = {
      type = "double",
      required = { when = "radiation.discretization", equals = "implicit" },
      doc = "Maximum allowed fractional change in radiation energy. Timestep control for implicit transport.",
    },
    max_change_f = {
      type = "double",
      required = { when = "radiation.discretization", equals = "implicit" },
      doc = "Maximum allowed absolute change in radiation reduced flux. Timestep control for implicit transport.",
    },
  },
  newton = {
    max_iter = {
      type = "int",
      default = 10,
      doc = "Maximum Newton iterations for implicit transport solve.",
    },
    tol = {
      type = "double",
      default = 1.0e-8,
      doc = "Convergence tolerance for implicit transport Newton iteration.",
    },
  },
  limiter = {
    do_limiter = {
      type = "bool",
      default = true,
      doc = "Enable slope limiter for radiation.",
    },
    type = {
      type = "string",
      default = "minmod",
      doc = "Limiter type. Options: 'minmod', 'weno'.",
    },
    b_tvd = {
      type = "double",
      default = 1.0,
      doc = "TVD parameter b. Used with minmod limiter.",
    },
    m_tvb = {
      type = "double",
      default = 0.0,
      doc = "TVB parameter M. Used with minmod limiter.",
    },
    tci_opt = {
      type = "bool",
      default = false,
      doc = "Enable troubled-cell indicator.",
    },
    tci_val = {
      type = "double",
      required = { when = "radiation.limiter.tci_opt", is_true = true },
      doc = "Troubled-cell indicator threshold.",
    },
    characteristic = {
      type = "bool",
      default = false,
      doc = "Enable characteristic limiting. Currently unsupported for radiation.",
    },
    gamma_i = {
      type = "double",
      required = { when = "radiation.limiter.type", equals = "weno" },
      doc = "WENO central weight. Required for WENO limiter.",
    },
    gamma_l = {
      type = "double",
      doc = "WENO left weight. Inferred from gamma_i if omitted.",
    },
    gamma_r = {
      type = "double",
      doc = "WENO right weight. Inferred from gamma_i if omitted.",
    },
    weno_r = {
      type = "double",
      default = 2.0,
      doc = "WENO smoothness exponent. Must be > 0.",
    },
  },
}

-- -------------------------
-- bc
-- -------------------------
schema.bc = {
  fluid = {
    bc_i = {
      type = "string",
      required = true,
      doc = "Inner fluid boundary condition. Options: 'reflecting', 'outflow', 'dirichlet', 'periodic'.",
    },
    bc_o = {
      type = "string",
      required = true,
      doc = "Outer fluid boundary condition. Options: 'reflecting', 'outflow', 'dirichlet', 'periodic'.",
    },
    dirichlet_values_i = {
      type = "array",
      required = { when = "bc.fluid.bc_i", equals = "dirichlet" },
      doc = "Dirichlet state values at inner boundary.",
    },
    dirichlet_values_o = {
      type = "array",
      required = { when = "bc.fluid.bc_o", equals = "dirichlet" },
      doc = "Dirichlet state values at outer boundary.",
    },
  },
  radiation = {
    bc_i = {
      type = "string",
      required = { when = "physics.radiation", is_true = true },
      doc = "Inner radiation boundary condition. Options: 'reflecting', 'outflow', 'marshak', 'dirichlet'.",
    },
    bc_o = {
      type = "string",
      required = { when = "physics.radiation", is_true = true },
      doc = "Outer radiation boundary condition. Options: 'reflecting', 'outflow', 'dirichlet'.",
    },
    dirichlet_values_i = {
      type = "array",
      required = { when = "bc.radiation.bc_i", equals = "dirichlet" },
      doc = "Dirichlet state values at inner boundary.",
    },
    dirichlet_values_o = {
      type = "array",
      required = { when = "bc.radiation.bc_o", equals = "dirichlet" },
      doc = "Dirichlet state values at outer boundary.",
    },
  },
}

-- -------------------------
-- output
-- -------------------------
schema.output = {
  ncycle_out = {
    type = "double",
    default = 100,
    doc = "Number of cycles between stdout output.",
  },
  dt_hdf5 = {
    type = "double",
    doc = "Time interval between HDF5 outputs. Default: t_end / 100.",
  },
  dt_init = {
    type = "double",
    default = 1.0e-16,
    doc = "Initial timestep. Must be > 0.",
  },
  dt_init_frac = {
    type = "double",
    default = 1.05,
    doc = "Initial timestep growth factor. Must be > 1.",
  },
  dt_fixed = {
    type = "double",
    doc = "Fixed timestep override. Optional.",
  },
  history = {
    fn = {
      type = "string",
      default = "athelas.hst",
      doc = "History output filename.",
    },
    dt = {
      type = "double",
      doc = "Time interval between history writes. Default: dt_hdf5 / 10.",
    },
  },
}

-- -------------------------
-- gravity
-- -------------------------
schema.gravity = {
  model = {
    type = "string",
    doc = "Gravity model. Options: 'constant', 'spherical'.",
    required = { when = "physics.gravity", is_true = true },
  },
  operator_split = {
    type = "bool",
    default = false,
    doc = "Apply gravity as an operator-split source.",
  },
  gval = {
    type = "double",
    --    default = 1.0,
    doc = "Gravitational acceleration (constant model). Must be > 0.",
    required = { when = "gravity.model", equals = "constant" },
  },
}

-- -------------------------
-- composition
-- -------------------------
schema.composition = {
  ncomps = {
    type = "int",
    required = { when = "physics.composition", is_true = true },
    doc = "Number of composition species.",
  },
}

-- -------------------------
-- ionization
-- -------------------------
schema.ionization = {
  fn_ionization = {
    type = "string",
    required = { when = "physics.ionization", is_true = true },
    doc = "Path to ionization atomic data file.",
  },
  fn_degeneracy = {
    type = "string",
    required = { when = "physics.ionization", is_true = true },
    doc = "Path to degeneracy factors atomic data file.",
  },
  ncomps = {
    type = "int",
    required = { when = "physics.ionization", is_true = true },
    doc = "Number of species for Saha solver.",
  },
  solver = {
    type = "string",
    default = "linear",
    doc = "Saha solver mode. Options: 'linear', 'log'.",
  },
}

-- -------------------------
-- heating
-- -------------------------
schema.heating = {
  nickel = {
    enabled = {
      type = "bool",
      required = { when = "physics.heating", is_true = true },
      doc = "Enable Ni56 decay heating.",
    },
    model = {
      type = "string",
      required = { when = "physics.heating", is_true = true },
      doc = "Nickel heating model. Options. 'jeffery' and 'full_trapping'.",
    },
    operator_split = {
      type = "bool",
      default = false,
      doc = "Apply nickel heating as operator-split source.",
    },
  },
}

-- -------------------------
-- engine
-- -------------------------
schema.engine = {
  thermal = {
    enabled = {
      type = "bool",
      required = { when = "physics.engine", is_true = true },
      doc = "Enable thermal energy injection engine.",
    },
    mode = {
      type = "string",
      required = { when = "engine.thermal.enabled", is_true = true },
      doc = "Injection mode. Options: 'direct', 'asymptotic'.",
    },
    energy = {
      type = "double",
      required = { when = "engine.thermal.enabled", is_true = true },
      doc = "Total injected energy in erg.",
    },
    tend = {
      type = "double",
      required = { when = "engine.thermal.enabled", is_true = true },
      doc = "End time for energy injection. Must be > 0.",
    },
    mend = {
      type = "double",
      required = { when = "engine.thermal.enabled", is_true = true },
      doc = "Mass coordinate of injection upper boundary in Msun. Must be > 0.",
    },
    operator_split = {
      type = "bool",
      default = false,
      doc = "Apply engine as operator-split source.",
    },
  },
}

-- -------------------------
-- opacity
-- -------------------------
schema.opacity = {
  type = {
    type = "string",
    required = { when = "physics.radiation", is_true = true },
    doc = "Opacity model. Options: 'tabular', 'constant', 'powerlaw'.",
  },
  filename = {
    type = "string",
    required = { when = "opacity.type", equals = "tabular" },
    doc = "Path to tabular opacity HDF5 file.",
  },
  kR = {
    type = "double",
    required = { when = "opacity.type", equals = "constant" },
    doc = "Rosseland mean opacity (cm^2/g). Required for constant and powerlaw.",
  },
  kP = {
    type = "double",
    required = { when = "opacity.type", equals = "constant" },
    doc = "Planck mean opacity (cm^2/g). Required for constant and powerlaw.",
  },
  rho_exp = {
    type = "double",
    required = { when = "opacity.type", equals = "powerlaw" },
    doc = "Density exponent for powerlaw opacity.",
  },
  t_exp = {
    type = "double",
    required = { when = "opacity.type", equals = "powerlaw" },
    doc = "Temperature exponent for powerlaw opacity.",
  },
  kR_offset = {
    type = "double",
    default = 0.0,
    doc = "Rosseland opacity additive offset. Powerlaw only.",
  },
  kP_offset = {
    type = "double",
    default = 0.0,
    doc = "Planck opacity additive offset. Powerlaw only.",
  },
  floors = {
    type = {
      type = "string",
      default = "core_envelope",
      doc = "Opacity floor model. Options: 'core_envelope', 'constant'.",
    },
    core_rosseland = {
      type = "double",
      default = 0.24,
      doc = "Core Rosseland floor. Used with core_envelope model.",
    },
    core_planck = {
      type = "double",
      default = 0.24,
      doc = "Core Planck floor. Used with core_envelope model.",
    },
    env_rosseland = {
      type = "double",
      default = 0.01,
      doc = "Envelope Rosseland floor. Used with core_envelope model.",
    },
    env_planck = {
      type = "double",
      default = 0.01,
      doc = "Envelope Planck floor. Used with core_envelope model.",
    },
    rosseland = {
      type = "double",
      default = 1.0e-3,
      doc = "Rosseland floor value. Used with constant floor model.",
    },
    planck = {
      type = "double",
      default = 1.0e-3,
      doc = "Planck floor value. Used with constant floor model.",
    },
  },
}

return schema
