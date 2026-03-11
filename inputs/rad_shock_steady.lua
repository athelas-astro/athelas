local config = {}

config.problem = {
  name = "rad_shock_steady",
  t_end = 1.000000e-11,
  nlim = -1,
  geometry = "planar",
  xl = -0.03,
  xr = 0.03,
  cfl = 0.8,
  nx = 128,
  grid_type = "uniform",

  params = {
    v0 = 0.0,
    rhoL = 1.0,
    rhoR = 2.286,
    T_L = 1.160452e+06,
    T_R = 2.410900e+06,
  },
}

config.physics = {
  radiation = true,
  gravity = false,
  composition = false,
  ionization = false,
  heating = false,
  engine = false,
}

config.basis = {
  nnodes = 1,
}

config.bc = {
  fluid = {
    bc_i = "outflow",
    bc_o = "outflow",
  },
  radiation = {
    bc_i = "outflow",
    bc_o = "outflow",
  },
}

config.output = {
  ncycle_out = 100,
  dt_init_frac = 1.0001,
  history = {
    fn = config.problem.name .. ".hst",
  },
}

config.fluid = {
  limiter = {
    type = "minmod",
    tci_opt = false,
    tci_val = 0.1,
    characteristic = false,
  },
}

config.radiation = {
  limiter = {
    tci_opt = false,
    tci_val = 0.1,
    characteristic = false,
  },
}

config.time = {
  integrator = "IMEX_SSPRK22_DIRK",
}

config.eos = {
  type = "ideal",
  gamma = 5.0 / 3.0,
}

config.opacity = {
  type = "powerlaw_rho",
  kP = 422.99,
  kR = 788.03,
  exp = -1.0,
  floors = {
    type = "constant",
    planck = 0.0,
    rosseland = 0.0,
  },
}

return config
