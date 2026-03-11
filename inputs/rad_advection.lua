local config = {}

config.problem = {
  name = "rad_advection",
  t_end = 1.0e-10,
  nlim = -1,
  geometry = "planar",
  xl = 0.0,
  xr = 1.0,
  cfl = 0.85,
  nx = 128,
  grid_type = "uniform",

  params = {
    v0 = 0.0,
    rho = 1.0,
    amp = 1.0,
    width = 0.05,
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
  dt_init_frac = 1.001,
  history = {
    fn = "rad_advection.hst",
  },
}

config.fluid = {
  limiter = {
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
  kP = 4.000000e-08,
  kR = 4.000000e-08,
  exp = -1.0,
  floors = {
    type = "constant",
    planck = 0.0,
    rosseland = 0.0,
  },
}

return config
