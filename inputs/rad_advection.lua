local config = {}

config.problem = {
  name = "rad_advection",
  cfl = 0.85,

  params = {
    v0 = 0.0,
    rho = 1.0,
    amp = 1.0,
    width = 0.05,
  },
}

config.mesh = {
  geometry = "planar",
  nx = 128,
  xl = 0.0,
  xr = 1.0,
  grid_type = "uniform",
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
    bc_i = "interior",
    bc_o = "interior",
  },
}

config.output = {
  ncycle_out = 100,
  dt_growth_frac = 1.1,
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
  discretization = "explicit",
  limiter = {
    tci_opt = false,
    tci_val = 0.1,
    characteristic = false,
  },
}

config.time = {
  t_end = 1.0e-10,
  nlim = -1,
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
