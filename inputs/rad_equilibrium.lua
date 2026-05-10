local config = {}

config.problem = {
  name = "rad_equilibrium",
  t_end = 1.0e-07,
  nlim = -1,
  geometry = "planar",
  xl = 0.0,
  xr = 1.0,
  cfl = 0.8,
  nx = 2,
  grid_type = "uniform",
  params = {
    v0 = 0.0,
    logrho = -7.0,
    logE_gas = 10.0,
    logE_rad = 12.0,
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
  dt_init_frac = 1.005,
  history = {
    fn = "rad_equilibrium.hst",
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
  discretization = "implicit",
  timestep = {
    max_fractional_change_e = 0.0005,
    max_change_f = 0.001,
  },
  limiter = {
    tci_opt = false,
    tci_val = 0.1,
    characteristic = false,
  },
}

config.time = {
  integrator = "IMEX_SSPRK11",
}

config.eos = {
  type = "ideal",
  gamma = 5.0 / 3.0,
}

config.opacity = {
  type = "powerlaw",
  kP = 4.000000e-08,
  kR = 4.000000e-08,
  rho_exp = -1.0,
  t_exp = 0.0,
}

return config
