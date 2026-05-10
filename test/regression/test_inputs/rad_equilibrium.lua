local config = {}

config.problem = {
  name = "rad_equilibrium",
  t_end = 1.0e-07,
  nlim = -1,
  geometry = "planar",
  restart = false,
  xl = 0.0,
  xr = 1.0,
  bc = "homogenous",
  cfl = 0.9,
  nx = 1,
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

config.time = {
  integrator = "IMEX_ARK32_ESDIRK",
}

config.basis = {
  nnodes = 3,
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
  dt_init_frac = 1.05,
}

config.fluid = {
  limiter = {
    do_limiter = true,
    type = "minmod",
    tci_opt = false,
    tci_val = 0.1,
    characteristic = false,
  },
}

config.radiation = {
  discretization = "explicit",
  limiter = {
    do_limiter = true,
    type = "minmod",
    m_tvb = 0.0,
    b_tvd = 1.0,
    tci_opt = false,
    tci_val = 0.1,
    characteristic = false,
  },
}

config.eos = {
  type = "ideal",
  gamma = 1.66666666666667,
}

config.opacity = {
  type = "powerlaw",
  kP = 4.000000e-08,
  kR = 4.000000e-08,
  rho_exp = -1.0,
  t_exp = 0.0,
}

return config
