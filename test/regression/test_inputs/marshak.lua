local config = {}

config.problem = {
  name = "marshak",
  t_end = 1.0e-14,
  nlim = -1,
  geometry = "planar",
  restart = false,
  xl = 0.0,
  xr = 0.003466205,
  cfl = 0.3,
  nx = 512,
  grid_type = "uniform",

  params = {
    v0 = 0.0,
    rho0 = 10.0,
    T0 = 10000.0,
    epsilon = 1.0,
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
    bc_i = "marshak",
    bc_o = "outflow",
    dirichlet_values_i = { 1.1113063385e+12, 0.0 },
  },
}

config.output = {
  ncycle_out = 100,
  dt_init_frac = 1.0005,
  dt_hdf5 = 1.0,
}

config.fluid = {
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
  type = "marshak",
  gamma = 1.66666666666667,
}

config.opacity = {
  type = "constant",
  kP = 577.0,
  kR = 577.0,
}

return config
