local config = {}

config.problem = {
  name = "ejecta_csm",
  t_end = 0.025,
  nlim = -1,
  geometry = "spherical",
  restart = false,
  xl = 0.0,
  xr = 1.25,
  cfl = 0.5,
  nx = 128,
  grid_type = "uniform",
  params = {
    rstar = 0.01,
    vmax = 1.8257418583505538,
  },
}

config.physics = {
  radiation = false,
  gravity = false,
  composition = false,
  ionization = false,
  heating = false,
  engine = false,
}

config.basis = {
  nnodes = 2,
}

config.bc = {
  fluid = {
    bc_i = "reflecting",
    bc_o = "outflow",
  },
}

config.output = {
  ncycle_out = 100,
  dt_init_frac = 1.01,
  history = {
    fn = "ejecta_csm.hst",
    dt = 0.001,
  },
}

config.fluid = {
  limiter = {
    do_limiter = true,
    type = "minmod",
    m_tvb = 0.0,
    b_tvd = 1.0,
    tci_opt = true,
    tci_val = 0.075,
    characteristic = false,
  },
}

config.time = {
  integrator = "EX_SSPRK33",
}

config.eos = {
  type = "ideal",
  gamma = 1.6666666667,
}

return config
