local config = {}

config.problem = {
  name = "ejecta_csm",
  t_end = 1.0,
  nlim = -1,
  geometry = "spherical",
  xl = 0.0,
  xr = 1.25,
  cfl = 0.5,
  nx = 256,
  grid_type = "uniform",
  params = {
    rstar = 0.01,
    vmax = math.sqrt(10.0 / 3.0),
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
    tci_opt = false,
    tci_val = 0.075,
    characteristic = false,
  },
}

config.time = {
  integrator = "EX_SSPRK22",
}

config.eos = {
  type = "ideal",
  gamma = 5.0 / 3.0,
}

return config
