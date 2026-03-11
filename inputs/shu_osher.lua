local config = {}

config.problem = {
  name = "shu_osher",
  t_end = 1.8,
  nlim = -1,
  geometry = "planar",
  xl = -10.0,
  xr = 10.0,
  cfl = 0.35,
  nx = 128,
  grid_type = "uniform",

  params = {
    v0 = 2.629369,
    rhoL = 3.857143,
    pL = 10.33333333,
    pR = 1.0,
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
    bc_i = "outflow",
    bc_o = "outflow",
  },
}

config.output = {
  ncycle_out = 100,
  dt_hdf5 = 0.01,
  history = {
    fn = config.problem.name .. "hst",
  },
}

config.fluid = {
  limiter = {
    do_limiter = true,
    type = "minmod",
    b_tvd = 1.0,
    m_tvb = 0.0,
    tci_opt = false,
    tci_val = 0.1,
    characteristic = false,
  },
}

config.time = {
  integrator = "EX_SSPRK33",
}

config.eos = {
  type = "ideal",
  gamma = 1.4,
}

return config
