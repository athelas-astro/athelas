local config = {}

config.problem = {
  name = "shu_osher",
  cfl = 0.35,

  params = {
    v0 = 2.629369,
    rhoL = 3.857143,
    pL = 10.33333333,
    pR = 1.0,
  },
}

config.mesh = {
  geometry = "planar",
  nx = 128,
  xl = -10.0,
  xr = 10.0,
  grid_type = "uniform",
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
    enabled = true,
    type = "minmod",
    b_tvd = 1.0,
    m_tvb = 0.0,
    tci_opt = false,
    tci_val = 0.1,
    characteristic = false,
  },
}

config.time = {
  t_end = 1.8,
  nlim = -1,
  integrator = "EX_SSPRK33",
}

config.eos = {
  type = "ideal",
  gamma = 1.4,
}

return config
