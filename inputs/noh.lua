local config = {}

config.problem = {
  name = "noh",
  cfl = 0.1,

  params = {
    v0 = -1.0,
    rho0 = 1.0,
    p0 = 1.000000e-06,
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
  radiation = false,
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
    bc_i = "reflecting",
    bc_o = "surface",
  },
}

config.output = {
  ncycle_out = 100,
  dt_hdf5 = 0.1,
  dt_growth_frac = 1.01,
  history = {
    fn = "noh.hst",
  },
}

config.fluid = {
  limiter = {
    enabled = true,
    type = "moment",
    b_tvd = 1.0,
    m_tvb = 0.0,
    tci_opt = true,
    tci_val = 0.1,
    characteristic = false,
  },
}

config.time = {
  t_end = 0.5,
  nlim = -1,
  integrator = "EX_SSPRK22",
}

config.eos = {
  type = "ideal",
  gamma = 5.0 / 3.0,
}

return config
