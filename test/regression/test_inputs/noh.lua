local config = {}

config.problem = {
  name = "noh",
  t_end = 0.5,
  nlim = -1,
  geometry = "planar",
  xl = 0.0,
  xr = 1.0,
  cfl = 0.1,
  nx = 128,
  grid_type = "uniform",

  params = {
    v0 = -1.0,
    rho0 = 1.0,
    p0 = 1.0e-06,
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
    bc_o = "surface",
  },
}

config.output = {
  ncycle_out = 100000,
  dt_hdf5 = 1.0,
  dt_init_frac = 1.01,
}

config.fluid = {
  limiter = {
    enabled = true,
    type = "moment",
    b_tvd = 1.0,
    m_tvb = 0.0,
    tci_opt = false,
    tci_val = 0.1,
    characteristic = true,
  },
}

config.time = {
  integrator = "EX_SSPRK22",
}

config.eos = {
  type = "ideal",
}

return config
