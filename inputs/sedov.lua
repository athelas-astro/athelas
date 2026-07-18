local config = {}

config.problem = {
  name = "sedov",
  cfl = 0.25,

  params = {
    v0 = 0.0,
    rho0 = 1.0,
    E0 = 0.5,
  },
}

config.mesh = {
  geometry = "spherical",
  nx = 256,
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

config.time = {
  t_end = 0.05,
  nlim = -1,
  integrator = "EX_SSPRK33",
}

config.basis = {
  nnodes = 3,
}

config.bc = {
  fluid = {
    bc_i = "reflecting",
    bc_o = "outflow",
  },
}

config.output = {
  ncycle_out = 100,
  dt_hdf5 = 0.001,
  dt_growth_frac = 1.05,
  history = {
    fn = "sedov.hst",
  },
}

config.fluid = {
  limiter = {
    enabled = true,
    type = "minmod",
    b_tvd = 1.0,
    m_tvb = 0.0,
    tci_opt = true,
    tci_val = 0.25,
    characteristic = false,
    gamma_i = 0.9,
    weno_r = 2.0,
  },
}

config.gravity = {
  model = "constant",
  gval = 1.0,
}

config.eos = {
  type = "ideal",
  gamma = 5.0 / 3.0,
}

return config
