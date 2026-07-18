local config = {}

config.problem = {
  name = "gas_collapse",
  cfl = 0.5,

  params = {
    rho0 = 1.0e9,
    -- p0 << 4 pi G xr^2 rho0^2
    p0 = 5.0e25,
    v0 = 0.0,
  },
}

config.mesh = {
  geometry = "spherical",
  nx = 1024,
  xl = 0.0,
  xr = 6.5e+08,
  grid_type = "logarithmic",
}

config.physics = {
  radiation = false,
  gravity = true,
  composition = false,
  ionization = false,
  heating = false,
  engine = false,
}

config.time = {
  t_end = 0.0368,
  nlim = -1,
  integrator = "EX_SSPRK33",
}

config.basis = {
  nnodes = 3,
}

config.bc = {
  fluid = {
    bc_i = "reflecting",
    bc_o = "surface",
  },
}

config.output = {
  ncycle_out = 100,
  dt_growth_frac = 1.05,
  dt_hdf5 = config.time.t_end / 100.0,
  history = {
    fn = "gas_collapse.hst",
  },
}

config.fluid = {
  limiter = {
    enabled = true,
    type = "minmod",
    b_tvd = 1.0,
    m_tvb = 0.0,
    tci_opt = false,
    tci_val = 0.25,
  },
}

config.gravity = {
  model = "spherical",
}

config.eos = {
  type = "ideal",
  gamma = 1.4,
}

return config
