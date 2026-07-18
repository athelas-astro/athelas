local config = {}

config.problem = {
  name = "hydrostatic_balance",
  cfl = 0.5,

  params = {
    rho_c = 150.0,
    p_threshold = 0.2,
  },
}

config.mesh = {
  geometry = "spherical",
  nx = 1024,
  xl = 0.0,
  xr = 7.000000e+08,
  grid_type = "uniform",
}

config.physics = {
  radiation = false,
  gravity = true,
  composition = false,
  ionization = false,
  heating = false,
  engine = false,
}

config.basis = {
  nnodes = 4,
}

config.bc = {
  fluid = {
    bc_i = "reflecting",
    bc_o = "outflow",
  },
}

config.output = {
  ncycle_out = 100,
  dt_growth_frac = 1.05,
  history = {
    fn = "hydrostatic_balance.hst",
  },
}

config.fluid = {
  limiter = {
    enabled = false,
    type = "minmod",
    b_tvd = 1.0,
    m_tvb = 0.0,
    tci_opt = false,
    tci_val = 0.25,
  },
}

config.gravity = {
  model = "spherical",
  gval = 1000.0,
}

config.time = {
  t_end = 600.0,
  nlim = -1,
  integrator = "EX_SSPRK54",
}

config.eos = {
  type = "polytropic",
  k = 1.000000e+15,
  n = 3.0,
}

return config
