local config = {}

config.problem = {
  name = "hydrostatic_balance",
  t_end = 600.0,
  nlim = -1,
  geometry = "spherical",
  xl = 0.0,
  xr = 7.000000e+08,
  cfl = 0.5,
  nx = 1024,
  grid_type = "uniform",

  params = {
    rho_c = 150.0,
    p_threshold = 0.2,
  },
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
  dt_init_frac = 1.05,
  history = {
    fn = "hydrostatic_balance.hst",
  },
}

config.fluid = {
  limiter = {
    do_limiter = false,
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
  integrator = "EX_SSPRK54",
}

config.eos = {
  type = "polytropic",
  k = 1.000000e+15,
  n = 3.0,
}

return config
