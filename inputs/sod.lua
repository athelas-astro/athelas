local config = {}

config.problem = {
  name = "sod",
  t_end = 0.2,
  nlim = -1,
  geometry = "planar",
  xl = 0.0,
  xr = 1.0,
  cfl = 0.5,
  nx = 256,
  grid_type = "uniform",

  params = {
    vL = 0.0,
    vR = 0.0,
    rhoL = 1.0,
    rhoR = 0.125,
    pL = 1.0,
    pR = 0.1,
    x_d = 0.5,
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

config.time = {
  integrator = "EX_SSPRK33",
}

config.basis = {
  nnodes = 3,
}

config.output = {
  ncycle_out = 100,
  dt_hdf5 = 2.5e-2, -- default: 100 outputs
  dt_init_frac = 1.1,

  history = {
    fn = config.problem.name .. ".hst",
  },
}

config.bc = {
  fluid = { bc_i = "outflow", bc_o = "outflow" },
}

config.fluid = {
  limiter = {
    do_limiter = true,
    type = "minmod",
    m_tvb = 10.0,
    b_tvd = 1.0,
    tci_opt = true,
    tci_val = 0.075,
    characteristic = false,
  },
}

config.eos = {
  type = "ideal",
  gamma = 1.4,
}

return config
