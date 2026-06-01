local config = {}

config.problem = {
  name = "shocktube",
  t_end = 0.2,
  nlim = -1,
  geometry = "planar",
  xl = 0.0,
  xr = 1.0,
  cfl = 0.8,
  nx = 512,
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

config.basis = {
  nnodes = 3,
}

config.physics = {
  radiation = false,
  gravity = false,
  composition = false,
  ionization = false,
  heating = false,
  engine = false,
}

config.bc = {
  fluid = {
    bc_i = "outflow",
    bc_o = "outflow",
  },
}

config.output = {
  ncycle_out = 100000,
  dt_hdf5 = 1.0,
  dt_init_frac = 1.5,
}

config.fluid = {
  limiter = {
    enabled = true,
    type = "minmod",
    b_tvd = 1.0,
    m_tvb = 0.0,
    tci_opt = false,
    tci_val = 1.0,
    characteristic = true,
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
