local config = {}

config.problem = {
  name = "smooth_flow",
  t_end = 0.5,
  nlim = -1,
  geometry = "planar",
  xl = 0.0,
  xr = 1.0,
  cfl = 0.5,
  nx = 128,
  grid_type = "uniform",

  params = {
    amp = 0.99999999999999,
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
    bc_i = "periodic",
    bc_o = "periodic",
  },
}

config.output = {
  ncycle_out = 100,
  dt_hdf5 = 0.1,
  history = {
    fn = config.problem.name .. "hst",
  },
}

config.fluid = {
  limiter = {
    tci_opt = false,
    tci_val = 0.1,
    characteristic = true,
    gamma_i = 0.8,
  },
}

config.time = {
  integrator = "EX_SSPRK22",
}

config.eos = {
  type = "ideal",
}

return config
