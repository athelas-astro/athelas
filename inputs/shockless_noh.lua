local config = {}

config.problem = {
  name = "shockless_noh",
  t_end = 0.5,
  nlim = -1,
  geometry = "planar",
  xl = 0.0,
  xr = 1.0,
  cfl = 0.35,
  nx = 128,
  grid_type = "uniform",

  params = {
    rho = 1.0,
    specific_energy = 1.0,
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
    bc_o = "reflecting",
  },
}

config.output = {
  ncycle_out = 100,
  dt_hdf5 = 0.05,
  history = {
    fn = config.problem.name .. ".hst",
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
  gamma = 1.4,
}

return config
