local config = {}

config.problem = {
  name = "moving_contact",
  t_end = 1.0,
  nlim = -1,
  geometry = "planar",
  xl = 0.0,
  xr = 1.0,
  cfl = 0.35,
  nx = 128,
  grid_type = "uniform",

  params = {
    v0 = 0.1,
    rhoL = 1.4,
    rhoR = 1.0,
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
  nnodes = 1,
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
    fn = "moving_contact.hst",
  },
}

config.fluid = {
  limiter = {
    tci_opt = false,
    tci_val = 0.1,
    characteristic = true,
  },
}

config.time = {
  integrator = "EX_SSPRK11",
}

config.eos = {
  type = "ideal",
}

return config
