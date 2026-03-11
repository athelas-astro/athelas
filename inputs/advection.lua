local config = {}

config.problem = {
  name = "smooth_advection",
  t_end = 1.0,
  nlim = -1,
  geometry = "planar",
  xl = 0.0,
  xr = 1.0,
  cfl = 0.35,
  nx = 128,
  grid_type = "uniform",

  params = {
    v0 = 1.0,
    p0 = 0.1,
    amp = 1.0,
  },
}

config.physcs = {
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

config.bc = {
  fluid = { bc_i = "periodic", bc_o = "periodic" },
}

config.output = {
  ncycle_out = 100,
  dt_hdf5 = 0.1,
  history = {
    fn = "advection.hst",
    dt = 0.001,
  },
}

config.eos = {
  type = "ideal",
}

return config
