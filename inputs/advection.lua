local config = {}

config.problem = {
  name = "smooth_advection",
  cfl = 0.2,

  params = {
    v0 = 1.0,
    p0 = 0.1,
    amp = 1.0,
  },
}

config.mesh = {
  geometry = "planar",
  nx = 2048,
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
  t_end = 1.0,
  nlim = -1,
  integrator = "EX_SSPRK54",
}

config.basis = {
  nnodes = 1,
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
