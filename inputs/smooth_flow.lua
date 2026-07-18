local config = {}

config.problem = {
  name = "smooth_flow",
  cfl = 0.1,

  params = {
    amp = 0.1,
    -- amp = 0.99999999999999,
  },
}

config.mesh = {
  geometry = "planar",
  nx = 2048,
  xl = -1.0,
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
  t_end = 0.15,
  nlim = -1,
  integrator = "EX_SSPRK33",
}

config.basis = {
  nnodes = 3,
}

config.bc = {
  fluid = {
    bc_i = "periodic",
    bc_o = "periodic",
  },
}

config.output = {
  -- dt_fixed = 1.0e-10,
  ncycle_out = 100,
  -- dt_hdf5 = 0.1,
  history = {
    fn = config.problem.name .. ".hst",
  },
}

config.fluid = {
  limiter = {
    enabled = false,
  },
}

config.eos = {
  type = "ideal",
  gamma = 3.0,
}

return config
