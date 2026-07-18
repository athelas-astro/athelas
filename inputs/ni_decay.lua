local config = {}

config.problem = {
  name = "ni_decay",
  cfl = 0.5,

  params = {
    temperature = 1000.0,
    rho = 1000.0,
  },
}

config.mesh = {
  geometry = "spherical",
  nx = 16,
  xl = 0.0,
  xr = 3.622000e+09,
  grid_type = "uniform",
}

config.physics = {
  radiation = false,
  gravity = false,
  composition = true,
  ionization = false,
  heating = true,
  engine = false,
}

config.basis = {
  nnodes = 2,
}

config.composition = {
  ncomps = 3,
}

config.heating = {
  nickel = {
    enabled = true,
    model = "jeffery",
    operator_split = false,
  },
}

config.bc = {
  fluid = {
    bc_i = "reflecting",
    bc_o = "outflow",
  },
}

config.output = {
  ncycle_out = 250,
  dt_growth_frac = 1.01,
  history = {
    fn = "ni_decay.hst",
  },
}

config.fluid = {
  limiter = {
    enabled = true,
    tci_opt = false,
    tci_val = 0.1,
    characteristic = false,
  },
}

config.time = {
  t_end = 524880.0,
  nlim = -1,
  integrator = "EX_SSPRK22",
}

config.eos = {
  type = "ideal",
  gamma = 1.4,
}

return config
