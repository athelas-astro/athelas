local config = {}

config.problem = {
  name = "ni_decay",
  t_end = 52488.0,
  nlim = 100,
  geometry = "spherical",
  restart = false,
  xl = 0.0,
  xr = 3.622000e+09,
  cfl = 0.5,
  nx = 64,
  grid_type = "uniform",

  params = {
    temperature = 1000.0,
    rho = 1000.0,
  },
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
    model = "full_trapping",
  },
}

config.bc = {
  fluid = {
    bc_i = "reflecting",
    bc_o = "outflow",
  },
}

config.output = {
  ncycle_out = 100,
  dt_init_frac = 1.01,
  history = {
    fn = "ni_decay.hst",
  },
}

config.fluid = {
  limiter = {
    do_limiter = true,
    tci_opt = false,
    tci_val = 0.1,
    characteristic = false,
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
