local config = {}

config.problem = {
  name = "one_zone_ionization",
  cfl = 0.8,
  params = {
    temperature = 10000.0,
    rho = 2.800000e-09,
  },
}

config.mesh = {
  geometry = "planar",
  nx = 1,
  xl = 0.0,
  xr = 1.0,
  grid_type = "uniform",
}

config.physics = {
  radiation = false,
  gravity = false,
  composition = true,
  ionization = true,
  heating = false,
  engine = false,
}

config.basis = {
  nnodes = 2,
}

config.ionization = {
  fn_ionization = "../../../data/atomic_data_ionization.dat",
  fn_degeneracy = "../../../data/atomic_data_degeneracy_factors.dat",
  ncomps = 3,
}

config.composition = {
  ncomps = 3,
}

config.bc = {
  fluid = {
    bc_i = "outflow",
    bc_o = "outflow",
  },
}

config.output = {
  ncycle_out = 1,
  dt_init_frac = 1.005,
  history = {
    fn = "one_zone_ionization.hst",
  },
}

config.fluid = {
  limiter = {
    enabled = false,
    tci_opt = false,
    tci_val = 0.1,
    characteristic = false,
  },
}

config.time = {
  t_end = 1.0,
  nlim = 2,
  integrator = "EX_SSPRK11",
}

config.eos = {
  type = "ideal",
  gamma = 1.66666666666667,
}

return config
