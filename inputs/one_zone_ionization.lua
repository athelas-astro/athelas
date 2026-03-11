local config = {}

config.problem = {
  name = "one_zone_ionization",
  t_end = 1.0,
  nlim = 2,
  geometry = "planar",
  xl = 0.0,
  xr = 1.0,
  cfl = 0.8,
  nx = 1,
  grid_type = "uniform",

  params = {
    temperature = 20000.0,
    rho = 1.000000e-04,
  },
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
  nnodes = 1,
}

config.composition = {
  ncomps = 3,
}

config.ionization = {
  fn_ionization = "../data/atomic_data_ionization.dat",
  fn_degeneracy = "../data/atomic_data_degeneracy_factors.dat",
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
    do_limiter = false,
    tci_opt = false,
    tci_val = 0.1,
    characteristic = false,
  },
}

config.time = {
  integrator = "EX_SSPRK11",
}

config.eos = {
  type = "ideal",
  gamma = 1.4,
}

return config
