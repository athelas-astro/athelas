local config = {}

config.problem = {
  name = "sedov",
  t_end = 0.05,
  nlim = -1,
  geometry = "spherical",
  restart = false,
  xl = 0.0,
  xr = 1.0,
  cfl = 0.25,
  nx = 128,
  grid_type = "uniform",

  params = {
    v0 = 0.0,
    rho0 = 1.0,
    E0 = 0.5,
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
  nnodes = 3,
}

config.bc = {
  fluid = {
    bc_i = "reflecting",
    bc_o = "outflow",
  },
}

config.output = {
  ncycle_out = 100,
  dt_hdf5 = 0.05,
  dt_init_frac = 1.05,
}

config.fluid = {
  limiter = {
    do_limiter = true,
    type = "minmod",
    b_tvd = 1.0,
    m_tvb = 0.0,
    tci_opt = true,
    tci_val = 0.1,
    characteristic = false,
    gamma_i = 0.998,
    weno_r = 2.0,
  },
}

config.time = {
  integrator = "EX_SSPRK33",
}

config.eos = {
  type = "ideal",
  gamma = 1.4,
}

return config
