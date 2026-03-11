local config = {}

config.problem = {
  name = "sedov",
  t_end = 0.05,
  nlim = -1,
  geometry = "spherical",
  xl = 0.0,
  xr = 1.0,
  cfl = 0.25,
  nx = 256,
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
  dt_hdf5 = 0.001,
  dt_init_frac = 1.05,
  history = {
    fn = "sedov.hst",
  },
}

config.fluid = {
  limiter = {
    do_limiter = true,
    type = "minmod",
    b_tvd = 1.0,
    m_tvb = 0.0,
    tci_opt = true,
    tci_val = 0.25,
    characteristic = false,
    gamma_i = 0.9,
    weno_r = 2.0,
  },
}

config.gravity = {
  model = "constant",
  gval = 1.0,
}

config.time = {
  integrator = "EX_SSPRK22",
}

config.eos = {
  type = "ideal",
  gamma = 5.0 / 3.0,
}

return config
