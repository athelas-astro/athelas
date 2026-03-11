local config = {}

config.problem = {
  name = "rad_shock",
  t_end = 2.0e-10,
  nlim = -1,
  geometry = "planar",
  xl = 0.0,
  xr = 0.01575,
  x_d = 0.0132,
  cfl = 0.75,
  nx = 256,
  grid_type = "uniform",

  params = {
    vL = 5.190000e+07,
    vR = 1.730000e+07,
    rhoL = 5.69,
    rhoR = 17.1,
    T_L = 2.180000e+06,
    T_R = 7.980000e+06,
  },
}

config.physics = {
  radiation = true,
  gravity = false,
  composition = false,
  ionization = false,
  heating = false,
  engine = false,
}

config.basis = {
  nnodes = 2,
}

config.bc = {
  fluid = {
    bc_i = "outflow",
    bc_o = "outflow",
    dirichlet_values_i = { 0.1757469, 5.190000e+07, 1.616577e+15 },
    dirichlet_values_o = { 0.05847953, 1.730000e+07, 1.137159e+15 },
  },
  radiation = {
    bc_i = "outflow",
    bc_o = "outflow",
    dirichlet_values_i = { 1.708744e+11, 0.0 },
    dirichlet_values_o = { 3.068051e+13, 0.0 },
  },
}

config.output = {
  ncycle_out = 250,
  dt_init_frac = 1.001,
  history = {
    fn = config.problem.name .. ".hst",
  },
}

config.fluid = {
  limiter = {
    do_limiter = true,
    type = "minmod",
    m_tvb = 0.5,
    b_tvd = 0.5,
    tci_opt = true,
    tci_val = 0.45,
    characteristic = false,
  },
}

config.radiation = {
  limiter = {
    type = "minmod",
    m_tvb = 0.0,
    b_tvd = 0.5,
    tci_opt = false,
    tci_val = 0.1,
    characteristic = false,
  },
}

config.time = {
  integrator = "IMEX_PDARS_ESDIRK",
}

config.eos = {
  type = "ideal",
  gamma = 5.0 / 3.0,
}

config.opacity = {
  type = "constant",
  kP = 577.0,
  kR = 577.0,
}

return config
