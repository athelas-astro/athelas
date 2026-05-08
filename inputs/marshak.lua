local config = {}

config.problem = {
  name = "marshak",
  t_end = 5.781e-14,
  nlim = -1,
  geometry = "planar",
  xl = 0.0,
  xr = 3.466205e-3,
  cfl = 0.3,
  nx = 512,
  grid_type = "uniform",

  params = {
    v0 = 0.0,
    rho0 = 10.0,
    T0 = 1.0e4,
    epsilon = 1.0,
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

config.time = {
  -- integrator = "IMEX_ARK32_ESDIRK",
  integrator = "IMEX_PDARS_ESDIRK",
  -- integrator = "IMEX_SSPRK11",
}

config.basis = {
  nnodes = 1,
}

config.bc = {
  fluid = {
    bc_i = "outflow",
    bc_o = "outflow",
  },
  radiation = {
    bc_i = "marshak",
    bc_o = "outflow",
    dirichlet_values_i = { 1.1113063385e12, 0.0 },
  },
}

config.output = {
  ncycle_out = 100,
  dt_init_frac = 1.1,
  dt_init = 1.0e-21,
  --   dt_hdf5 = 1.0,
  history = {
    fn = config.problem.name .. ".hst",
  },
}

config.fluid = {
  limiter = {
    do_limiter = true,
    type = "minmod",
    m_tvb = 0.0,
    b_tvd = 1.0,
    tci_opt = false,
    tci_val = 0.1,
    characteristic = false,
  },
}

config.radiation = {
  discretization = "implicit",
  timestep = {
    max_fractional_change_e = 0.01,
    max_change_f = 0.01,
  },
  newton = {
    max_iter = 8,
    tol = 1.0e-10,
  },
  limiter = {
    do_limiter = true,
    type = "minmod",
    m_tvb = 0.0,
    b_tvd = 1.0,
    tci_opt = false,
    tci_val = 0.1,
    characteristic = false,
  },
}

config.eos = {
  type = "marshak",
  gamma = 5.0 / 3.0, -- default: 1.4
}

config.opacity = {
  type = "constant",
  kP = 577.0, -- cm^2 / g
  kR = 577.0, -- cm^2 / g
}

return config
