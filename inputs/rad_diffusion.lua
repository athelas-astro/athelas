local config = {}

config.problem = {
  name = "rad_diffusion_wave",
  t_end = 2.0e-8,
  nlim = -1,
  geometry = "planar",
  xl = 0.0,
  xr = 1.0,
  cfl = 0.1,
  nx = 256,
  grid_type = "uniform",

  params = {
    profile = "diffusion_mode",
    v0 = 0.0,
    rho = 1.0e5,
    amp = 1.0,
    perturbation = 1.0e-3,
    mode = 1,
    T_gas = 1.0e6,
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
  -- integrator = "IMEX_SSPRK11",
  integrator = "IMEX_PDARS_ESDIRK",
}

config.basis = {
  nnodes = 2,
}

config.bc = {
  fluid = {
    bc_i = "outflow",
    bc_o = "outflow",
  },
  radiation = {
    bc_i = "reflecting",
    bc_o = "reflecting",
  },
}

config.output = {
  dt_hdf5 = config.problem.t_end / 10000.0,
  ncycle_out = 10,
  dt_growth_frac = 1.2,
  dt_init = 1.0e-10,
}

config.fluid = {
  limiter = {
    enabled = true,
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
  ap_coefficient = 0.0,
  timestep = {
    max_fractional_change_e = 0.2,
    max_change_f = 0.2,
    energy_change_scale = 1.0e-12,
  },
  newton = {
    max_iter = 10,
    tol = 1.0e-10,
  },
  limiter = {
    enabled = true,
    type = "minmod",
    m_tvb = 0.0,
    b_tvd = 1.0,
    tci_opt = false,
    tci_val = 0.1,
    characteristic = false,
  },
}

config.eos = {
  type = "ideal",
  gamma = 5.0 / 3.0,
}

config.opacity = {
  type = "constant",
  kP = 0.0,
  kR = 6400.0,
}

return config
