-- Sedov configuration for the restart regression test.
-- dt_hdf5 is tuned so a midpoint dump (sedov_000001.ath @ t ~ 0.025) lands
-- well before t_end, giving us a checkpoint to restart from.

local config = {}

config.problem = {
  name = "sedov",
  t_end = 0.05,
  nlim = -1,
  geometry = "spherical",
  xl = 0.0,
  xr = 1.0,
  cfl = 0.25,
  nx = 64,
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
  nnodes = 2,
}

config.bc = {
  fluid = {
    bc_i = "reflecting",
    bc_o = "outflow",
  },
}

config.output = {
  ncycle_out = 500,
  dt_hdf5 = 0.025,
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
  integrator = "EX_SSPRK22",
}

config.eos = {
  type = "ideal",
  gamma = 1.4,
}

return config
