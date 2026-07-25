-- Small P1 hydrostatic-balance breathing mode for the gravity energy
-- conservation regression test. The velocity perturbation v = A r / R excites a
-- radial breathing mode that keeps the slope limiter active, so the limiter
-- relocates interior nodes and moves the discrete gravitational potential
-- energy W_h. With gravity.limiter_energy_correction enabled, that energy is
-- returned to the fluid and total energy is conserved. Kept small (nx = 64,
-- short t_end) so the test runs in a few seconds.
local config = {}

config.problem = {
  name = "hydrostatic_balance",
  cfl = 0.5,

  params = {
    rho_c = 150.0,
    p_threshold = 0.2,
    velocity_amplitude = 1.0e6,
  },
}

config.mesh = {
  geometry = "spherical",
  nx = 128,
  xl = 0.0,
  xr = 7.000000e+08,
  grid_type = "uniform",
}

config.physics = {
  radiation = false,
  gravity = true,
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
  ncycle_out = 1000,
  dt_growth_frac = 1.05,
  history = {
    fn = "gravity_energy_conservation.hst",
    dt = 0.05,
  },
}

config.fluid = {
  limiter = {
    enabled = true,
    type = "minmod",
    b_tvd = 1.0,
    m_tvb = 0.0,
    tci_opt = false,
    tci_val = 0.25,
  },
}

config.gravity = {
  model = "spherical",
  gval = 1000.0,
  limiter_energy_correction = true,
}

config.time = {
  t_end = 1.0,
  nlim = -1,
  integrator = "EX_SSPRK54",
}

config.eos = {
  type = "polytropic",
  k = 1.000000e+15,
  n = 3.0,
}

return config
