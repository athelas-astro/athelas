local config = {}

-- Polytrope parameters
local rho_c = 150.0
local poly_k = 1.0e15
local poly_n = 1.5

-- Central pressure of a polytrope, P = K rho^(1 + 1/n).
local function central_pressure(k, n, rho)
  return k * rho ^ (1.0 + 1.0 / n)
end

-- Set the surface pressure as a fixed fraction of the central pressure rather
-- than an absolute number, so the surface stays resolved (and above the EOS
-- floor) if the star is retuned. p_threshold = P_c / ratio: a ratio of ~1e5
-- gives a ~1e3 density contrast with the surface at r/R ~ 0.99; larger ratios
-- steepen the surface (a harder limiter test) and eventually push it toward the
-- floor.
local p_central_to_surface = 1.0e6
local p_threshold = central_pressure(poly_k, poly_n, rho_c) / p_central_to_surface

config.problem = {
  name = "hydrostatic_balance",
  cfl = 0.5,

  params = {
    rho_c = rho_c,
    p_threshold = p_threshold,
  },
}

-- The mesh bound xr is rebuilt by the problem generator.
config.mesh = {
  geometry = "spherical",
  nx = 1024,
  xl = 0.0,
  xr = 7.0e+08,
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
  nnodes = 3,
}

config.bc = {
  fluid = {
    bc_i = "reflecting",
    bc_o = "surface",
    surface_pressure_o = p_threshold,
  },
}

config.output = {
  ncycle_out = 100,
  dt_growth_frac = 1.05,
  history = {
    fn = "hydrostatic_balance.hst",
  },
}

config.fluid = {
  limiter = {
    enabled = false,
    type = "minmod",
    b_tvd = 1.0,
    m_tvb = 10.0,
    tci_opt = true,
    tci_val = 0.25,
  },
}

config.gravity = {
  model = "spherical",
  gval = 1000.0,
  limiter_energy_correction = true,
}

config.time = {
  t_end = 300.0,
  nlim = -1,
  integrator = "EX_SSPRK54",
}

config.eos = {
  type = "polytropic",
  k = poly_k,
  n = poly_n,
}

return config
