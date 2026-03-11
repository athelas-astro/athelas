local msun = 1.989e33 -- g
local foe = 1.0e51 -- erg
local day = 86400.0 -- s

local config = {}

-- Note: The domain bounds xl and xr are overwritten.
config.problem = {
  name = "supernova",
  t_end = 1.0 * day,
  nlim = -1,
  geometry = "spherical",
  xl = 100.0,
  xr = 7.0e5,
  cfl = 0.8,
  nx = 512,
  grid_type = "logarithmic",

  -- Parameters unique to a setup.
  params = {
    fn_hydro = "../profiles/profile_hydro.dat",
    fn_comps = "../profiles/profile_comps.dat",
    mass_cut = 1.4,
    ni_mass = 0.07,
    ni_boundary = 3.0,
  },
}

-- Enable physics units. Their individual controls appear in tables below.
config.physics = {
  radiation = true,
  gravity = true,
  composition = true,
  ionization = true,
  heating = false,
  engine = true,
}

config.time = {
  integrator = "IMEX_SSPRK11",
  -- integrator = "IMEX_SSPRK22_DIRK",
  -- integrator = "IMEX_ARK32_ESDIRK",
  -- integrator = "IMEX_PDARS_ESDIRK",
}

config.basis = {
  nnodes = 1,
}

-- Thermal engine
config.engine = {
  thermal = {
    enabled = true,
    mode = "asymptotic", -- asymptotic or direct
    energy = 1.0 * foe,
    tend = 0.1,
    mend = 0.1,
  },
}

-- Set the number of compositional species. Must match number of columns
-- in the input profile (checked).
config.composition = {
  ncomps = 21,
}

config.ionization = {
  fn_ionization = "../data/atomic_data_ionization.dat",
  fn_degeneracy = "../data/atomic_data_degeneracy_factors.dat",
  ncomps = 1,
  solver = "log", -- linear or log
}

config.bc = {
  fluid = { bc_i = "reflecting", bc_o = "outflow" },
  radiation = { bc_i = "reflecting", bc_o = "outflow" },
}

config.fluid = {
  limiter = {
    do_limiter = true,
    type = "minmod",
    b_tvd = 1.0,
    m_tvb = 0.0,
    tci_opt = false,
    tci_val = 0.25,
  },
}

config.radiation = {
  limiter = {
    type = "minmod",
    m_tvb = 0.0,
    b_tvd = 1.0,
    tci_opt = false,
    tci_val = 0.1,
  },
}

config.heating = {
  nickel = {
    enabled = true,
    model = "jeffery",
    operator_split = true,
  },
}

config.gravity = {
  model = "spherical",
  operator_split = false,
}

config.eos = {
  type = "paczynski",
}

config.opacity = {
  type = "tabular",
  filename = "../data/opac_new.h5",
  floors = {
    type = "core_envelope",
    core_rosseland = 0.24,
    core_planck = 0.24,
    env_rosseland = 0.01,
    env_planck = 0.01,
  },
}

return config
