-- One-zone ionization configuration for the restart regression test.
-- Uses output.dt_fixed so dump cadence is deterministic w.r.t. cycle count,
-- and nlim governs the runtime so the test is fast and predictable.

local config = {}

config.problem = {
  name = "one_zone_ionization",
  cfl = 0.8,
  params = {
    temperature = 10000.0,
    rho = 2.800000e-09,
  },
}

config.mesh = {
  geometry = "planar",
  nx = 1,
  xl = 0.0,
  xr = 1.0,
  grid_type = "uniform",
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
  nnodes = 2,
}

config.ionization = {
  fn_ionization = "../../../data/atomic_data_ionization.dat",
  fn_degeneracy = "../../../data/atomic_data_degeneracy_factors.dat",
  ncomps = 3,
  solver = "linear",
}

config.composition = {
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
  -- Fixed dt removes CFL surprises; midpoint dump lands at cycle 10.
  dt_fixed = 1.0e-4,
  dt_hdf5 = 1.0e-3,
}

config.fluid = {
  limiter = {
    enabled = false,
    tci_opt = false,
    tci_val = 0.1,
    characteristic = false,
  },
}

config.time = {
  t_end = 0.005,
  nlim = 20,
  integrator = "EX_SSPRK11",
}

config.eos = {
  type = "ideal",
  gamma = 1.66666666666667,
}

return config
