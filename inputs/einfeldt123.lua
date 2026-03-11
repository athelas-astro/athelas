local config = dofile("../inputs/sod.lua")
-- NOTE: This relies on the path. We may want to embed this logic into the C++.

-- Override key pieces
config.problem.xr = 1.0
config.problem.nx = 2048

config.problem.params = {
  vL = -2.0,
  vR = 2.0,
  rhoL = 1.0,
  rhoR = 1.0,
  pL = 0.4,
  pR = 0.4,
  x_d = 0.5,
}

config.basis.nnodes = 3
config.time.integrator = "EX_SSPRK54"

config.fluid.limiter = {
  do_limiter = true,
  type = "minmod",
  m_tvb = 5.0,
  b_tvd = 1.0,
  tci_opt = true,
  tci_val = 0.1,
  characteristic = false,
}

return config
