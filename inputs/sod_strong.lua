local config = dofile("../inputs/sod.lua")
-- NOTE: This relies on the path. We may want to embed this logic into the C++.

-- Override key pieces
config.problem.xr = 5.0
config.problem.nx = 256
config.problem.cfl = 0.8
config.problem.params = {
  vL = 0.0,
  vR = 0.0,
  rhoL = 10.0,
  rhoR = 1.0,
  pL = 100.0,
  pR = 1.0,
  x_d = 2.0,
}

config.basis.nnodes = 2
config.time.integrator = "EX_SSPRK54"

return config
