local config = dofile("../inputs/sod.lua")
-- NOTE: This relies on the path. We may want to embed this logic into the C++.

-- Override key pieces
config.problem.params = {
  vL = 0.0,
  vR = 0.0,
  rhoL = 1.0,
  rhoR = 1.0e-3,
  pL = 0.066666667,
  pR = 0.666666667e-10,
  x_d = 3.0,
}

config.basis.nnodes = 2
config.time.integrator = "EX_SSPRK54"
config.eos.gamma = 5.0 / 3.0

return config
