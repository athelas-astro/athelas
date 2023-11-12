"""
Functions for computing Equation of State quantities

Prerequisites:
--------------
None
"""

"""
IDEAL
Compute pressure from conserved Lagrangian variables.
"""
function ComputePressure(Tau, Vel, EmT, GAMMA = 1.4)
  Em = EmT - 0.5 * Vel .* Vel
  Ev = Em ./ Tau
  P = (GAMMA - 1.0) .* Ev

  return P
end
