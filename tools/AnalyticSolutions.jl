#!/usr/bin/env julia
"""
Functions for plotting analytic solutions to test problems

Prerequisites:
--------------
None
"""

using NLsolve

"""
SmoothFlow problem
"""
function SmoothFlow( r::Float64, t::Float64 )

  a = 0.999999999999999999999999999999995
  function f!(F, x)
    F[1] = x[1] - sqrt(3) * (1.0 - a * sin(pi * (r - x[1]*t)))
    F[2] = x[2] + sqrt(3) * (1.0 - a * sin(pi * (r - x[2]*t)))
  end
  function j!(J,x)
    J[1,1] = 1.0 - sqrt(3) * pi * a * t * cos( pi * (r - x[1]*t) )
    J[1,2] = 0.0
    J[2,1] = 0.0
    J[2,2] = 1.0 + sqrt(3) * pi * a * t * cos( pi * (r - x[2]*t) )
  end
  return nlsolve(f!, j!, [-1.0, 1.0]).zero
end


"""
SmoothFlow problem (array)
"""
function SmoothFlow(r::Array{Float64,1}, t::Float64)
  N::Integer = length(r)
  sol::Array{Float64,1} = zeros(N)
  a::Float64 = 0.0
  b::Float64 = 0.0
  for i in 1:N
    a, b = SmoothFlow(- r[i], t)
    @inbounds sol[i] = - ( a + b ) / 2.0
  end
  return sol
end

"""
Shockless Noh problem. Assuming a domain of [0,1] and Gamma = 5/3
Just return density.
"""
function ShocklessNoh(time::Float64, X::Array{Float64,1}, n_points::Int64)
  beta::Float64 = 1.0
  gamma::Float64 = 5.0 / 3.0

  # Setup domain
  x_l::Float64 = 0.0
  x_r::Float64 = 1.0

  X0::Array{Float64,1} = X ./ (1.0 - time)

  V::Array{Float64,1} = -X0

  Em::Float64 = 1.0 * (1.0 - time)^(-beta * (gamma - 1.0))
  Rho::Float64 = 1.0 * (1.0 - time)^(-beta)

  Em_T::Array{Float64,1} = Em .* ones(n_points) .+ 0.5 .* V .* V

  return Rho
end

"""
Sod shock tube solution. Assuming typical perfect gas, gamma = 1.4, 
domain [0,1] with discontinuity at 0.5.
For now: evaluated at t=0.2.
Returns density at given radius.
--- (https://pypi.org/project/sodshock/)
Positions:
Shock      : 0.8504311464060357
Contact Discontinuity : 0.6854905240097902
Head of Rarefaction : 0.26335680867601535
Foot of Rarefaction : 0.4859454374877634
Regions:
Region 1   : (1, 1, 0)
Region 2   : RAREFACTION
Region 3   : (0.30313017805064707, 0.42631942817849544, 0.92745262004895057)
Region 4   : (0.30313017805064707, 0.26557371170530725, 0.92745262004895057)
Region 5   : (0.1, 0.125, 0.0)
"""
function Sod(rad::Float64)
  x_mid::Float64 = 0.5
  x_shock = 0.8504311464060357
  x_rare_left::Float64 = 0.26335680867601535
  x_rare_right::Float64 = 0.4859454374877634
  x_contact::Float64 = 0.6854905240097902

  gamma::Float64 = 1.4

  if (rad < x_rare_left) # I
    return 1.0
  elseif (rad > x_rare_left && rad <= x_rare_right) # II
    c::Float64 = sqrt(1.4)
    t::Float64 = 0.2
    u2::Float64 = (2.0 / (gamma + 1.0)) * (c + (rad - x_mid) / t)
    rho2::Float64 = 1.0 *
                    (1 - ((gamma - 1.0) / 2.0) * u2 / c)^(2.0 / (gamma - 1.0))
    P2::Float64 = 1.0 *
                  (1 - ((gamma - 1.0) / 2.0) * u2 / c)^(2.0 * gamma /
                                                        (gamma - 1.0))
    return rho2
  elseif (rad > x_rare_right && rad <= x_contact) # III
    return 0.42631942817849544
  elseif (rad >= x_contact && rad <= x_shock) # IV
    return 0.26557371170530725
  else # V
    return 0.125
  end
end

# x = LinRange(-1.0, 1.0, 100 )
# sol = zeros(100)

# for i in 1:100
  # a, b = SmoothFlow( -x[i], 0.1 )
  # sol[i] =  -(a .+ b) ./ 2
  # println(x[i], " ", sol[i])
# end

# a,b = SmoothFlow( 0.0, 0.1 )
# println(a, " ", b)
# println((a .+ b) ./ 2)
# using PyPlot
# pygui(:qt5)

# fig, ax = subplots()
# ax.plot( x, sol )
# show()