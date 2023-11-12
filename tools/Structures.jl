#!/usr/bin/env julia
"""
Structures for holding output data

Prerequisites:
--------------
None
"""

"""
Holds the similation state at a given time.
"""
struct State
  time::Float64

  uCF::Array{Float64,3}
  uPF::Array{Float64,3}
  uAF::Array{Float64,3}

  SlopeLimiter::Array{Int64}
end


"""
Holds the polynomial basis
"""
struct BasisType
  order::Int64

  Phi::Array{Float64,3}
end


"""
Hold grid data
"""
struct GridType
  r::Array{Float64,1}
  dr::Array{Float64,1}
end