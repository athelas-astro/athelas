#!/usr/bin/env julia
"""
Simple reader routines for loading output data.

Prerequisites:
--------------
HDF5.jl
FastGaussQuadrature.jl
"""

using HDF5
using CairoMakie
using FastGaussQuadrature

include("Structures.jl")
include("Basis.jl")

"""
Simple load routine for Athelas data.
"""
function Load_Output( fn::String )
  println(fn)
  fid = h5open(fn, "r")

  # Time
  time::Float64 = fid["/Metadata/Time"][1]
  order::Int64 = fid["/Metadata/Order"][1]
  nX::Int64 = fid["/Metadata/nX"][1]

  x1::Array{Float64,1} = fid["/Spatial Grid/Grid"][:]
  dr::Array{Float64,1} = fid["/Spatial Grid/Widths"][:]

  uCF::Array{Float64,3} = zeros(order, nX, 3)
  uPF::Array{Float64,3} = zeros(order, nX, 3)

  for i in 1:order
    uCF[i, :, 1] = fid["/Conserved Fields/Specific Volume"][((i - 1) * nX + 1):(i * nX)]
    uCF[i, :, 2] = fid["/Conserved Fields/Velocity"][((i - 1) * nX + 1):(i * nX)]
    uCF[i, :, 3] = fid["/Conserved Fields/Specific Internal Energy"][((i - 1) * nX + 1):(i * nX)]

    uPF[i, :, 1] = fid["/Primitive Fields/Specific Volume"][((i - 1) * nX + 1):(i * nX)]
    uPF[i, :, 2] = fid["/Primitive Fields/Velocity"][((i - 1) * nX + 1):(i * nX)]
    uPF[i, :, 3] = fid["/Primitive Fields/Specific Internal Energy"][((i - 1) * nX + 1):(i * nX)]
  end

  SlopeLimiter::Array{Int64,1} = fid["/Diagnostic Fields/Limiter"][:]

  uAF::Array{Float64,3} = zeros(order, nX, 3)

  state::State = State(time, uCF, uPF, uAF, SlopeLimiter)
  grid ::GridType = GridType(x1,dr) 

  close(fid)

  return state, grid, order
end

"""
Load athelas basis
"""
function LoadBasis( fn::String )
  println(fn)
  fid = h5open(fn, "r")

  data::Array{Float64,3} = permutedims(fid["Basis"][:, :, :], [1, 3, 2])

  Basis = BasisType(order, data)

  close(fid)

  return Basis
end

fn = "../bin/athelas_Sod_final.h5"
data, grid, order = Load_Output( fn )


fig = Figure()
scatter( fig[1,1], grid.r, 1.0 ./ data.uPF[1,:,1] )
scatter!( fig[1,1], grid.r, 1.0 ./ data.uCF[1,:,1] )
save("sod.png", fig )
