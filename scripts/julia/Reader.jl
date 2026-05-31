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

  evolved::Array{Float64,3} = fid["/fields/evolved"][:, :, :]
  derived::Array{Float64,3} = fid["/fields/derived"][:, :, :]

  SlopeLimiter::Array{Int64,1} = zeros(Int64, nX)

  state::State = State(time, evolved, derived, SlopeLimiter)
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


#fn = "../bin/athelas_Sod_final.h5"
files = readdir("../build/")
time_array = Float64[]
em_array = Float64[]
@inbounds for fn in files
  n = length(fn)
  if (fn[n-2:end] == ".h5" && fn != "athelas_basis_Sod.h5")
    println(fn)
    data, grid, order = Load_Output( "../build/" * fn )

    fig = Figure()
    tau = data.evolved[1,:,1]
    v = data.evolved[1,:,2]
    em = data.evolved[1,:,3] .- 0.5 .* v .* v
    println(v)
    em1 = em[1]
    #push!( time_array, data.time )
    #push!( em_array, em1 )
    #scatter( fig[1,1], grid.r, 1.0 ./ tau )

    #lines( fig[1,1], grid.r, 1.0 ./ tau )
    #scatter!( fig[1,1], grid.r, 1.0 ./ tau, color="black"  )
    lines( fig[1,1], grid.r, 1.0 ./ tau )
    #scatter!( fig[1,1], grid.r, tau, color="black"  )

    #lines!( fig[1,1], grid.r, v )
    save(fn*".png", fig )
  end
end

#fig = Figure()
#println(em_array)
#lines( fig[1,1], time_array, em_array )
#save("test.png", fig)
