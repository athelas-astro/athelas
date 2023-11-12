#!/usr/bin/env julia
"""
Tools for managing Taylor basis functions

Prerequisites:
--------------
FastGaussQuadrature.jl
"""

"""
Compute the n-point Gauss-Legendre quadrature.
Scaled to interval [-0.5, +0.5]
"""
function ComputeQuadrature(nNodes::Int64)
  nodes, weights = gausslegendre(nNodes)
  return nodes / 2.0, weights / 2.0
end


"""
Evaluate Basis
"""
function BasisEval(Basis::BasisType, Data::State, iCF::Int64, iX::Int64,
                   iEta::Int64)
  order::Int64 = Basis.order

  result::Float64 = 0.0
  for k in 1:order
    @inbounds result += Data.uCF[k, iX, iCF] * Basis.Phi[k, iX, iEta]
  end
  return result
end


"""
Return an array of nodal values on the whole domain for a given quantity
"""
function Modal2Nodal(Basis::BasisType, Data::State, iCF::Integer)
  N::Integer = length( Data.uCF[1,:,iCF] ) * Basis.order

  out::Array{Float64,1} = zeros( N )
  for i in 1:length(Data.uCF[1,:,iCF])
    for j in 1:Basis.order
      @inbounds out[(i-1)*Basis.order + j] = BasisEval(Basis, Data, iCF, i, j+1)
    end
  end
  return out
end


"""
Construct a nodal grid
"""
function NodalGrid(Grid::GridType, Nodes::Array{Float64,1})
  function NodeCoordinate( Rad::Array{Float64,1}, Widths::Array{Float64,1}, 
                           eta_q::Float64, iX::Integer )
    return Rad[iX] + Widths[iX] * eta_q
  end
  new_grid::Array{Float64,1} = zeros( length(Grid.r) * length(Nodes) )
  for i in 1:length( Grid.r )
    for j in 1:length(Nodes)
      @inbounds new_grid[(i-1)*length(Nodes) + j] = NodeCoordinate(Grid.r, Grid.dr, Nodes[j], i)
    end
  end
  return new_grid
end


"""
Integrate on an element
Takes the full Data state
"""
function Integrate( Basis::BasisType, Data::State, Weights::Array{Float64, 1}, 
                    iCF::Int64, iX::Int64 )
  result :: Float64 = 0.0
  for iN in 1:length(Weights)
    result += Weights[iN] * BasisEval( Basis, Data, iCF, iX, iN + 1 )
  end
  return result
end


"""
Integrate on an element
takes an array
"""
function Integrate(Data::Array{Float64,1}, Weights::Array{Float64, 1})
  result :: Float64 = 0.0
  for iN in 1:length(Weights)
    result += Weights[iN] * Data[iN]
  end
  return result
end