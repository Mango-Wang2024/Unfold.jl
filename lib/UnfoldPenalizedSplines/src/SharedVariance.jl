# Native ReMat storage with one shared scale for every coefficient in a smooth.
# The wrapper delegates matrix operations to the unchanged native container.
struct SharedScaleVector <: AbstractVector{Float64}
    values::Vector{Float64}
end
Base.size(v::SharedScaleVector) = size(v.values)
Base.getindex(v::SharedScaleVector,i::Int) = v.values[i]
function Base.setindex!(v::SharedScaleVector,x,i::Int)
    checkbounds(v.values,i)
    fill!(v.values,x)
    return x
end
Base.IndexStyle(::Type{SharedScaleVector}) = IndexLinear()

struct SharedVarianceTerm{S} <: MixedModels.AbstractReMat{Float64}
    native::MixedModels.ReMat{Float64,S}
    scale::Diagonal{Float64,SharedScaleVector}
    indices::Vector{Int}
end
function SharedVarianceTerm(native::MixedModels.ReMat{Float64,S}) where S
    if S==1
        # MixedModels' scalar kernels expect LowerTriangular.data.
        values=[native.λ[1,1]]
        native.λ=LowerTriangular(reshape(values,1,1))
    else
        MixedModels.zerocorr!(native)
        values=native.λ.diag
    end
    return SharedVarianceTerm(native,Diagonal(SharedScaleVector(values)),[1])
end
function Base.getproperty(A::SharedVarianceTerm,s::Symbol)
    s===:λ && return getfield(A,:scale)
    s===:inds && return getfield(A,:indices)
    s in (:native,:scale,:indices) && return getfield(A,s)
    return getproperty(getfield(A,:native),s)
end
Base.size(A::SharedVarianceTerm) = size(A.native)
Base.getindex(A::SharedVarianceTerm,i::Int,j::Int) = A.native[i,j]
Base.Matrix(A::SharedVarianceTerm) = Matrix(A.native)
MixedModels.getθ(A::SharedVarianceTerm) = [A.λ[1,1]]
MixedModels.nθ(A::SharedVarianceTerm) = 1
MixedModels.nranef(A::SharedVarianceTerm) = MixedModels.nranef(A.native)
MixedModels.nlevs(A::SharedVarianceTerm) = MixedModels.nlevs(A.native)
MixedModels.fname(A::SharedVarianceTerm) = MixedModels.fname(A.native)
MixedModels.reweight!(A::SharedVarianceTerm,w::Vector) = (MixedModels.reweight!(A.native,w); A)
StatsModels.isnested(A::SharedVarianceTerm,B::SharedVarianceTerm) = StatsModels.isnested(A.native,B.native)
Base.:*(A::Adjoint{Float64,<:SharedVarianceTerm},B::SharedVarianceTerm) = A.parent.native'*B.native
Base.:*(A::Adjoint{Float64,<:MixedModels.FeMat},B::SharedVarianceTerm) = A*B.native
LinearAlgebra.mul!(C::AbstractMatrix,A::Adjoint{Float64,<:MixedModels.FeMat},B::SharedVarianceTerm) = mul!(C,A,B.native)
LinearAlgebra.mul!(C::AbstractMatrix,A::Adjoint{Float64,<:MixedModels.FeMat},B::SharedVarianceTerm,a::Number,b::Number) = mul!(C,A,B.native,a,b)
LinearAlgebra.mul!(y::AbstractVector,A::SharedVarianceTerm,b::AbstractVecOrMat,a::Number,c::Number) = mul!(y,A.native,b,a,c)
MixedModels.rmulΛ!(A::AbstractMatrix,B::SharedVarianceTerm) = MixedModels.rmulΛ!(A,B.native)
MixedModels.lmulΛ!(A::Adjoint{Float64,<:SharedVarianceTerm},B::AbstractMatrix) = MixedModels.lmulΛ!(A.parent.native',B)
MixedModels.copyscaleinflate!(L::AbstractMatrix,A::AbstractMatrix,B::SharedVarianceTerm) = MixedModels.copyscaleinflate!(L,A,B.native)
MixedModels.σvals(A::SharedVarianceTerm,sc::Number) = MixedModels.σvals(A.native,sc)
MixedModels.σρs(A::SharedVarianceTerm,sc::Number) = MixedModels.σρs(A.native,sc)
MixedModels.rowlengths(A::SharedVarianceTerm) = MixedModels.rowlengths(A.native)
