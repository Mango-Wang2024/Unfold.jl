module ModelInputs

using LinearAlgebra
export prepare_inputs

const RANK_RTOL = 1e-10

"""
Prepare fixed/random spline columns inside the combined workflow.

Input B is n-by-p, and S is p-by-p. The returned coefficient map C satisfies
    z = C * [tau; alpha]
    B*z = Z*tau + X0*alpha
    z'*S*z = tau'*tau.
"""
function prepare_inputs(B::AbstractMatrix, S::AbstractMatrix)
    p = size(B, 2)
    size(S) == (p, p) || throw(DimensionMismatch("S must be p-by-p for the supplied B"))
    Bf, Sf = Matrix{Float64}(B), Matrix{Float64}(S)
    all(isfinite, Bf) && all(isfinite, Sf) || throw(ArgumentError("B and S must be finite"))
    norm(Sf - Sf') <= RANK_RTOL * max(norm(Sf), eps(Float64)) ||
        throw(ArgumentError("S is not symmetric within tolerance"))
    E = eigen(Symmetric((Sf + Sf') / 2))
    order = sortperm(E.values; rev=true) # R's symmetric eigen() sorts descending.
    values = E.values[order]
    U = E.vectors[:, order]
    tol = RANK_RTOL * maximum(abs, values)
    minimum(values) >= -tol || throw(ArgumentError("S is not positive semidefinite"))
    r = count(>(tol), values)
    scales = vcat(1 ./ sqrt.(values[1:r]), ones(p-r))
    C = U * Diagonal(scales)
    transformed = Bf * C
    return (; C, Z=transformed[:, 1:r], X0=transformed[:, r+1:end],
             penalty_rank=r, null_rank=p-r)
end


end
