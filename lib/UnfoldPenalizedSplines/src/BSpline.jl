module BSplineBasis

using LinearAlgebra, Statistics
import BSplineKit
export default_knots, extend_knots, evaluate_basis, build_spline

const DEGREE = 3

function checked_knots(knots)
    t = Float64.(collect(knots))
    length(t) >= 8 || throw(ArgumentError("A cubic B-spline needs at least eight full knot entries"))
    all(isfinite,t) || throw(ArgumentError("Knots must be finite"))
    all(>(0),diff(t)) || throw(ArgumentError("Full knots must be strictly increasing"))
    return t
end

"""Add three exterior knots on each side of the supplied interval breakpoints."""
function extend_knots(breaks)
    b = Float64.(collect(breaks))
    length(b)>=2 && all(isfinite,b) && all(>(0),diff(b)) ||
        throw(ArgumentError("Use at least two strictly increasing finite breakpoints"))
    left,right = b[2]-b[1],b[end]-b[end-1]
    return vcat([b[1]-j*left for j in 3:-1:1],b,[b[end]+j*right for j in 1:3])
end

"""mgcv bs default: uniform full knots, with a 0.1% data-range expansion."""
function default_knots(x,k::Int)
    xv = Float64.(collect(x))
    !isempty(xv) && all(isfinite,xv) || throw(ArgumentError("Predictors must be nonempty and finite"))
    k>=4 || throw(ArgumentError("Use at least four cubic B-spline basis functions"))
    xl,xu = extrema(xv)
    xr = xu-xl
    xr>0 || throw(ArgumentError("Predictor range must be positive"))
    xl -= 0.001*xr
    xu += 0.001*xr
    step = (xu-xl)/(k-DEGREE)
    return collect(range(xl-DEGREE*step,xu+DEGREE*step;length=k+DEGREE+1))
end

# Full knots already include exterior knots; do not repeat or augment them.
package_basis(t) = BSplineKit.BSplineBasis(BSplineKit.BSplineOrder(DEGREE+1),t;augment=Val(false))

function basis_matrix(basis,x,derivative)
    B = zeros(length(x),length(basis))
    return BSplineKit.collocation_matrix!(B,basis,x,BSplineKit.Derivative(derivative);clip_threshold=0)
end

"""Evaluate cubic B-spline shapes or derivatives, using mgcv's linear prediction tails."""
function evaluate_basis(knots,x;derivative::Int=0)
    t = checked_knots(knots)
    derivative in 0:2 || throw(ArgumentError("Derivative must be 0, 1 or 2"))
    xv = Float64.(collect(x))
    all(isfinite,xv) || throw(ArgumentError("Predictors must be finite"))
    lo,hi = t[4],t[end-3]
    basis = package_basis(t)
    clipped = clamp.(xv,lo,hi)
    B = basis_matrix(basis,clipped,derivative)
    outside = findall(i->xv[i]<lo || xv[i]>hi,eachindex(xv))
    if derivative==0 && !isempty(outside)
        slope = basis_matrix(basis,clipped[outside],1)
        B[outside,:] += (xv[outside]-clipped[outside]).*slope
    elseif derivative==2
        B[outside,:] .= 0
    end
    return B
end

"""Construct cubic B-splines with the integrated squared-second-derivative penalty."""
function build_spline(x,knots)
    t = checked_knots(knots)
    xv = Float64.(collect(x))
    k = length(t)-4
    !isempty(xv) && all(isfinite,xv) || throw(ArgumentError("Predictors must be nonempty and finite"))
    length(unique(xv))>=k || throw(ArgumentError("Need at least k distinct predictor values"))
    all(v->t[4]<=v<=t[end-3],xv) || throw(ArgumentError("Training predictors must lie inside the middle knot interval"))
    # Training values and knots are checked above; reuse one package basis.
    basis = package_basis(t)
    B_raw = basis_matrix(basis,xv,0)
    D2 = basis_matrix(basis,xv,2)
    # Galerkin integration covers [t[4],t[end-3]], matching mgcv's penalty domain.
    # Both factors are second derivatives: S_raw[i,j] = integral(b_i'' * b_j'').
    second = BSplineKit.Derivative(2)
    S_raw = Matrix(BSplineKit.galerkin_matrix(basis,(second,second),Matrix{Float64}))
    penalty_scale = opnorm(S_raw,1)/opnorm(B_raw,Inf)^2
    S_scaled = S_raw/penalty_scale
    constraint = vec(mean(B_raw;dims=1))
    # Preserve the existing mean-zero constraint and R QR sign convention.
    v = copy(constraint)
    v[1] += copysign(norm(constraint),constraint[1])
    Q = Matrix{Float64}(I,k,k)-2*(v*v')/dot(v,v)
    H = Q[:,2:end]
    B = B_raw*H
    S = Matrix(Symmetric(H'*S_scaled*H))
    return (;knots=t,D2,B_raw,S_raw,S_scaled,penalty_scale,constraint,H,B,S)
end

end
