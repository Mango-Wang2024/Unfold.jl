module SplineFit
using LinearAlgebra, MixedModels, StatsModels
export build_model, fit_reml, predict_response
include("SharedVariance.jl")

"""Native dense ReMat for each smooth, with one tied diagonal variance per component."""
function build_model_terms(y,Xf,Z_blocks)
    n,p=size(Xf)
    fixed_names=Symbol.("fixed_".*string.(1:p))
    entries=Pair{Symbol,Any}[:response=>y]
    append!(entries,[key=>Xf[:,j] for (j,key) in enumerate(fixed_names)])
    fixed=foldl(+,term.(fixed_names);init=term(0))
    random_terms=Any[]
    for (j,Z) in enumerate(Z_blocks)
        group=Symbol("smooth_"*lpad(string(j),length(string(length(Z_blocks))),'0'))
        push!(entries,group=>fill("spline",n))
        columns=Symbol.("z"*string(j)*"_".*string.(1:size(Z,2)))
        append!(entries,[key=>Z[:,k] for (k,key) in enumerate(columns)])
        lhs=foldl(+,term.(columns);init=term(0))
        push!(random_terms,lhs|term(group))
    end
    tbl=NamedTuple{Tuple(first.(entries))}(Tuple(last.(entries)))
    formula=FormulaTerm(term(:response),fixed+sum(random_terms))
    form=MixedModels.schematize(formula,tbl,Dict{Symbol,Any}())
    _,design=StatsModels.modelcols(form,tbl)
    reterms=[SharedVarianceTerm(t) for t in design if t isa MixedModels.ReMat]
    feterm=MixedModels.FeTerm(Xf,String.(fixed_names))
    group_names=[MixedModels.fname(t) for t in reterms]
    return (;feterm,reterms,form,group_names)
end

"""Accept a variable-length list of spline designs, one shared variance per entry."""
function build_model(y::AbstractVector,X::AbstractMatrix,Z_blocks::AbstractVector{<:AbstractMatrix})
    yf,Xf=Vector{Float64}(y),Matrix{Float64}(X)
    Zs=Matrix{Float64}.(Z_blocks)
    n,p=size(Xf)
    isempty(Zs) && throw(ArgumentError("Supply at least one spline component"))
    length(yf)==n && all(size(Z,1)==n for Z in Zs) || throw(DimensionMismatch("All inputs must share observation rows"))
    p>0 && all(size(Z,2)>0 for Z in Zs) && n>p+sum(size(Z,2) for Z in Zs) || throw(ArgumentError("Require fixed and penalized columns and n > p + total penalized columns"))
    all(isfinite,yf) && all(isfinite,Xf) && all(all(isfinite,Z) for Z in Zs) || throw(ArgumentError("Model inputs must be finite"))
    rank(Xf)==p || throw(ArgumentError("Fixed-effect columns must be linearly independent"))
    terms=build_model_terms(yf,Xf,Zs)
    # Keep spline blocks separate and retain their single synthetic grouping level.
    model=MixedModels.LinearMixedModel(yf,terms.feterm,terms.reterms,terms.form,Float64[],nothing,false,1000,false)
    # The constructor sorts terms by size. Preserve the user-facing component order.
    group_names=MixedModels.fname.(model.reterms)
    model_order=[only(findall(==(key),group_names)) for key in terms.group_names]
    return (;model,model_order)
end

"""Joint REML fitting through MixedModels; each component has its own r² and lambda."""
function fit_reml(y::AbstractVector,X::AbstractMatrix,Z_blocks::AbstractVector{<:AbstractMatrix})
    prepared=build_model(y,X,Z_blocks);m=prepared.model
    # Multi-parameter reference checks need tighter objective stopping tolerances.
    m.optsum.ftol_rel=1e-14
    m.optsum.ftol_abs=1e-12
    m.optsum.xtol_abs=fill(1e-10,length(m.θ))
    MixedModels.fit!(m;REML=true,progress=false)
    status=m.optsum.returnvalue
    status in (:SUCCESS,:STOPVAL_REACHED,:FTOL_REACHED,:XTOL_REACHED) || error("MixedModels did not converge: $status")
    theta=[abs(m.reterms[i].λ[1,1]) for i in prepared.model_order]
    sigma2=MixedModels.varest(m)
    r2=sigma2.*theta.^2;r=sqrt.(r2)
    lambda=inv.(theta.^2)
    b=MixedModels.coef(m) # Combined fixed coefficients: b=[β; α₁; ...; αₘ].
    random=MixedModels.ranef(m)
    tau=[vec(random[i]) for i in prepared.model_order]
    fitted=MixedModels.fitted(m)
    return (;prepared,lambda,sigma2,r2,r,theta,b,tau,fitted,status,
             objective=MixedModels.objective(m),evaluations=m.optsum.feval)
end

function predict_response(fit,X,Z_blocks)
    length(Z_blocks)==length(fit.tau) || throw(DimensionMismatch("Component counts disagree"))
    return X*fit.b+sum(Z*coef for (Z,coef) in zip(Z_blocks,fit.tau))
end
end
