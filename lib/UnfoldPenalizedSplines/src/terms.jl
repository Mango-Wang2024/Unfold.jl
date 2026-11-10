mutable struct PenalizedSplineTerm{T} <: StatsModels.AbstractTerm
    term::T
    k::Int
    knots::Union{Nothing,Vector{Float64}}
    basis::Union{Nothing,NamedTuple}
    prepared::Union{Nothing,NamedTuple}
end

"""
    pspl(x, k)
    pspl(:x, k; knots=nothing)

Cubic B-spline with an integrated squared-second-derivative penalty.
Use `pspl(x, k)` inside `@formula`. The symbol form supports supplied full knots
when constructing a formula programmatically.
"""
function pspl(t::StatsModels.AbstractTerm, k::Integer; knots=nothing)
    supplied = isnothing(knots) ? nothing : Float64.(collect(knots))
    isnothing(supplied) || length(supplied) == k + 4 ||
        throw(ArgumentError("A cubic spline with k basis functions needs k + 4 full knots"))
    return PenalizedSplineTerm(t, Int(k), supplied, nothing, nothing)
end
pspl(t::Symbol, k::Integer; kwargs...) = pspl(term(t), k; kwargs...)

function StatsModels.apply_schema(
    t::StatsModels.FunctionTerm{typeof(pspl)}, sch::StatsModels.Schema, Mod::Type,
)
    return StatsModels.apply_schema(pspl(t.args[1], Int(t.args[2].n)), sch, Mod)
end

function StatsModels.apply_schema(t::PenalizedSplineTerm, sch::StatsModels.Schema, Mod::Type)
    predictor = StatsModels.apply_schema(t.term, sch, Mod)
    predictor isa StatsModels.ContinuousTerm ||
        throw(ArgumentError("pspl requires a continuous predictor"))
    return PenalizedSplineTerm(predictor, t.k, t.knots, t.basis, t.prepared)
end

function StatsModels.modelcols(t::PenalizedSplineTerm, data::NamedTuple)
    x = StatsModels.modelcols(t.term, data)
    if isnothing(t.basis)
        knots = isnothing(t.knots) ? default_knots(x, t.k) : t.knots
        t.basis = build_spline(x, knots)
        t.prepared = prepare_inputs(t.basis.B, t.basis.S)
        return hcat(t.prepared.X0, t.prepared.Z)
    end
    # Prediction reuses the training knots, centering H and coefficient map C.
    transformed = evaluate_basis(t.basis.knots, x) * t.basis.H * t.prepared.C
    r = t.prepared.penalty_rank
    return hcat(transformed[:, r+1:end], transformed[:, 1:r])
end

StatsModels.terms(t::PenalizedSplineTerm) = StatsModels.terms(t.term)
StatsModels.termvars(t::PenalizedSplineTerm) = StatsModels.termvars(t.term)
StatsModels.width(t::PenalizedSplineTerm) = t.k - 1

function StatsModels.coefnames(t::PenalizedSplineTerm)
    prefix = "pspl($(only(StatsModels.termvars(t))),$(t.k))"
    return vcat(prefix * ".fixed", [prefix * ".penalized[$j]" for j in 1:t.k-2])
end
Base.show(io::IO, t::PenalizedSplineTerm) = print(io, "pspl($(t.term), $(t.k))")

spline_terms(t::PenalizedSplineTerm) = [t]
spline_terms(t::StatsModels.MatrixTerm) = reduce(vcat, spline_terms.(t.terms); init=PenalizedSplineTerm[])
spline_terms(t::Unfold.TimeExpandedTerm) = spline_terms(t.term)
spline_terms(t) = PenalizedSplineTerm[]
