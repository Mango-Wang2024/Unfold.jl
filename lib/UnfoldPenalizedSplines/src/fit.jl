struct REMLAdapter
    fixed::Vector{Int}
    random::Vector{Vector{Int}}
    components::Vector{String}
end

function REMLAdapter(form::StatsModels.FormulaTerm)
    names = StatsModels.coefnames(form.rhs)
    # Unfold labels expanded columns as "event : predictor column : lag".
    base_names = form.rhs isa Unfold.TimeExpandedTerm ?
        [split(name, " : ")[2] for name in names] : names
    random = Vector{Int}[]
    components = String[]
    for t in spline_terms(form.rhs)
        penalized = Set(StatsModels.coefnames(t)[2:end])
        push!(random, findall(in(penalized), base_names))
        push!(components, string(only(StatsModels.termvars(t))))
    end
    fixed = setdiff(collect(eachindex(names)), reduce(vcat, random; init=Int[]))
    return REMLAdapter(fixed, random, components)
end

function (adapter::REMLAdapter)(X, data::AbstractArray{T,N}) where {T,N}
    # Ordinary event designs can continue to use Unfold's existing solver.
    isempty(adapter.random) && return Unfold.solver_default(X, data; show_progress=false)
    Xf = Matrix{Float64}(X[:, adapter.fixed])
    Zs = [Matrix{Float64}(X[:, columns]) for columns in adapter.random]
    channels = size(data, 1)
    times = N == 3 ? size(data, 2) : 1
    shape = N == 3 ? (channels, times, size(X, 2)) : (channels, size(X, 2))
    coefficients = zeros(Float64, shape)
    fits = Matrix{NamedTuple}(undef, channels, times)
    for ch in 1:channels, t in 1:times
        y = N == 3 ? view(data, ch, t, :) : view(data, ch, :)
        fitted = SplineFit.fit_reml(y, Xf, Zs)
        # Put b and tau back into Unfold's actual column order for native predict().
        combined = zeros(size(X, 2))
        combined[adapter.fixed] = fitted.b
        for (columns, tau) in zip(adapter.random, fitted.tau)
            combined[columns] = tau
        end
        if N == 3
            coefficients[ch, t, :] = combined
        else
            coefficients[ch, :] = combined
        end
        fits[ch, t] = (; lambda=fitted.lambda, r2=fitted.r2, sigma2=fitted.sigma2,
                        objective=fitted.objective, status=fitted.status)
    end
    info = (; method=:REML, components=adapter.components, fits)
    return Unfold.LinearModelFit{Float64,N}(; estimate=coefficients, info)
end

"""
    fit_reml!(uf, data)

Fit a prepared Unfold model containing `pspl` terms. Each spline term has one
shared variance and smoothing strength. Channels and epoched time points are
fitted separately. Uses Unfold's existing `solver(X, data)` callback.
"""
function fit_reml!(uf::Unfold.UnfoldLinearModel, data::AbstractArray)
    data = Unfold.check_data(typeof(uf), data)
    adapters = IdDict(Unfold.modelmatrix(d) => REMLAdapter(d.formula)
                      for d in Unfold.designmatrix(uf))
    Unfold.fit!(uf, data; solver=(X, y) -> adapters[X](X, y))
    return uf
end

function fit_reml!(uf::Unfold.UnfoldLinearModelContinuousTime, data::AbstractArray)
    data = Unfold.check_data(typeof(uf), data)
    fixed = Int[]
    random = Vector{Int}[]
    components = String[]
    offset = 0
    for d in Unfold.designmatrix(uf)
        adapter = REMLAdapter(d.formula)
        append!(fixed, adapter.fixed .+ offset)
        append!(random, [columns .+ offset for columns in adapter.random])
        append!(components, adapter.components)
        offset += size(Unfold.modelmatrix(d), 2)
    end
    adapter = REMLAdapter(fixed, random, components)
    Unfold.fit!(uf, data; solver=adapter)
    return uf
end

"""
    smoothing_parameters(uf)

Return the fitted lambda, spline variance and noise variance for each spline,
channel and epoched time point. For a time-expanded fit, each spline shares one
lambda across its lag columns; `time` is missing in the returned table.
"""
function smoothing_parameters(uf::Unfold.UnfoldModel)
    info = Unfold.modelfit(uf).info
    blocks = info isa NamedTuple ? [info] : info
    continuous = uf isa Unfold.UnfoldLinearModelContinuousTime
    rows = NamedTuple[]
    for (event, block) in enumerate(blocks)
        block isa NamedTuple && block.method == :REML || continue
        for ch in axes(block.fits, 1), t in axes(block.fits, 2), j in eachindex(block.components)
            fitted = block.fits[ch, t]
            time = continuous ? missing : last(last(Unfold.design(uf)[event]))[t]
            push!(rows, (; event=continuous ? "combined" : string(first(Unfold.design(uf)[event])),
                          channel=ch, time, predictor=block.components[j],
                          lambda=fitted.lambda[j], r2=fitted.r2[j], sigma2=fitted.sigma2))
        end
    end
    return DataFrame(rows)
end
