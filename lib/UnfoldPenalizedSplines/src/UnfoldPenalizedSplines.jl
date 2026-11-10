module UnfoldPenalizedSplines

using DataFrames, LinearAlgebra, StatsModels
import Unfold

# Reuse the research project's spline construction and MixedModels fitter.
include("BSpline.jl")
include("ModelInputs.jl")
include("SplineFit.jl")
using .BSplineBasis
using .ModelInputs: prepare_inputs

include("terms.jl")
include("fit.jl")

export pspl, fit_reml!, smoothing_parameters

end
