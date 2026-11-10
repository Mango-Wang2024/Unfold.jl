# UnfoldPenalizedSplines.jl

An additive penalized-spline integration for Unfold. Each predictor has its own cubic spline and REML-estimated smoothing strength. The existing Unfold source functions are unchanged.

## Workflow

1. **Construct:** `pspl(x, k)` calls the BSplineKit-based construction and prepares the unpenalized columns $X_{0,j}$ and penalized columns $Z_j$ from $B_j,S_j$.
2. **Assemble:** Unfold combines these columns with ordinary predictors and, for continuous data, applies its existing time expansion.
3. **Fit:** `fit_reml!` supplies a solver adapter through Unfold's existing `solver(X, data)` interface. The adapter selects $X_f,Z_1,\ldots,Z_m$ and calls the existing MixedModels REML fitter.
4. **Return:** coefficients are placed in Unfold's column order, so native `predict` and `coeftable` work. `smoothing_parameters` returns each fitted $\lambda_j=\sigma^2/r_j^2$ and its variance estimates.

Training knots, centering and coefficient transformations are retained for new-data prediction. The number of predictors is determined by the formula. Supplied full knot vectors can be used with `pspl(:x, k; knots=...)` in programmatically constructed formulas.

Channels and epoched time points are fitted separately. In continuous models, each spline term has one shared smoothing strength across its time-expanded columns. This applies the predictor-spline penalty; regularization along the TRF time axis is a separate modelling step.

## Example

From this fork's root, use Julia 1.11 or later:

```sh
julia --project=lib/UnfoldPenalizedSplines -e 'using Pkg; Pkg.instantiate()'
julia --project=lib/UnfoldPenalizedSplines lib/UnfoldPenalizedSplines/examples/two_predictors.jl
```

The example uses the research project's 200 saved observations and 961 prediction points for $f(x_1,x_2)=\sin(1.7x_1)+0.15x_1+0.5\cos(1.2x_2)$. It builds an Unfold model without fitting, then calls:

```julia
formula = @formula(0 ~ 1 + pspl(x1, 8) + pspl(x2, 8))
model = fit(UnfoldModel, formula, events, data, [0.0]; fit=false)
fit_reml!(model, data)
smoothing_parameters(model)
```

As in ordinary Unfold formulas, `0` on the left is a placeholder; the responses are supplied in `data`.

The bundled example retains the original grid RMSE of **0.0794**. Across the research project's five saved cases, the largest absolute difference from its R-reference grid predictions is **1.2 × 10⁻⁸**. Native Unfold prediction and coefficient reporting are used in these comparisons.

## Source organization

- `terms.jl`: the formula term, training transformations and numerical spline columns.
- `fit.jl`: column selection, the custom solver callback and Unfold-compatible results.
- `BSpline.jl`, `ModelInputs.jl`, `SplineFit.jl`, `SharedVariance.jl`: the existing construction, preparation and fitting calculations reused from [Penalized-Splines](https://github.com/Mango-Wang2024/Penalized-Splines/tree/4d8d6e2159bd7a571f32de7b0561f8482f6cf5f4).

The fitter retains the research project's pinned BSplineKit 0.19.3 and MixedModels 5.9.0 versions. R is used for the original reference comparisons, rather than as a runtime dependency.

Sources: [StatsModels formula extensions](https://juliastats.org/StatsModels.jl/stable/internals/#Extending-@formula-syntax), [Unfold spline terms](../../ext/UnfoldBSplineKitExt/splinepredictors.jl), [Unfold custom solver interface](../../src/fit.jl), and [mgcv smooth-to-random preparation](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/smooth2random.html).
