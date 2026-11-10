using Unfold, UnfoldPenalizedSplines, DataFrames, DelimitedFiles, Statistics

function read_table(file)
    values, header = readdlm(file, ',', Float64; header=true)
    return DataFrame(values, Symbol.(vec(header)))
end

# The same saved observations and prediction grid as the research project.
training = read_table(joinpath(@__DIR__, "data", "training.csv"))
grid = read_table(joinpath(@__DIR__, "data", "prediction_grid.csv"))
events = select(training, :x1, :x2)
data = reshape(training.y, 1, 1, :) # channel × time point × observation

formula = @formula(0 ~ 1 + pspl(x1, 8) + pspl(x2, 8))
model = fit(UnfoldModel, formula, events, data, [0.0]; fit=false)
fit_reml!(model, data)

# Native Unfold prediction uses the saved spline transformations at new x values.
prediction = vec(only(predict(model, select(grid, :x1, :x2); overlap=true)))
rmse = sqrt(mean(abs2, prediction - grid.truth))
println(smoothing_parameters(model))
println("Grid RMSE against the known additive surface: ", rmse)

# Native coefficient reporting also includes the fitted penalized coefficients.
coefficients = coeftable(model)
