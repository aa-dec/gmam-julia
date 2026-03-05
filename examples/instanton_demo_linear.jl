ENV["GKSwstype"]=100

using GMAM
using Plots
using LinearAlgebra, MatrixEquations

# Plotting params

Base.@kwdef struct Params
    # Plotting grid 
    x_range = 3
    y_range = 3
    n_grid = 20

    # Level set 
    contour_cmap::Symbol = :viridis
    levels::Int = 15
    fill_contour::Bool = true
    
    # Vector field 
    arrow_color::Symbol = :white
    arrow_scale::Float64 = 0.5
    arrow_head::Float64 = 10
    grid_density::Int = 20
    
    # Layout
    title::String = "Solver Output"
    size::Tuple{Int, Int} = (800, 800)

    # Errors 
    ticks::Vector{Float64} = [10.0^(-i) for i in 0:5]

    # Instanton
    line_width::Float32 = 2
    n_path::Int = 1000
    dt::Float64 = 1e-1

    # Saving
    prefix::String = "test"
    dir::String = "plots"
end

cfg = Params(title="Jordan Block", prefix="linear")

# dXt = drift(Xt) dt + sqrt(2 eps) sig(Xt) dt

#############################
##### Linear drift
#############################

@info "Testing against an analytic example: Linear drift Jordan Block."

# Important that params are const for Enzyme AD
const lam = 0.5
const A = [-lam 1; 0 -lam]
const B = [1 0 ; 0 1]

@debug "The drift matrix: " a=A

drift(x::AbstractArray) = A*x 
sig(x::AbstractArray) = B 

# A cov + cov A' + 2BB' = 0
const cov = lyapc(A, 2*B*B')  
V(x::AbstractArray) = 1/2 * x ⋅ (cov\x)
∇V(x::AbstractArray) = cov\x

x_grid = range(-cfg.x_range, cfg.x_range, length=cfg.n_grid)
y_grid = range(-cfg.y_range, cfg.y_range, length=cfg.n_grid)

x_prod = [x for (x,y) in Iterators.product(x_grid, y_grid)]
y_prod = [y for (x,y) in Iterators.product(x_grid, y_grid)]

vector_field = [drift([x,y]) for (x,y) in Iterators.product(x_grid, y_grid)] # Matrix of vectors
potential_gradients = [∇V([x,y]) for (x,y) in Iterators.product(x_grid, y_grid)]
potentials = [V([x,y]) for (x,y) in Iterators.product(x_grid, y_grid)] 
m = maximum(norm.(vector_field))

# Plot vector_field and potential level sets 

@info "Plotting analytic field + potential"

plt = contour(x_grid, y_grid, (x,y) -> V([x,y]),
        fill=true,
        c=cfg.contour_cmap,
        levels=cfg.levels,
        size=cfg.size)

quiver!( x_prod, y_prod,
        quiver=(getindex.(cfg.arrow_scale*vector_field/m, 1), 
            getindex.(cfg.arrow_scale * vector_field/m, 2)), 
        color=cfg.arrow_color)

title!(cfg.title)
filename = joinpath(cfg.dir, cfg.prefix * "-analytic.svg")
savefig(filename);

@info "Plots saved in $filename"

# Compute the instanton

eq = [0,0]
endpt = [0, 1]

@info "Computing instanton between $eq and $endpt."
action, phi, ps, lambdas, action_list, iters = Instanton(drift, sig, eq, endpt; N=cfg.n_path, dt=cfg.dt)
@info "... converged in $iters steps. action=$action"

plot!(getindex.(phi,1), getindex.(phi, 2), label="instanton", lw=cfg.line_width)
filename = joinpath(cfg.dir, cfg.prefix * "-instanton.svg")
savefig(filename);
@info "Instanton plot saved in $filename"


action_plt = plot(abs.(action_list .- V(endpt)), 
yaxis = :log10, label="", yticks=cfg.ticks,
 title="abs err between computed and analytic")
filename = joinpath(cfg.dir, cfg.prefix * "-actions.svg")
savefig(filename);
@info "Objective convergence saved in $filename"

lambda_plt = plot(lambdas)
filename = joinpath(cfg.dir, cfg.prefix * "-lambdas.svg")
savefig(filename);
@info "Lambda variables saved in $filename"

# Solve Riccati

@info "Solving Riccati equation"
Rs = Riccati(drift, sig, eq, phi, ps)
@info "Final Matrix: " R1 = Rs[end]
@info "Analytic Matrix: " D2V=inv(cov)

# @info "Computing instanton on grid of points"
# computed_gradients = Matrix{Vector{Float64}}(undef, cfg.n_grid, cfg.n_grid)
# computed_potentials = Matrix{Float64}(undef, cfg.n_grid, cfg.n_grid)

# for (i, (x, y)) in enumerate(Iterators.product(x_grid, y_grid))
#     V_comp, dV_comp, iters = Potential(drift, sig, eq, [x,y]; N=cfg.n_path, dt=cfg.dt)
#     @info "x: $x, y: $y, action: $V, iters: $iters"
#     computed_gradients[i] = dV_comp
#     computed_potentials[i] = V_comp
# end

# plt_computed = contour(x_grid, y_grid, computed_potentials', 
#         fill=true,
#         c=cfg.contour_cmap,
#         levels=cfg.levels,
#         size=cfg.size)
# filename = joinpath(cfg.dir, cfg.prefix * "-computed.svg")
# savefig(filename);
# @info "Contour of the computed potential saved in $filename"

# errs = abs.(computed_potentials - potentials) .+ 1e-15
# plt_err = heatmap(x_grid, y_grid, errs, zscale=:log10, c=:balance, aspect_ratio=:equal)
# filename = joinpath(cfg.dir, cfg.prefix * "-error.svg")
# savefig(filename);
# @info "Error in computed potential saved in $filename"

# grad_errs = norm.(computed_gradients .- potential_gradients) .+ 1e-15
# plt_grad_err = heatmap(x_grid, y_grid, grad_errs, zscale=:log10, c=:balance, aspect_ratio=:equal)
# filename = joinpath(cfg.dir, cfg.prefix * "-grad-error.svg")
# savefig(filename);
# @info "Error in computed gradients saved in $filename"

# 

