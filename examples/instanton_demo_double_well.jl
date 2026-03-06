ENV["GKSwstype"]=100

using GMAM
using Plots
using LinearAlgebra, MatrixEquations
using Optim

# Plotting params

Base.@kwdef struct Params
    # Plotting grid 
    x_range = 1.5
    y_range = 1.5
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
    line_width::Float32 = 4
    n_path::Int = 1000
    dt::Float64 = 1e-1

    # Saving
    prefix::String = "test"
    dir::String = "plots"
end

cfg = Params(title="Double Well potential", prefix="dbl-well")

# dXt = drift(Xt) dt + sqrt(2 eps) sig(Xt) dt

#############################
##### Multi- Well potential drift
#############################

@info "Testing against an analytic example: Nonlinear potential drift."
const B = [1 0 ; 0 1]
const ϵ = 1e-1

"""
    V(u::Vector{Float64})
Potential with wells at (-1, -1) and (1, 0), biased toward (1, 0).
"""
function V(u::Vector{Float64})
    x, y = u[1], u[2]
    return ((x-1)^2 + y^2) * ((x+1)^2 + (y+1)^2) + ϵ * (x-1)^2
end

"""
    ∇V(u::Vector{Float64})
Gradient vector: [dV/dx, dV/dy]
"""
function ∇V(u::Vector{Float64})
    x, y = u[1], u[2]
    dx = 2*ϵ*(x - 1) + 4*x^3 + 4*x*y^2 + 4*x*y - 2*x - 4*y - 2
    dy = 2*y*((x + 1)^2 + (y + 1)^2) + 2*(y + 1)*(y^2 + (x - 1)^2)
    return [dx, dy]
end

"""
    ∇²V(u::Vector{Float64})
2x2 Hessian matrix
"""
function ∇²V(u::Vector{Float64})
    x, y = u[1], u[2]
    
    h11 = 2*ϵ + 12*x^2 + 4*y^2 + 4*y - 2
    h12 = 8*x*y + 4*x - 4
    h22 = 4*x^2 + 12*y^2 + 12*y + 6
    
    return [h11 h12; 
            h12 h22]
end

# --- In-place versions for performance ---

function ∇V!(dv::Vector{Float64}, u::Vector{Float64})
    x, y = u[1], u[2]
    dv[1] = 2*ϵ*(x - 1) + 4*x^3 + 4*x*y^2 + 4*x*y - 2*x - 4*y - 2
    dv[2] = 2*y*((x + 1)^2 + (y + 1)^2) + 2*(y + 1)*(y^2 + (x - 1)^2)
end

function ∇²V!(ddv::Matrix{Float64}, u::Vector{Float64})
    x, y = u[1], u[2]
    ddv[1,1] = 2*ϵ + 12*x^2 + 4*y^2 + 4*y - 2
    ddv[1,2] = 8*x*y + 4*x - 4
    ddv[2,1] = ddv[1,2]
    ddv[2,2] = 4*x^2 + 12*y^2 + 12*y + 6
end

drift(x::Vector{Float64}) = - ∇V(x)
sig(x::Vector{Float64}) = B 

# Finding stable equilibrium near (1, 0)
function find_minimum(start_u)
    func = Optim.TwiceDifferentiable(V, ∇V!, ∇²V!, start_u)
    res = Optim.optimize(func, start_u, NewtonTrustRegion())

    return res
end

initial_u = [1.0, 0]
results = find_minimum(initial_u)

if Optim.converged(results)
    @info "Minimum found at: ", Optim.minimizer(results)
    @info "Function value: ", Optim.minimum(results)
else
    @info "Optimization failed to converge."
end

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

# min = minimum(potentials)
# custom_levels = [V([x_, x_]) for x_ in x_grid]
# custom_levels = [min-1e-1; custom_levels]
# custom_levels = sort(custom_levels)
# N = length(custom_levels)
sorted = sort(vec(potentials))
N = length(sorted)
idxs = round.(Int, range(1, N, length=cfg.levels))
custom_levels = unique(sorted[idxs])

plt = contour(x_grid, y_grid, (x,y) -> V([x,y]),
        fill=true,
        c = cfg.contour_cmap,
        levels=custom_levels,
        size=cfg.size, 
        colorbar = true,
        title = cfg.title)

quiver!(plt, x_prod, y_prod,
        quiver=(getindex.(cfg.arrow_scale*vector_field/m, 1), 
            getindex.(cfg.arrow_scale * vector_field/m, 2)), 
        color=cfg.arrow_color)

filename = joinpath(cfg.dir, cfg.prefix * "-analytic.svg")
savefig(filename);

@info "Plots saved in $filename"

# Compute the instanton

eq = Optim.minimizer(results)
endpt = [0.0, -0.0]

@info "Computing instanton between $eq and $endpt."
action, phi, ps, lambdas, action_list, iters = Instanton(drift, sig, eq, endpt; N=cfg.n_path, dt=cfg.dt)
@info "... converged in $iters steps. action=$action"

plot!(plt, getindex.(phi,1), getindex.(phi, 2),  
    lw=1.5*cfg.line_width,
    label="",
    lc = :white)
plot!(plt, getindex.(phi,1), getindex.(phi, 2), 
    label="instanton", 
    lw=cfg.line_width,
    line_z = 5* log10.(abs.(lambdas) .+ 1e-10), 
    seriescolor = :magma, 
    colorbar=false)
filename = joinpath(cfg.dir, cfg.prefix * "-instanton.svg")
savefig(plt, filename);
@info "Instanton plot saved in $filename"


action_plt = plot(abs.(action_list .- V(endpt)), 
yaxis = :log10, label="", yticks=cfg.ticks,
 title="abs err between computed and analytic")
filename = joinpath(cfg.dir, cfg.prefix * "-actions.svg")
savefig(filename);
@info "Objective convergence saved in $filename"

lambda_plt = plot(abs.(lambdas) .+ 1e-10)
yaxis!(:log10)
filename = joinpath(cfg.dir, cfg.prefix * "-lambdas.svg")
savefig(filename);
@info "Lambda variables saved in $filename"

# Solve Riccati

@info "Solving Riccati equation"
Rs = Riccati(drift, sig, eq, phi, ps)
@info "Final Matrix: " R1 = Rs[end]
@info "Analytic Matrix: " ∇²V(endpt)

@info "Computing instanton on grid of points"
computed_gradients = Matrix{Vector{Float64}}(undef, cfg.n_grid, cfg.n_grid)
computed_potentials = Matrix{Float64}(undef, cfg.n_grid, cfg.n_grid)

for (i, (x, y)) in enumerate(Iterators.product(x_grid, y_grid))
    try
        V_comp, dV_comp, iters = Potential(drift, sig, eq, [x,y]; N=cfg.n_path, dt=cfg.dt)
        @info "x: $x, y: $y, action_alg: $V_comp, action_real: $V([x,y]), iters: $iters"
        computed_gradients[i] = dV_comp
        computed_potentials[i] = V_comp
    catch e 
        @info "Algorithm seems to be unstable at (x,y) = ($x , $y). "
        @info "Caught error: $e"
    end
end

plt_computed = contour(x_grid, y_grid, computed_potentials', 
        fill=true,
        c=cfg.contour_cmap,
        levels=custom_levels,
        size=cfg.size)
filename = joinpath(cfg.dir, cfg.prefix * "-computed.svg")
savefig(filename);
@info "Contour of the computed potential saved in $filename"

errs = abs.(computed_potentials - potentials) .+ 1e-15
plt_err = heatmap(x_grid, y_grid, errs, zscale=:log10, c=:balance, aspect_ratio=:equal)
filename = joinpath(cfg.dir, cfg.prefix * "-error.svg")
savefig(filename);
@info "Error in computed potential saved in $filename"

grad_errs = norm.(computed_gradients .- potential_gradients) .+ 1e-15
plt_grad_err = heatmap(x_grid, y_grid, grad_errs, zscale=:log10, c=:balance, aspect_ratio=:equal)
filename = joinpath(cfg.dir, cfg.prefix * "-grad-error.svg")
savefig(filename);
@info "Error in computed gradients saved in $filename"

# 

