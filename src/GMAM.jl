"""
    GMAM

A module for computing the instanton and potential value for 
stochastic systems governed by the evolution SDE
dXt = drift(Xt) dt + sqrt{2 eps} sigma(X_t) dWt

Implementation of
 The geometric minimum action method: A least action principle on the space of curves
 - Heymann & Vanden-Eijnden 2008

"""
module GMAM

using DataInterpolations

include("builders.jl")

"""
    Instanton(drift, sigma, stable_eq, x; [N, dt, max_iter, tol])

Computes the instanton via the flow 
∂ₜ ϕ = λ^2 ϕ'' - λ DₚDₓH ϕ' + DₚDₚH DₓH + λ λ' ϕ' + μ ϕ'
where μ is the lagrange multiplier to enforce |ϕ'| = const
In practice we don't compute μ, but simply reparameterize 
ϕ in every update step in dt.

We compute the ∂ₜ ϕ update via forward differencing on 
the second derivative ϕ'':
1/dt ϕ_next(i) - λ^2 ( ϕ_next(i+1) - 2 ϕ_next(i) + ϕ_next(i-1)) / dtau^2
= 1/dt ϕ(i) - λ DₚDₓH ϕ'(i) + DₚDₚH DₓH(i) + λ λ' ϕ'(i) 
ϕ_next(1) = stable_eq
ϕ_next(end) = x

### Arguments
* `drift`: Function representing the drift term of the SDE. Takes in Vector{Float64} of size dim and returns a Vector{Float64} 
* `sigma`: Function or constant representing the noise strength. Takes in Vector{Float64} of size dim and returns a Matrix{Float64} of size dimxdim.
* `stable_eq`: Boundary condition at \$\\tau = 0\$ (the stable equilibrium).
* `x`: Boundary condition at \$\\tau = 1\$ (the target state).

### Keyword Arguments
* `N::Int`: Number of spatial grid points (default: 1000).
* `dt::Float64`: Time step for the flow evolution (default: 0.1).
* `max_iter::Int`: Maximum flow iterations (default: 200).
* `tol::Float64`: Convergence tolerance for the flow (default: 0.001).

### Returns
* `action::Float64`: The computed action
* `ϕ::Vector{Vector{Float64}}`: The computed instanton path as a list of N+1 vectors of size dim
* `ps::Vector{Vector{Float64}}`: The adjoint/momentum variable representing ∇V along the instanton path. A list of N-1 vectors (endpoints left out due to finite differencing).
* `lambdas::Vector{Float64}`: A list of N+1 numbers representing the speed of the instanton in the original time coordinate.
* `action_list::Vector{Float64}`: A list containing the evolution of the action through the algorithm's iteration.  
"""
function Instanton(drift, sigma, stable_eq, x; 
    N=500, dt=0.01, max_iter=200, tol=0.001)

    # building functions

    ham = build_ham(drift, sigma)
    Dx_ham = build_Dₓ_ham(ham)
    DpDx_ham = build_DₚDₓ_ham(ham)
    DpDp_ham = build_DₚDₚ_ham(ham)
    lam = build_λ(drift, sigma)
    theta = build_θ(drift, sigma)

    dim = length(x)

    # initialize the optimization path with linear interpolation
    taus = collect(range(0, 1, N+1))
    phi_0 = [stable_eq*(1-t) + x*t for t in taus]


    #tau parameterizes the path where dt controls the gradient descent step size
    dtau = taus[2] - taus[1] 
    inc = Inf
    iter = 0

    phi = phi_0

    action_list = []

    while (iter < max_iter) && (inc > tol)
        phi_prime = (phi[3:end] - phi[1:end-2])/(2 * dtau)
        ps = theta.(phi[2:end-1], phi_prime)

        # action = 1/2 ∫₀¹ <phi_prime, ps> dtau
        action = 1/2 * dtau*(phi_prime ⋅ ps)
        push!(action_list, action)

        lambdas = lam.(phi[2:end-1], phi_prime)
        # extrapolate
        lambda_1 = 3*lambdas[1] - 3*lambdas[2] + lambdas[3] 
        lambda_end = 3*lambdas[end] - 3*lambdas[end-1] + lambdas[end - 2]
        pushfirst!(lambdas, lambda_1)
        push!(lambdas, lambda_end)
        lambdas_prime = (lambdas[3:end] - lambdas[1:end-2])/(2 * dtau)

        term1 = 1/dt*phi[2:end-1]

        H_xp_list = DpDx_ham.(phi[2:end-1], ps)
        term2 = -lambdas[2:end-1] .* [M' * v for (M, v) in zip(H_xp_list, phi_prime)]

        H_pp_list = DpDp_ham.(phi[2:end-1], ps)
        H_x_list = Dx_ham.(phi[2:end-1], ps)
        term3 = [M *v for (M,v) in zip(H_pp_list, H_x_list)]

        term4 = (lambdas[2:end-1].*lambdas_prime).*phi_prime

        # Construct the (N+1)x(N+1) forward difference matrix
        dg = ones(N+1)
        udg = - lambdas[2:end-1].^2/(dtau^2)
        pushfirst!(udg, 0)
        ldg = - lambdas[2:end-1].^2/(dtau^2)
        push!(ldg, 0)
        dg[2:end-1] = 1/dt .+ 2*lambdas[2:end-1].^2/(dtau^2)

        A = Tridiagonal(ldg, dg, udg)

        # Construct rhs
        B = term1 .+ term2 .+ term3 .+ term4
        pushfirst!(B, stable_eq)
        push!(B, x)
        B = stack(B)' # matrix of size (N+1)xdim

        phi_next = A \ B  # matrix of size (N+1)xdim
        phi_next = [phi_next[i,:] for i in 1:(N+1)]
        
        # Reparameterize based on arc length
        dists = [norm(phi_next[i+1] - phi_next[i]) for i in 1:N]
        path_length = [0; cumsum(dists)]
        path_length = path_length./path_length[end]
        itp = CubicSpline(phi_next, path_length)
        phi_next = itp.(taus)
        inc = sum(norm.(phi_next .- phi)) 

        if (norm(phi_next[1] - stable_eq) > 0.1) || (norm(phi_next[end] - x) > 0.1)
            @warn """Path is drifting away from boundary points. \
            The algorithm might be unstable. Iteration $iter""" increment=inc maxlog=20
        end

        # @debug "Checkpoint. Iteration $iter" increment=inc action=action

        iter = iter + 1
        phi = phi_next

    end

    phi_prime = (phi[3:end] - phi[1:end-2])/(2 * dtau)
    ps = theta.(phi[2:end-1], phi_prime)
    action =1/2 * dtau*(phi_prime ⋅ ps)
    push!(action_list, action)
    lambdas = lam.(phi[2:end-1], phi_prime)
    # extrapolate
    lambda_1 = 3*lambdas[1] - 3*lambdas[2] + lambdas[3] 
    lambda_end = 3*lambdas[end] - 3*lambdas[end-1] + lambdas[end - 2]
    pushfirst!(lambdas, lambda_1)
    push!(lambdas, lambda_end)

    @debug "Took $iter iterates to converge. Difference norm was $inc"

    return action, phi, ps, lambdas, action_list
end

"""
    Potential(drift, sigma, stable_eq, x; [N, dt, max_iter, tol])

Returns the value of the Friedlin - Wentzel potential by computing the instanton.

### Arguments
* `drift`: Function representing the drift term of the SDE. Takes in Vector{Float64} of size dim and returns a Vector{Float64} 
* `sigma`: Function or constant representing the noise strength. Takes in Vector{Float64} of size dim and returns a Matrix{Float64} of size dimxdim.
* `stable_eq`: Boundary condition at \$\\tau = 0\$ (the stable equilibrium).
* `x`: Boundary condition at \$\\tau = 1\$ (the target state).

### Keyword Arguments
* `N::Int`: Number of spatial grid points (default: 1000).
* `dt::Float64`: Time step for the flow evolution (default: 0.1).
* `max_iter::Int`: Maximum flow iterations (default: 200).
* `tol::Float64`: Convergence tolerance for the flow (default: 0.001).

### Returns
* `action::Float64`: The computed action V(x)
* `∇V::Vector{Float64}`: The gradient ∇V(x) 
"""
function Potential(drift, sigma, stable_eq, x, 
    N=1000, dt=0.001, max_iter=1000, tol=1e-8)
    
    action, _, ps, _, _ = Instanton(drift, sigma, 
        stable_eq, x, N=N, dt=dt, max_iter=max_iter, tol=tol)

    return action, ps[end]
end

export Instanton, Potential


end # module GMAM
