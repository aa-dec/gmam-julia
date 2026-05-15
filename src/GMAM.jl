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
using LinearAlgebra, MatrixEquations
using DifferentialEquations

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
* `iters::Int`: Number of Iteration steps took
"""
function Instanton(drift, sigma, stable_eq, x; 
    N=200, dt=0.1, max_iter=1000, tol=1e-3)

    # building functions

    ham = build_ham(drift, sigma)
    Dx_ham = build_Dₓ_ham(ham)
    DpDx_ham = build_DₚDₓ_ham(ham)
    DpDp_ham = build_DₚDₚ_ham(ham)
    lam = build_λ(drift, sigma)
    theta = build_θ(drift, sigma)

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

        iter % 10 ==0 && @debug "Iteration $iter, Max Hamiltonian: " ham=maximum(ham.(phi[2:end-1], ps)) maxlog=10

        # action = ∫₀¹ <phi_prime, ps> dtau
        action = dtau*(phi_prime ⋅ ps)
        push!(action_list, action)

        lambdas = lam.(phi[2:end-1], phi_prime)
        # extrapolate
        lambda_1 = 3*lambdas[1] - 3*lambdas[2] + lambdas[3] 
        lambda_end = 3*lambdas[end] - 3*lambdas[end-1] + lambdas[end - 2]
        pushfirst!(lambdas, lambda_1)
        push!(lambdas, lambda_end)
        lambdas_prime = (lambdas[3:end] - lambdas[1:end-2])/(2 * dtau)

        term1 = 1/dt .* phi[2:end-1]

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
        inc = sum((norm.(phi_next .- phi))) 

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
    action = dtau*(phi_prime ⋅ ps)
    push!(action_list, action)
    lambdas = lam.(phi[2:end-1], phi_prime)
    # extrapolate
    lambda_1 = 3*lambdas[1] - 3*lambdas[2] + lambdas[3] 
    lambda_end = 3*lambdas[end] - 3*lambdas[end-1] + lambdas[end - 2]
    pushfirst!(lambdas, lambda_1)
    push!(lambdas, lambda_end)

    @debug "Took $iter iterates to converge. Difference norm was $inc" maxlog=10

    return action, phi, ps, lambdas, action_list, iter
end

"""
    Potential(drift, sigma, stable_eq, x; [N, dt, max_iter, tol])

Returns the value of the Friedlin - Wentzel potential by computing the instanton.
Wraps Instanton to only output relevant variables.

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
* `iters::Int`: Number of Iteration steps took
"""
function Potential(drift, sigma, stable_eq, x; 
    N=200, dt=0.1, max_iter=1000, tol=1e-4)
    
    action, _, ps, _, _, iters = Instanton(drift, sigma, 
        stable_eq, x; N=N, dt=dt, max_iter=max_iter, tol=tol)

    return action, ps[end], iters
end

function Riccati(drift, sigma, stable_eq, phis, thetas)
    N = length(phis)-1
    taus = collect(range(0, 1, N+1))
    stable_eq = Float64.(stable_eq)

    ham = build_ham(drift, sigma)
    lam = build_λ(drift, sigma)
    dtau = taus[2] - taus[1]

    phis_prime = (phis[3:end] - phis[1:end-2])/(2 * dtau)
    lambdas = lam.(phis[2:end-1], phis_prime)

    # Second order derivatives
    DpDx_ham = build_DₚDₓ_ham(ham)
    DpDp_ham = build_DₚDₚ_ham(ham)
    DxDx_ham = build_DₓDₓ_ham(ham)

    # Initial Condition 
    # TODO: We really want R(dtau) = R(0) + dtau R'(0)
    # TODO: Add functionality to compute dtau R'(0)
    # Solve DxDpH^T R + R DxDpH + R DpDpH R = 0
    theta0  = similar(stable_eq, Float64)
    theta0 .= 0.0
    R0 = lyapc(DpDx_ham(stable_eq, theta0)', DpDp_ham(stable_eq, theta0))
    R0 = inv(R0)

    # Interpolations 

    lam_itp = CubicSpline(lambdas, taus[2:end-1])
    phi_itp = CubicSpline(phis, taus)
    theta_itp =CubicSpline(thetas, taus[2:end-1])

    function riccati_system!(dR, R, p, t)
        phi_ = phi_itp(t)
        theta_ = theta_itp(t)
        lam_ = lam_itp(t)
        
        A = - DpDx_ham(phi_, theta_)
        Q = - DpDp_ham(phi_, theta_)
        C = - DxDx_ham(phi_, theta_)

        dR .= (A*R + R*A' + R*Q*R + C)./lam_
    end

    prob = ODEProblem(riccati_system!, R0, (taus[2], taus[end-1]))
    sol = solve(prob, Rosenbrock23(autodiff=AutoFiniteDiff()), reltol=1e-8, abstol=1e-8)

    return [sol(t) for t in taus]

end

function Prefactor(drift, sigma, stable_eq, phis, thetas, lambdas, ric_sol)

    # Assuming sig constant for now 
    # TODO: Implement for general sigma

    N = length(phis)-1
    D_drift = build_Dₓ_drift(drift)

    a(x) = sigma(x) * sigma(x)'

    # first element of the sum is 0 (integral starts at 0)
    term_1 = sum([tr(D_drift(phi))/lam for (phi, lam) in
                    Iterators.zip(phis[2:end-1], lambdas[2:end-1])])
    term_2 = sum([tr(a(phi) * DDV)/lam for (phi, DDV, lam) in 
                    Iterators.zip(phis[2:end-1], ric_sol[2:end-1], lambdas[2: end-1] )])  # Here is the error for nonconstant sigma
    F_integral = (term_1 + term_2)/N

    DDV_init = ric_sol[1]
    DDV_end = ric_sol[end]
    DV_end = thetas[end]
    a_end = a(phis[end])
    @info "Initial det: " d=det(DDV_init)
    @info "End det: " d=det(DDV_end)   
    @info "Dot product: " dot=(DV_end ⋅ (DDV_end \ DV_end))

    constant = DV_end ⋅ (a_end * DV_end) / sqrt(2* pi) * sqrt(
        det(DDV_init)/det(DDV_end) /(DV_end ⋅ (DDV_end \ DV_end))
    )

    @info "Integral (must be 0 in reversible systems): " F_integral 
    @info "Constant: " constant

    return constant * exp(- F_integral)
end

export Instanton, Potential, Riccati, Prefactor

end # module GMAM
