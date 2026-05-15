using Enzyme
using LinearAlgebra

function build_ham(drift, σ)
    return (x,p) -> 
    begin
        v = drift(x) 
        s = σ(x)
        w = s'*p 
        return p ⋅ v + w ⋅ w
    end
end

function build_λ(drift, σ) 
    return function (x, y)
        a = σ(x)*σ(x)'
        return sqrt((drift(x) ⋅ (a\drift(x)))/(y ⋅ (a\y)))
    end
end

function build_θ(drift, σ) 
    λ = build_λ(drift, σ)
    return function (x, y) 
        a = σ(x)*σ(x)'
        return 1/2 * (a\(λ(x,y)*y - drift(x)))
    end
end

format_output(result::NamedTuple) = stack(result)
format_output(result::Tuple) = stack(result)
format_output(result::AbstractArray) = reshape(result, :, 1)

# To use enzyme, it seems necessary to use in place functions
# Don't completely understand why. Something about needing to trace 
# operations in memory.

function build_Dₓ_drift(drift)

    return function (x)
        n = length(x)
        # Input tangents
        batched_dx = Matrix{Float64}(I, n, n)
        batched_dx = ntuple(i -> batched_dx[:, i], n)

        d_drift_batch = autodiff(Forward, Const(drift), BatchDuplicated(x, batched_dx))[1]

        return format_output(d_drift_batch)
    end
end

function build_DₚDₚ_ham(ham::F) where {F<:Function}
    # Forward over reverse 

    return function (x, p) 
        m = length(p)

        # In place gradient computation (reverse step)
        function Dₚ_ham!(x, p, buffer_in)
            autodiff(Reverse, Const(ham), Active, Const(x), Duplicated(p, buffer_in))
            return nothing
        end

        # Input tangents
        batched_dp = Matrix{Float64}(I, m, m)
        batched_dp = ntuple(i -> batched_dp[:, i], m)

        # Output accumulation
        ddh_batch = ntuple(i -> zeros(m), m)
        dhdp = zeros(m)

        # Forward over reverse
        autodiff(set_runtime_activity(Forward), Const(Dₚ_ham!), Const(x), BatchDuplicated(p, batched_dp), 
            BatchDuplicated(dhdp, ddh_batch))

        return format_output(ddh_batch)
        
    end
end


function build_DₓDₓ_ham(ham::F) where {F<:Function}
    # Forward over reverse 

    return function (x, p) 
        n = length(x)

        # In place gradient computation (reverse step)
        function Dₓ_ham!(x, p, buffer_in)
            autodiff(Reverse, Const(ham), Active, Duplicated(x, buffer_in), Const(p))
            return nothing
        end

        # Input tangents
        batched_dx = Matrix{Float64}(I, n, n)
        batched_dx = ntuple(i -> batched_dx[:, i], n)

        # Output accumulation
        ddh_batch = ntuple(i -> zeros(n), n)
        dhdx = zeros(n)

        # Forward over reverse
        autodiff(set_runtime_activity(Forward), Const(Dₓ_ham!), BatchDuplicated(x, batched_dx), Const(p), 
            BatchDuplicated(dhdx, ddh_batch))

        return format_output(ddh_batch)
        
    end
end

function build_Dₓ_ham(ham::F) where {F<:Function}
    # Reverse for gradients

    return function (x, p) 
        n = length(x)
        dhdx = zeros(n)
        autodiff(Reverse, Const(ham), Active, Duplicated(x, dhdx), Const(p))
        return dhdx
    end
end

function build_DₚDₓ_ham(ham::F) where {F<:Function}
    # Forward over reverse
    return function (x,p)
        n = length(x)
        m = length(p)

        # In place gradient computation (reverse step)
        function Dₓ_ham!(x, p, dhdx)
            autodiff(Reverse, Const(ham), Active, Duplicated(x, dhdx), Const(p))
            return nothing
        end

        # Input tangents
        batched_dp = Matrix{Float64}(I, m, m)
        batched_dp = ntuple(i -> batched_dp[:, i], m)

        # Output accumulation
        ddh_batch = ntuple(i -> zeros(n), m)
        dhdx = zeros(n)

        autodiff(set_runtime_activity(Forward), Const(Dₓ_ham!), Const(x), BatchDuplicated(p, batched_dp),
            BatchDuplicated(dhdx, ddh_batch))

        return format_output(ddh_batch)        
    end
end