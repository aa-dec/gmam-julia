using Test 
using GMAM
using LinearAlgebra, MatrixEquations
using Random

Random.seed!(111) 
const TEST_A = let 
    tmp = randn(3, 3)
    -tmp' * tmp 
end

const TEST_B = randn(3, 3)
const dt = 1e-2
const N = 1000

@testset "GMAM.jl Tests" begin

    @testset "Testing hamiltonian builders" begin

        drift(x) = TEST_A*x
        sig(x) = TEST_B

        ham = GMAM.build_ham(drift, sig)
        theta = GMAM.build_θ(drift, sig)

        dx_ham = GMAM.build_Dₓ_ham(ham)
        dpdx_ham = GMAM.build_DₚDₓ_ham(ham)
        dpdp_ham = GMAM.build_DₚDₚ_ham(ham)
        dxdx_ham = GMAM.build_DₓDₓ_ham(ham)

        x = rand(3)
        p = rand(3)
        y = rand(3)

        p_compute = theta(x, y)

        @test ham(x, p) ≈ p ⋅ (TEST_A*x + TEST_B*TEST_B'*p) atol=1e-6
        @test ham(x, p_compute) ≈ 0 atol=1e-6
        @test dx_ham(x, p) ≈ TEST_A'*p atol=1e-6
        @test dpdp_ham(x, p) ≈ 2*TEST_B*TEST_B' atol=1e-6
        @test dpdx_ham(x, p) ≈ TEST_A' atol=1e-6
        @test dxdx_ham(x, p) ≈ 0.0 * TEST_A atol=1e-6
    end

    @testset "Testing instanton against linear analytic results" begin
        drift(x) = TEST_A*x
        sig(x) = TEST_B

        test_points = randn(3, 4)
        stable_eq = zeros(3)

        # Solve Lyapunov Equation to find stationary measure
        S = lyapc(TEST_A, 2*TEST_B*TEST_B')
        
        V(x) = 1/2 * (x ⋅ (S\x))
        ∇V(x) = S\x 

        tol = 10/N # can only expect O(h) accuracy due to Riemann sums

        @test Potential(drift, sig, stable_eq, test_points[:, 1]; N=N, dt=dt)[1] ≈ V(test_points[:, 1]) atol=tol
        @test Potential(drift, sig, stable_eq, test_points[:, 2]; N=N, dt=dt)[1] ≈ V(test_points[:, 2]) atol=tol
        @test Potential(drift, sig, stable_eq, test_points[:, 3]; N=N, dt=dt)[1] ≈ V(test_points[:, 3]) atol=tol
        @test Potential(drift, sig, stable_eq, test_points[:, 4]; N=N, dt=dt)[1] ≈ V(test_points[:, 4]) atol=tol

        # @test Potential(drift, sig, stable_eq, test_points[:, 1]; N=N, dt=dt)[2] ≈ ∇V(test_points[:, 1]) atol=tol
        # @test Potential(drift, sig, stable_eq, test_points[:, 2]; N=N, dt=dt)[2] ≈ ∇V(test_points[:, 2]) atol=tol
        # @test Potential(drift, sig, stable_eq, test_points[:, 3]; N=N, dt=dt)[2] ≈ ∇V(test_points[:, 3]) atol=tol
        # @test Potential(drift, sig, stable_eq, test_points[:, 4];)[2] ≈ ∇V(test_points[:, 4]) atol=tol


    end
    

end