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

@testset "GMAM.jl Tests" begin

    @testset "Testing hamiltonian builders" begin

        drift(x) = TEST_A*x
        sig(x) = TEST_B

        ham = GMAM.build_ham(drift, sig)
        dx_ham = GMAM.build_Dₓ_ham(ham)
        dpdx_ham = GMAM.build_DₚDₓ_ham(ham)
        dpdp_ham = GMAM.build_DₚDₚ_ham(ham)

        x = rand(3)
        p = rand(3)

        @test ham(x, p) ≈ p ⋅ (TEST_A*x + TEST_B*TEST_B'*p) atol=1e-6
        @test dx_ham(x, p) ≈ TEST_A'*p atol=1e-6
        @test dpdp_ham(x, p) ≈ 2*TEST_B*TEST_B' atol=1e-6
        @test dpdx_ham(x, p) ≈ TEST_A' atol=1e-6
    end

    @testset "Testing instanton against linear analytic results" begin
        drift(x) = TEST_A*x
        sig(x) = TEST_B

        test_points = randn(3, 4)
        stable_eq = zeros(3)

        # Solve Lyapunov Equation to find stationary measure
        S = lyapc(TEST_A, 2*TEST_B*TEST_B')
        
        V(x) = x ⋅ (S\x)
        ∇V(x) = 2* S\x 

        @test Potential(drift, sig, stable_eq, test_points[:, 1])[1] ≈ V(test_points[:, 1]) atol=1e-2
        @test Potential(drift, sig, stable_eq, test_points[:, 2])[1] ≈ V(test_points[:, 2]) atol=1e-2
        @test Potential(drift, sig, stable_eq, test_points[:, 3])[1] ≈ V(test_points[:, 3]) atol=1e-2
        @test Potential(drift, sig, stable_eq, test_points[:, 4])[1] ≈ V(test_points[:, 4]) atol=1e-2

        # @test Potential(drift, sig, stable_eq, test_points[:, 1])[2] ≈ ∇V(test_points[:, 1]) atol=1e-2
        # @test Potential(drift, sig, stable_eq, test_points[:, 2])[2] ≈ ∇V(test_points[:, 2]) atol=1e-2
        # @test Potential(drift, sig, stable_eq, test_points[:, 3])[2] ≈ ∇V(test_points[:, 3]) atol=1e-2
        # @test Potential(drift, sig, stable_eq, test_points[:, 4])[2] ≈ ∇V(test_points[:, 4]) atol=1e-2


    end
    

end