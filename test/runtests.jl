using Diffusions
using Test

@testset "Diffusions.jl" begin
    # Write your tests here.
end

@testset "Scheduling" begin
    for T in [Float32, Float64]
        lb, ub, len = T(1e-3), T(1e+2), 6

        t = sqrt_schedule(lb, ub, len)
        @test t isa AbstractVector{T}
        @test length(t) == 6
        @test t[1] ≈ lb
        @test t[2] ≈ 4.101832885125387
        @test t[6] ≈ ub

        t = log_schedule(lb, ub, len)
        @test t isa AbstractVector{T}
        @test length(t) == 6
        @test t[1] ≈ lb
        @test t[2] ≈ 1e-2
        @test t[6] ≈ ub
    end
end

@testset "RandomFourierFeatures" begin
    for T in [Float32, Float64]
        d = 128
        rff = RandomFourierFeatures(d, T(1.0))
        @test size(rff(T[1.0, 2.0]))      == (d, 2)
        @test size(rff(T[1.0, 2.0, 3.0])) == (d, 3)
    end
end
