using Diffusions
using Random
using Test

@testset "Diffusions.jl" begin
    # Write your tests here.
end

@testset "Scheduling" begin
    for T in [Float32, Float64]
        lb, ub, len = T(1e-3), T(1e+2), 6

        t = timeschedule(square, lb, ub, len)
        @test t isa AbstractVector{T}
        @test length(t) == 6
        @test t[1] ≈ lb
        @test t[2] ≈ 4.101832885125387
        @test t[6] ≈ ub

        t = timeschedule(exp, lb, ub, len)
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
        @test rff(T(1.0)) isa Vector{T}
        @test rff(T[1.0]) isa Matrix{T}
        @test size(rff(T(1.0))) == (d,)
        @test size(rff(T[1.0, 2.0])) == (d, 2)
        @test size(rff(T[1.0, 2.0, 3.0])) == (d, 3)
    end

    # mixed types
    rff = RandomFourierFeatures(128, 1.0)
    @test rff isa RandomFourierFeatures{Float64}
    @test rff(1.0f0) isa Vector{Float64}
    @test rff([1.0f0]) isa Matrix{Float64}

    rff = RandomFourierFeatures(128, 1.0f0)
    @test rff isa RandomFourierFeatures{Float32}
    @test rff(1.0) isa Vector{Float32}
    @test rff([1.0]) isa Matrix{Float32}
end

@testset "Random categorical" begin
    rng = Xoshiro(12345)
    p = [
        0.1 0.3
        0.2 0.5
        0.7 0.2
    ]
    x = randcat(rng, p)
    @test size(x) == (2,)
    @test x isa Vector{Int}

    # test a point to avoid stupid bugs
    N = 1_000_000
    n = 0
    for _ in 1:N
        x = randcat(rng, p)
        n += x[1] == 1
    end
    @test 0.099 < n / N < 0.101
end
