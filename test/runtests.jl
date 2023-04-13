using Diffusions
using Random
using OneHotArrays
using StaticArrays
using Test

@testset "Diffusion" begin
    diffusion = OrnsteinUhlenbeckDiffusion(0.0)

    # 2d diffusion
    x_0 = randn(2)
    x_t = sampleforward(diffusion, 1.0, x_0)
    @test x_t isa typeof(x_0)
    @test size(x_t) == size(x_0)

    # 2x3 diffusion
    x_0 = randn(2, 3)
    x_t = sampleforward(diffusion, 1.0, x_0)
    @test x_t isa typeof(x_0)
    @test size(x_t) == size(x_0)

    # batched 2d diffusion 
    x_0 = randn(2, 3)
    x_t = sampleforward(diffusion, [1.0, 2.0, 3.0], x_0)
    @test x_t isa typeof(x_0)
    @test size(x_t) == size(x_0)

    diffusion = (
        OrnsteinUhlenbeckDiffusion(0.0),
        UniformDiscreteDiffusion(1.0, 4),
    )

    # 2x3 diffusion with multiple processes
    x_0 = (randn(2, 3), rand(1:4, 2, 3))
    x_t = sampleforward(diffusion, 1.0, x_0)
    @test x_t isa typeof(x_0)
    @test size(x_t[1]) == size(x_0[1])
    @test size(x_t[2]) == size(x_0[2])

    # batched 2d diffusion with multiple processes
    x_0 = (randn(2, 3), rand(1:4, 2, 3))
    x_t = sampleforward(diffusion, [1.0, 2.0, 3.0], x_0)
    @test x_t isa typeof(x_0)
    @test size(x_t[1]) == size(x_0[1])
    @test size(x_t[2]) == size(x_0[2])
end

@testset "Discrete Diffusions" begin
    for T in [Float32, Float64]
        diffusion = IndependentDiscreteDiffusion(one(T), ones(SVector{10, T}))
        @test diffusion isa IndependentDiscreteDiffusion{10, T}
        @test diffusion.r ≈ T(1.0)
        @test diffusion.π ≈ ones(T, 10) ./ 10  # check normalization
    end

    diffusion = IndependentDiscreteDiffusion(1.0f0, ones(SVector{10, Float64}))
    @test diffusion isa IndependentDiscreteDiffusion{10, Float64}

    n_samples = 10_000_000
    rtol = 0.01
    ratematrix(π) = [i == j ? π[j] - 1 : π[j] for i in eachindex(π), j in eachindex(π)]
    for r in [0.5, 1.0, 2.0], t in [0.1, 1.0, 10.0]
        # uniform distribution at equilibrium
        k = 5
        Q = ratematrix(ones(k) / k)
        P_t = exp(Q * r * t)

        diffusion = UniformDiscreteDiffusion(r, k)
        x_0 = fill(1, n_samples)
        x_t = sampleforward(diffusion, t, x_0)
        p1 = sum(x_t .== 1) / n_samples
        @test isapprox(p1,  P_t[1,1]; rtol)
        p2 = sum(x_t .== 2) / n_samples
        @test isapprox(p2, P_t[1,2]; rtol)

        diffusion = IndependentDiscreteDiffusion(r, ones(SVector{k, Float64}))
        x_0 = fill([1; zero(SVector{k-1, Int})], n_samples)
        x_t = sampleforward(diffusion, t, x_0)
        p1 = sum(onecold(x_t) .== 1) / n_samples
        @test isapprox(p1, P_t[1,1]; rtol)
        p2 = sum(onecold(x_t) .== 2) / n_samples
        @test isapprox(p2, P_t[1,2]; rtol)

        # non-uniform distribution at equilibrium
        π = @SVector [0.05, 0.05, 0.2, 0.3, 0.4]
        Q = ratematrix(π)
        P_t = exp(Q * r * t)
        diffusion = IndependentDiscreteDiffusion(r, π)
        x_0 = fill([1; zero(SVector{k-1, Int})], n_samples)
        x_t = sampleforward(diffusion, t, x_0)
        p1 = sum(onecold(x_t) .== 1) / n_samples
        @test isapprox(p1, P_t[1,1]; rtol)
        p2 = sum(onecold(x_t) .== 2) / n_samples
        @test isapprox(p2, P_t[1,2]; rtol)
    end
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

#=
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
=#
