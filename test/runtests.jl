using Diffusions
using Diffusions: MaskedArray, mask, nmasked, maskedvec, flatquats
using Random
using OneHotArrays
using StaticArrays
using Rotations: QuatRotation
using Flux
using Test

#=
@testset "Random categorical" begin
    rng = Xoshiro(12345)
    p = [0.1, 0.2, 0.3, 0.4]
    x = randcat(rng, p)
    @test x isa Int

    N = 1_000_000
    x = [randcat(rng, p) for _ in 1:N]
    @test minimum(x) ≥ 1
    @test maximum(x) ≤ 4
    @test 0.099 < sum(==(1), x) / N < 0.101
    @test 0.199 < sum(==(2), x) / N < 0.201
    @test 0.299 < sum(==(3), x) / N < 0.301
    @test 0.399 < sum(==(4), x) / N < 0.401
end

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

    # batched 3d diffusion
    x_0 = randn(3)
    x_t = sampleforward(diffusion, [1.0, 2.0, 3.0], x_0)
    @test x_t isa typeof(x_0)
    @test size(x_t) == size(x_0)

    # batched 2x3 diffusion 
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
        p1 = sum(onecold.(x_t) .== 1) / n_samples
        @test isapprox(p1, P_t[1,1]; rtol)
        p2 = sum(onecold.(x_t) .== 2) / n_samples
        @test isapprox(p2, P_t[1,2]; rtol)

        # non-uniform distribution at equilibrium
        π = @SVector [0.05, 0.05, 0.2, 0.3, 0.4]
        Q = ratematrix(π)
        P_t = exp(Q * r * t)
        diffusion = IndependentDiscreteDiffusion(r, π)
        x_0 = fill([1; zero(SVector{k-1, Int})], n_samples)
        x_t = sampleforward(diffusion, t, x_0)
        p1 = sum(onecold.(x_t) .== 1) / n_samples
        @test isapprox(p1, P_t[1,1]; rtol)
        p2 = sum(onecold.(x_t) .== 2) / n_samples
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

@testset "Masked Diffusion" begin
    process = OrnsteinUhlenbeckDiffusion(0.0, 1.0, 0.5)
    x_0 = randn(5, 10)
    m = x_0 .< 0
    masked = mask(x_0, m)
    x_t = sampleforward(process, 1.0, masked)
    @test size(x_t) == size(x_0)
    @test x_t isa MaskedArray
    @test all(x_t[m] .!= x_0[m])
    @test all(x_t[.!m] .== x_0[.!m])

    function guess(x, t)
        # random "guess" for testing
        x = copy(x)
        maskedvec(x) .+= randn(nmasked(x))
        return x
    end
    x = samplebackward(guess, process, [1/8, 1/4, 1/2, 1/1], x_t)
    @test size(x) == size(x_t)
    @test x isa MaskedArray
    @test all(x[m] .!= x_t[m])
    @test all(x[.!m] .== x_t[.!m])
end
=#

@testset "Loss" begin
    p = OrnsteinUhlenbeckDiffusion(0.0, 1.0, 0.5)
    x_0 = randn(5, 10)
    t = rand(10)
    @test standardloss(p, t, x_0, x_0) == 0
    x = rand(5, 10)
    @test standardloss(p, t, x, x_0) > 0

    # unmasked elements don't contribute to the loss
    x = copy(x_0)
    m = x_0 .< 0
    x_0 = mask(x_0, m)
    x[.!m] .= 0
    @test standardloss(p, t, x, x_0) == 0

    # but masked elements do
    x[m] .= 0
    @test standardloss(p, t, x, x_0) > 0
end

@testset "Autodiff" begin
    n = 10
    p = OrnsteinUhlenbeckDiffusion(0.0f0, 1.0f0, 0.5f0)
    for t in (1.0f0, ones(Float32, n)), masked in (false, true)
        x_0 = rand(Float32, 1, n)
        if masked
            x_0 = mask(x_0, rand(size(x_0)...) .< 0.5)
        end
        d = 3
        f = Dense(d => 1)
        x = randn(Float32, d, n)
        (; val, grad) = Flux.withgradient(f -> standardloss(p, t, f(x), x_0), f)
        @test val ≥ 0
    end

    n = 10
    p = RotationDiffusion(1.0f0)
    for t in (1.0f0, ones(Float32, n)), masked in (false, true)
        x_0 = rand(QuatRotation{Float32}, n)
        if masked
            x_0 = mask(x_0, rand(size(x_0)...) .< 0.5)
        end
        d = 3
        f = Dense(d => 4)
        x = randn(Float32, d, n)
        (; val, grad) = Flux.withgradient(f -> standardloss(p, t, f(x), x_0), f)
        @test val ≥ 0
    end

    n = 10
    p = WrappedBrownianDiffusion(1.0f0)
    for t in (1.0f0, ones(Float32, n)), masked in (false, true)
        x_0 = rand(Float32, 1, n)
        if masked
            x_0 = mask(x_0, rand(size(x_0)...) .< 0.5)
        end
        d = 3
        f = Dense(d => 1)
        x = randn(Float32, d, n)
        (; val, grad) = Flux.withgradient(f -> standardloss(p, t, f(x), x_0), f)
        @test val ≥ 0
    end

    k, n = 4, 10
    p = UniformDiscreteDiffusion(1.0f0, k)
    for t in (1.0f0, ones(Float32, n)), masked in (false, true)
        x_0 = rand(1:k, n)
        if masked
            x_0 = mask(x_0, rand(size(x_0)...) .< 0.5)
        end
        d = 3
        f = Dense(d => k)
        x = randn(Float32, d, n)
        (; val, grad) = Flux.withgradient(f -> standardloss(p, t, f(x), x_0), f)
        @test val ≥ 0 
    end

    k, n = 4, 10
    p = IndependentDiscreteDiffusion(1.0f0, ones(SVector{k, Float32}))
    for t in (1.0f0, ones(Float32, n)), masked in (false, true)
        x_0 = [Diffusions.onehotsvec(k, rand(1:k)) for _ in 1:n]
        if masked
            x_0 = mask(x_0, rand(size(x_0)...) .< 0.5)
        end
        d = 3
        f = Dense(d => k)
        x = randn(Float32, d, n)
        (; val, grad) = Flux.withgradient(f -> standardloss(p, t, f(x), x_0), f)
        @test val ≥ 0
    end
end
