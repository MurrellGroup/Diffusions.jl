"""
If you want to use a manifold not in Manifolds.jl, you can define a new type that inherits from Manifold and implement the following functions:
- project(manifold, point)
- shortest_path_interpolation(rng, process, point_0, point_t, s, t)

x_t = Array(euclidean_dimension, batch_dims...)
"""

struct ManifoldBrownianDiffusion{M <: AbstractManifold, T <: Real} <: SamplingProcess
    manifold::M
    rate::T
    getsteps::Function
end

ManifoldBrownianDiffusion(manifold::AbstractManifold, rate::T) where T <: Real = ManifoldBrownianDiffusion(manifold, rate, (t) -> range(min(t, 0.05), t, step=0.05))

pointindices(X) = CartesianIndices(size(X)[2:end])

function project!(x_to, x_from, manifold)
    for pointindex in pointindices(x_from)
        x_to[:, pointindex] = project(manifold, x_from[:, pointindex])
    end
end

function sampleforward(rng::AbstractRNG, process::ManifoldBrownianDiffusion{M, T}, t::Real, x_0) where M where T 
    x_t = similar(x_0)
    project!(x_t, x_0, process.manifold)
    for step in process.getsteps(t * process.rate)
        x_t .+= randn(rng, T, size(x_t)...) .* sqrt(step)
        project!(x_t, x_t, process.manifold)
    end
    return x_t
end

shortestpath_interpolation(rng::AbstractRNG, process::ManifoldBrownianDiffusion, p_0::AbstractVector, p_t::AbstractVector, s, t) = 
    shortest_geodesic(process.manifold, p_0, p_t, s / t)

function shortestpath_interpolation!(x_s, rng::AbstractRNG, process::ManifoldBrownianDiffusion, x_0, x_t, s, t)
    for pointindex in pointindices(x_s)
        x_s[:, pointindex] = shortestpath_interpolation(rng, process, x_0[:, pointindex], x_t[:, pointindex], s, t)
    end
end

# Empirically, this is good - should be the same as diffusing C from B as is done in the rotational/angular cases (nice symmetries) for spheres at least.
function endpoint_conditioned_sample(rng::AbstractRNG, process::ManifoldBrownianDiffusion, s::Real, t::Real, x_0, x_t)
    B = sampleforward(rng, process, s, x_0)
    C = sampleforward(rng, process, t-s, x_t)
    shortestpath_interpolation!(C, rng, process, B, C, s, t)
    return C
end

# This has not been tested yet. An optional way of a heuristic brownian bridge for general manifolds.
function endpoint_conditioned_sample_distance_proportional(rng::AbstractRNG, process::ManifoldBrownianDiffusion, s::Real, t::Real, x_0, x_t)
    x_s = similar(x_0)
    D = process.rate # Diffusion coefficient
    dt = min(s, 0.05) # timestep
    for pointindex in pointindices(x_from)
        p_0 = x_0[:, pointindex]
        p_t = x_t[:, pointindex]
        d = distance(process.manifold, p_0, p_t)
        t_remaining = t
        p_cur = p_0
        while (t - t_remaining) < s
            dp = randn(rng, size(p_cur)...) .* (2 * D * dt)
            p_new = p_cur .+ dp
            d_remaining = distance(process.manifold, p_new, p_t)
            factor = 1 - (d_remaining / d) * (t_remaining / T)
            p_adjusted = shortest_geodesic(p_new, p_t, factor)

            p_cur = project(process.manifold, p_adjusted)
            t_remaining -= dt
        end
        x_s[:, pointindex] = p_cur
    end
    x_s
end
