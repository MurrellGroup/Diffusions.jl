"""
If you want to use a manifold not in Manifolds.jl, you can define a new type that inherits from Manifold and implement the following functions:
- project(manifold, point)
- shortest_path_interpolation(rng, process, point_0, point_t, s, t)
"""

# x_t = Array(euclidean_dimension, batch_size)

struct ManifoldBrownianDiffusion{M <: AbstractManifold, T <: Real} <: SamplingProcess
    manifold::M
    rate::T
    getsteps::Function
end

ManifoldBrownianDiffusion(manifold::AbstractManifold, rate::T) where T <: Real = ManifoldBrownianDiffusion(manifold, rate, (t) -> [t])

function project!(x_to, x_from, manifold)
    for ix in CartesianIndices(size(x_from)[2:end])
        x_to[:, ix] = project(manifold, x_from[:, ix])
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

shortestpath_interpolation_all(rng::AbstractRNG, process::ManifoldBrownianDiffusion, x_0, x_t, s, t) =
    mapslices(indicesofpoint -> shortestpath_interpolation(rng, process, x_0[indicesofpoint], x_t[indicesofpoint], s, t), CartesianIndices(x_0), dims = [1])

# Empirically, this is really good, but I am not completely sure why.
function endpoint_conditioned_sample(rng::AbstractRNG, process::ManifoldBrownianDiffusion, s::Real, t::Real, x_0, x_t)
    B = sampleforward(rng, process, s, x_0)
    C = sampleforward(rng, process, t-s, x_t)
    return shortestpath_interpolation_all(rng, process, B, C, s, t)
end

# This does the same as for the angular diffusion but the code is not as nice, and empirically it is not as good.
#=
function endpoint_conditioned_sample(rng::AbstractRNG, process::ManifoldBrownianDiffusion, s::Real, t::Real, x_0, x_t)
    B = sampleforward(rng, process, s, x_0)
    C = sampleforward(rng, process, t-s, B)
    return [exp(process.manifold, b, log(process.manifold, c, shortest_geodesic(process.manifold, c, endpoint, (t-s) / t))) for (b, c, endpoint) in zip(B, C, x_t)]
end
=#
