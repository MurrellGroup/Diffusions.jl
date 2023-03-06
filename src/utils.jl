#Stolen from https://github.com/FluxML/model-zoo/blob/master/vision/diffusion_mnist/diffusion_mnist.jl
"""
    GaussianFourierProjection(embed_dim, scale)

Returns a function that projects to a vector of length `embed_dim` using a random Fourier feature projection with Gaussian kernel and bandwidth `scale`.
"""
function GaussianFourierProjection(embed_dim, scale)
    # Instantiate W once
    W = randn(Float32, embed_dim ÷ 2) .* scale
    # Return a function that always references the same W
    function GaussFourierProject(t)
        t_proj = t' .* W * Float32(2π)
        [sin.(t_proj); cos.(t_proj)]
    end
end

# Random sampling from categorical distributions
randcatcold(p::AbstractArray) = randcatcold(Random.default_rng(), p)

function randcatcold(rng::AbstractRNG, p::AbstractArray)
    K = size(p, 1)
    @assert K ≥ 1
    X = zeros(Int, Base.tail(size(p)))
    for ix in CartesianIndices(size(X))
        # This algorithm is O(K), but it is fine because we don't generate many
        # samples from the same distribution.
        u = rand(rng, eltype(p))
        k = 0
        while u ≥ 0 && k < K
            k += 1
            u -= p[k,ix]
        end
        X[ix] = k
    end
    return X
end

# Random sampling from categorical distributions - returning onehots
randcat(p::AbstractArray) = randcat(Random.default_rng(), p)

function randcat(rng::AbstractRNG, p::AbstractArray)
    K = size(p, 1)
    return onehotbatch(randcatcold(rng, p), 1:K)
end

sqrt_schedule(lb::T, ub::T, length::Integer) where T <: Real = range(√lb, √ub; length) .^ 2
sqrt_schedule(lb::Real, ub::Real, length::Integer) = sqrt_schedule(promote(lb, ub)..., length)

log_schedule(lb::T, ub::T, length::Integer) where T <: Real = exp.(range(log(lb), log(ub); length))
log_schedule(lb::Real, ub::Real, length::Integer) = log_schedule(promote(lb, ub)..., length)
