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

function sqrt_schedule(lb,ub,steps; T = Float32)
    T.([sqrt(lb):(sqrt(ub)-sqrt(lb))/steps:sqrt(ub);].^2)
end

function log_schedule(lb,ub,steps; T = Float32)
    T.(exp.([log(lb):(log(ub)-log(lb))/steps:log(ub);]))
end