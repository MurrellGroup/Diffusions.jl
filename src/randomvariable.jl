# Random Variables
# ----------------

struct GaussianVariables{T, A <: AbstractArray{T}}
    # μ and σ² must have the same size
    μ::A   # mean
    σ²::A  # variance
end

Base.size(X::GaussianVariables) = size(X.μ)

sample(rng::AbstractRNG, X::GaussianVariables{T}) where T = randn(rng, T, size(X)) .* .√X.σ² .+ X.μ

function combine(X::GaussianVariables, lik)
    σ² = @. inv(inv(X.σ²) + inv(lik.σ²))
    μ = @. σ² * (X.μ / X.σ² + lik.μ / lik.σ²)
    return GaussianVariables(μ, σ²)
end


struct CategoricalVariables{T, A <: AbstractArray{T}}
    # the first dimension stores a vector of probabilities
    p::A
end

ncategories(X::CategoricalVariables) = size(X.p, 1)

Base.size(X::CategoricalVariables) = Base.tail(size(X.p))

function sample(rng::AbstractRNG, X::CategoricalVariables)
    x = zeros(Int, size(X))
    for i in CartesianIndices(size(X))
        x[i] = rand(rng, Categorical(X.p[:,i]))
    end
    return onehotbatch(x, 1:ncategories(X))
end

function combine(X::CategoricalVariables, lik)
    p = copy(X.p) .* lik
    for i in CartesianIndices(size(X))
        p[:,i] ./= sum(p[:,i])
    end
    return CategoricalVariables(p)
end
