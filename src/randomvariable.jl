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


struct CategoricalVariables{K, T}
    p::Array{SVector{K, T}}
end

ncategories(::CategoricalVariables{K}) where K = K

Base.size(X::CategoricalVariables) = size(X.p)

function sample(rng::AbstractRNG, X::CategoricalVariables{K, T}) where {K, T}
    x = Array{SVector{K, Int}}(undef, size(X))
    for i in eachindex(x, X.p)
        k = randcat(rng, X.p[i])
        x[i] = onehotsvec(K, k)
    end
    return x
end

onehotsvec(K, k::Integer) = SVector{K}(ntuple(_ -> 0, k - 1)..., 1, ntuple(_ -> 0, K - k)...) 
onehotsvec(K, x::AbstractArray{<: Integer}) = map(k -> onehotsvec(K, k), x)

function combine(X::CategoricalVariables, lik)
    p = map(.*, X.p, lik)
    for i in eachindex(p)
        p[i] = p[i] ./ sum(p[i])
    end
    return CategoricalVariables(p)
end
