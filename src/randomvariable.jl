# Random Variables
# ----------------

struct GaussianVariables{A, B}
    # μ and σ² must have the same size
    μ::A   # mean (array)
    σ²::B  # variance (scalar)
end

Base.size(X::GaussianVariables) = size(X.μ)

sample(rng::AbstractRNG, X::GaussianVariables) =
    elmwisemul.(randn(rng, eltype(X.μ), size(X)), √X.σ²) .+ X.μ

function combine(X::GaussianVariables, lik)
    σ² = inv(inv(X.σ²) + inv(lik.σ²))
    μ = elmwisemul.(σ², elmwisediv.(X.μ, X.σ²) .+ elmwisediv.(lik.μ, lik.σ²))
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
