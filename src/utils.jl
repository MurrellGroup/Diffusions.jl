#=
# Random sampling from categorical distributions
randcat(p::AbstractArray) = randcat(Random.default_rng(), p)

function randcat(rng::AbstractRNG, p::AbstractArray)
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
=#

function randcat(rng::AbstractRNG, p::AbstractVector)
    K = length(p)
    @assert K ≥ 1
    # This algorithm is O(K), but it is fine because we don't generate many
    # samples from the same distribution.
    u = rand(rng, eltype(p))
    k = 0
    while u ≥ 0 && k < K
        k += 1
        u -= p[k]
    end
    return k
end

"""
    timeschedule(f, lb, ub, n)

Create a vector of timestamps of length `n` from `lb` to `ub`, transformed by `f`.

This is equivalent to `f.(range(invf(lb), invf(ub), length = n))`, where `invf`
is the inverse function of `f`.

# Examples
```julia-repl
julia> timeschedule(exp10, 1e-3, 1e+2, 6)
6-element Vector{Float64}:
   0.001
   0.01
   0.1
   1.0
  10.0
 100.0
```
"""
timeschedule(f, lb::Real, ub::Real, length::Integer) = timeschedule(f, inverse(f), lb, ub, length)
timeschedule(f, invf, lb::Real, ub::Real, length::Integer) = f.(range(invf(lb), invf(ub); length))
