#This is basically F81, but with arb state size and linear prop ops
#IndependentDiscreteDiffusion = "Independent Jumps", as in every time a mutation event happens, you jump to a new state independent of the current state.
struct IndependentDiscreteDiffusion{K, T <: Real} <: DiscreteStateProcess
    r::T
    π::SVector{K, T}

    function IndependentDiscreteDiffusion{T}(r::T, π::SVector{K, T}) where {K, T <: Real}
        r > 0 || throw(ArgumentError("r must be positive"))
        sum(π) > 0 || throw(ArgumentError("sum of π must be positive"))
        all(≥(0), π) || throw(ArgumentError("elements of π must be non-negative"))
        return new{K, T}(r, π ./ sum(π))
    end
end

"""
    IndependentDiscreteDiffusion(r::Real, π::AbstractVector{<: Real})

Create a discrete diffusion process with independent jumps.

The new state after a state transition is independent of the current state.  The
transition probability matrix at time t is

    P(t) = exp(r Q t),

where Q is a rate matrix with equilibrium distribution π.
"""
function IndependentDiscreteDiffusion(r::Real, π::SVector{K, <: Real}) where K
    T = promote_type(typeof(r), eltype(π))
    return IndependentDiscreteDiffusion{T}(convert(T, r), convert(SVector{K, T}, π))
end

eq_dist(model::IndependentDiscreteDiffusion) = Categorical(model.π)

function forward(process::IndependentDiscreteDiffusion, x_s::AbstractArray, s::Real, t::Real)
    (;r, π) = process
    pow = exp(-r * (t - s))
    c1 = (1 - pow) .* π
    c2 = pow .+ c1
    return CategoricalVariables([@. c1 * (1 - x) + c2 * x for x in x_s])
end

function backward(process::IndependentDiscreteDiffusion, x_t::AbstractArray, s::Real, t::Real)
    (;r, π) = process
    pow = exp(-r * (t - s))
    c1 = (1 - pow) .* π
    return [pow * x .+ x'c1 for x in x_t]
end

_sampleforward(rng::AbstractRNG, process::IndependentDiscreteDiffusion, t::Real, x::AbstractArray) =
    sample(rng, forward(process, x, 0, t))
