#This is basically F81, but with arb state size and linear prop ops
#IndependentDiscreteDiffusion = "Independent Jumps", as in every time a mutation event happens, you jump to a new state independent of the current state.
struct IndependentDiscreteDiffusion{T <: Real} <: DiscreteStateProcess
    r::T
    π::Vector{T}

    function IndependentDiscreteDiffusion{T}(r::T, π::Vector{T}) where T <: Real
        r > 0 || throw(ArgumentError("r must be positive"))
        all(≥(0), π) || throw(ArgumentError("elements of π must be non-negative"))
        return new{T}(r, π ./ sum(π))
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
function IndependentDiscreteDiffusion(r::Real, π::AbstractVector{<: Real})
    T = promote_type(typeof(float(r)), eltype(π))
    return IndependentDiscreteDiffusion{T}(convert(T, r), convert(Vector{T}, π))
end

eq_dist(model::IndependentDiscreteDiffusion) = Categorical(model.π)

function forward(process::IndependentDiscreteDiffusion, x_s::AbstractArray, s::Real, t::Real)
    (;r, π) = process
    pow = exp(-r * (t - s))
    c1 = (1 - pow) .* π
    c2 = pow .+ c1
    return CategoricalVariables(@. c1 * (1 - x_s) + c2 * x_s)
end

function backward(process::IndependentDiscreteDiffusion, x_t::AbstractArray, s::Real, t::Real)
    (;r, π) = process
    pow = exp(-r * (t - s))
    c1 = (1 - pow) .* π
    return pow .* x_t .+ sum(x_t .* c1, dims = 1)
end
