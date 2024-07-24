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

function _endpoint_conditioned_sample(rng::AbstractRNG, process::IndependentDiscreteDiffusion, s::Real, t::Real, x_0::AbstractArray, x_t::AbstractArray)
    prior = forward(process, x_0, 0, s)
    likelihood = backward(process, x_t, s, t)
    return sample(rng, combine(prior, likelihood))
end


struct MaskedDiscreteDiffusion <: DiscreteStateProcess
    vocab_size::Int
    mask_token_id
    α::Function
end

function _sampleforward(rng::AbstractRNG, process::MaskedDiscreteDiffusion, t::Real, x::AbstractArray)
    # Get the keep probability
    α_t = process.α(t)[1] 
    
    vocab_size = process.vocab_size 
    num_tokens = length(x) ÷ vocab_size
    
    result = falses(length(x))
    
    for i in 1:num_tokens
        token_slice = (i-1)*vocab_size + 1 : i*vocab_size
        if rand(rng) < α_t
            # Keep the original token
            result[token_slice] = x[token_slice]
        else
            # Mask the token
            result[token_slice[process.mask_token_id]] = true
        end
    end
    
    return result
end

function _endpoint_conditioned_sample(
    rng::AbstractRNG, 
    process::MaskedDiscreteDiffusion, 
    s::Real, 
    t::Real, 
    x0::AbstractVector{Bool}, 
    xt::AbstractVector{Bool}
)   
    sequence_length = length(xt) ÷ process.vocab_size
    xs = copy(xt)
    
    α_s = process.α(s)[1]
    α_t = process.α(t)[1]

    for i in 1:sequence_length
        token_slice = (i-1)*process.vocab_size + 1 : i*process.vocab_size
        if xt[token_slice[process.mask_token_id]]
            probs = zeros(Float32, process.vocab_size)
            probs[process.mask_token_id] = 1 - α_s
            predicted_token = findfirst(x0[token_slice])
            probs[predicted_token] = α_s - α_t
            probs ./= sum(probs)
            sampled_token = rand(rng, Categorical(probs))
            xs[token_slice] .= false
            xs[token_slice[sampled_token]] = true
        end
    end

    return xs
end

