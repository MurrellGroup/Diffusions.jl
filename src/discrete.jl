#This is basically F81, but with arb state size and linear prop ops
#IndependentDiscreteDiffusion = "Independent Jumps", as in every time a mutation event happens, you jump to a new state independent of the current state.
struct IndependentDiscreteDiffusion{T <: Real} <: DiscreteStateProcess
    r::T
    π::Vector{T}
end

function IndependentDiscreteDiffusion(r::Real, π::AbstractVector{<: Real})
    return IndependentDiscreteDiffusion(float(r), π ./ sum(π))
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
