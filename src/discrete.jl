#Discrete process

#These work for the generic discrete type
function combine!(dest::DiscreteState, src::DiscreteState)
    # Update the state
    dest.state .*= src.state
    @views for j in axes(dest.state, 2)
        dest.state[:,j] ./= sum(dest.state[:,j])
    end
end

function one_hot_sample(vec::Vector{<:Real})
    vec .= vec ./ sum(vec)
    ind = rand(Categorical(vec))#sample(1:length(vec), Weights(vec))
    sampled = zeros(eltype(vec), size(vec))
    sampled[ind] = 1.0
    return sampled
end

#Replace each column with a 1-of-k vector that is a categorical draw from that row.
function sample!(partition::DiscreteState)
    for i = 1:partition.sites
        partition.state[:, i] .= one_hot_sample(partition.state[:, i])
    end
end

#I need to make a version of this with more dimensional flexibility
#This is not just a DiscreteState, because maybe there could be some <: DiscreteState with specialized ops, like AAs
mutable struct MultiDiscreteState <: DiscreteState
    state::Array{Float64,2}
    states::Int
    sites::Int
    function MultiDiscreteState(states, sites)
        new(zeros(states, sites), states, sites)
    end
    function MultiDiscreteState(freq_vec::Vector{Float64}, sites::Int64)
        state_arr = zeros(length(freq_vec), sites)
        state_arr .= freq_vec
        new(state_arr, length(freq_vec), sites)
    end
end

values(g::MultiDiscreteState) = argmax.(eachcol(g.state))

#This is basically F81, but with arb state size and linear prop ops
#IJ = "Independent Jumps", as in every time a mutation event happens, you jump to a new state independent of the current state.
struct IJ{T <: Real} <: DiscreteStateProcess
    r::T
    π::Vector{T}
    β::T

    function IJ(r::T, π::AbstractVector{<: T}) where T <: Real
        π = π ./ sum(π)
        β = inv(1 - sum(abs2, π))
        return new{T}(r, π, β)
    end
end

function forward(process::IJ, x_s::AbstractArray, s::Real, t::Real)
    (;r, π, β) = process
    pow = exp(-β * r * (t - s))
    c1 = (1 - pow) .* π
    c2 = pow .+ c1
    return CategoricalVariables(@. c1 * (1 - x_s) + c2 * x_s)
end

function backward(process::IJ, x_t::AbstractArray, s::Real, t::Real)
    (;r, π, β) = process
    pow = exp(-β * r * (t - s))
    c1 = (1 - pow) .* π
    return pow .* x_t .+ sum(x_t .* c1, dims = 1)
end

eq_dist(model::IJ) = Categorical(model.pi)
