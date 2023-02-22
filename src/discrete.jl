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
#Can maybe nuke the "normalize"
#IJ = "Independent Jumps", as in every time a mutation event happens, you jump to a new state independent of the current state.
mutable struct IJ <: DiscreteStateProcess
    r::Float64
    pi::Vector{Float64}
    beta::Float64

    function IJ(r::Float64,pi::Vector{Float64}; normalize=false)
        piNormed = pi ./ sum(pi)
        beta = normalize ? 1/(1-sum(abs2.(piNormed))) : 1.0
        new(r,piNormed,beta)
    end
    function IJ(pi::Vector{Float64}; normalize=false)
        IJ(1.0,pi;normalize=normalize)
    end
    function IJ(states::Int64; normalize=false)
        IJ(1.0,ones(states) ./ states;normalize=normalize)
    end
end

function backward!(dest::DiscreteState,
        source::DiscreteState,
        model::IJ,
        t::Float64)
    pow = exp(-model.beta*model.r*t)
    c1 = ((1 - pow).*model.pi)
    c2 = (pow .+ c1)
    vsum = sum(source.state .* c1, dims=1)
    dest.state .= pow .* source.state .+ vsum
end

#For speed reasons, these partitions are stored in wide format. Should consider switching everything to that.
function forward!(dest::DiscreteState,
        source::DiscreteState,
        model::IJ,
        t::Float64)
    #ToDo: check this is the same as F81 using the full matrix exponential.
    scals = sum(source.state,dims = 1)[:] #Requires V1.0 fix.
    pow = exp(-model.beta*model.r*t)
    c1 = ((1 - pow).*model.pi)
    c2 = (pow .+ ((1 - pow).*model.pi))
    dest.state .= (scals' .- source.state).*c1 .+ source.state.*c2
end

eq_dist(model::IJ) = Categorical(model.pi)