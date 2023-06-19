# sampleforward and samplebackward are the main interface functions.
# 
# A diffusion process type must implement two methods of the following functions:
# 1. _sampleforward(rng, process, t, x_0): draw a sample at time t, conditioned on x_0 at time 0
# 2. _endpoint_conditioned_sample(rng, process, s, t, x_0, x_t): draw a sample at time s, conditioned on x_0 at time 0 and x_t time t
#
# Note that arrays passed to these functions (i.e., x_0 and x_t) are not masked
# arrays because masking layers are removed before passed to them.

"""
    sampleforward(process, t, x)

Draw samples forward (i.e., diffuse).

# Arguments
- `process`: a diffusion process (e.g., an `OrnsteinUhlenbeckDiffusion` process)
- `t`: a target time, or a vector of target times matching the batch (ie. last) dim of x
- `x`: data points at time 0
"""
sampleforward(process, t, x) = sampleforward(Random.default_rng(), process, t, x)

sampleforward(
    rng::AbstractRNG,
    process::NTuple{N,Process},
    t::Union{Real,AbstractVector{<:Real}},
    x::NTuple{N,AbstractArray}
) where {N} = sampleforward.(rng, process, (t,), x)

function sampleforward(rng::AbstractRNG, process::Process, t::Real, x::AbstractArray)
    x = copy(x)
    maskedvec(x) .= _sampleforward(rng, process, t, maskedvec(x))
    return x
end

function sampleforward(rng::AbstractRNG, process::Process, t::AbstractVector{<:Real}, x::AbstractArray)
    x = copy(x)
    for i in axes(x, ndims(x))
        maskedvec(x, i) .= _sampleforward(rng, process, t[i], maskedvec(x, i))
    end
    return x
end

"""
    samplebackward(guess, process, timesteps, x)

Draw samples backward (i.e., denoise).

# Arguments
- `guess`: a callable object; `guess(X_t, t)` returns the guess (e.g., expectation) of X_0 given X_t at time t
- `process`: a diffusion process (e.g., an `OrnsteinUhlenbeckDiffusion` process)
- `timesteps`: a vector of positive times (e.g., `5.0 * (1 - 0.05).^(100:-1:0)`)
- `x`: data points at time `timesteps[end]` (e.g., samples from the equilibrium distribution)
"""
samplebackward(guess, process, timesteps, x; tracker=NullTracker()) =
    samplebackward(Random.default_rng(), guess, process, timesteps, x; tracker)

function samplebackward(rng::AbstractRNG, guess, process, timesteps, x; tracker=NullTracker())
    checktimesteps(timesteps)
    i = lastindex(timesteps)
    t = timesteps[i]
    while i > firstindex(timesteps)
        s = timesteps[i-1]
        x_0 = guess(x, t)
        x = endpoint_conditioned_sample(rng, process, s, t, x_0, x)
        track!(tracker, s, x, x_0)
        t = s
        i -= 1
    end
    return x
end

function endpoint_conditioned_sample(rng::AbstractRNG, process::Process, s::Real, t::Real, x_0::AbstractArray, x_t::AbstractArray)
    if x_t isa MaskedArray
        # x_0 inherits masked elements from x_t
        x_0 = MaskedArray(x_0, x_t.indices)
    end
    x = copy(x_t)
    maskedvec(x) .= _endpoint_conditioned_sample(rng, process, s, t, maskedvec(x_0), maskedvec(x_t))
    return x
end

endpoint_conditioned_sample(rng::AbstractRNG, process, s::Real, t::Real, x_0, x_t) =
    endpoint_conditioned_sample.(rng, process, s, t, x_0, x_t)

# sample x at time s conditioned on x_0 at time 0 and x_t at time t
endpoint_conditioned_sample(P::Process, s::Real, t::Real, x_0, x_t) = endpoint_conditioned_sample(Random.default_rng(), P, s, t, x_0, x_t)

function checktimesteps(timesteps)
    length(timesteps) ≥ 2 || throw(ArgumentError("timesteps must have at least two timesteps"))
    issorted(timesteps) || throw(ArgumentError("timesteps must be increasing"))
    all(>(0), timesteps) || throw(ArgumentError("all timesteps must be positive"))
end
