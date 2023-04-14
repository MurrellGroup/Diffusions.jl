"""
    sampleforward(process, t, x)

Draw samples forward (i.e., diffuse).

# Arguments
- `process`: a diffusion process (e.g., an `OrnsteinUhlenbeckDiffusion` process)
- `t`: a target time, or a vector of target times matching the batch (ie. last) dim of x
- `x`: data points at time 0
"""
sampleforward(process, t, x) = sampleforward(Random.default_rng(), process, t, x)

sampleforward(rng::AbstractRNG, process, t::Union{Real, AbstractVector{<: Real}}, x) = sampleforward.(rng, process, (t,), x)
sampleforward(rng::AbstractRNG, process::TractableProcess, t::Real, x) = sample(rng, forward(process, x, 0, t))

function sampleforward(rng::AbstractRNG, process::TractableProcess, t::Real, x::MaskedArray)
    x = copy(x)
    maskedvec(x) .= sampleforward(rng, process, t, maskedvec(x))
    return x
end

function sampleforward(rng::AbstractRNG, process::Process, t::AbstractVector{<: Real}, x)
    d = ndims(x)
    x_t = similar(x)
    for i in axes(x, d)
        selectdim(x_t, d, i) .= sampleforward(rng, process, t[i], selectdim(x, d, i))
    end
    return x_t
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
samplebackward(guess, process, timesteps, x; tracker = NullTracker()) =
    samplebackward(Random.default_rng(), guess, process, timesteps, x; tracker)

function samplebackward(rng::AbstractRNG, guess, process, timesteps, x; tracker = NullTracker())
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

# sample x at time s conditioned on x_0 at time 0 and x_t at time t
function endpoint_conditioned_sample(rng::AbstractRNG, process::Process, s::Real, t::Real, x_0, x_t)
    prior = forward(process, x_0, 0, s)
    likelihood = backward(process, x_t, s, t)
    return sample(rng, combine(prior, likelihood))
end

function endpoint_conditioned_sample(rng::AbstractRNG, process::Process, s::Real, t::Real, x_0::MaskedArray, x_t::MaskedArray)
    prior = forward(process, maskedvec(x_0), 0, s)
    likelihood = backward(process, maskedvec(x_t), s, t)
    x = copy(x_t)
    maskedvec(x) .= sample(rng, combine(prior, likelihood))
    return x
end

endpoint_conditioned_sample(rng::AbstractRNG, process, s::Real, t::Real, x_0, x_t) =
    endpoint_conditioned_sample.(rng, process, s, t, x_0, x_t)

endpoint_conditioned_sample(P::Process, s::Real, t::Real, x_0, x_t) = endpoint_conditioned_sample(Random.default_rng(), P, s, t, x_0, x_t)

function checktimesteps(timesteps)
    length(timesteps) ≥ 2 || throw(ArgumentError("timesteps must have at least two timesteps"))
    issorted(timesteps) || throw(ArgumentError("timesteps must be increasing"))
    all(>(0), timesteps) || throw(ArgumentError("all timesteps must be positive"))
end
