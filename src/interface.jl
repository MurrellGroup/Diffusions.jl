"""
    sampleforward(process, t, x)

Draw samples forward (i.e., diffuse).

# Arguments
- `process`: a diffusion process (e.g., an `OrnsteinUhlenbeck` process)
- `t`: a taregt time
- `x`: a sample at time 0
"""
sampleforward(process, t, x) = sampleforward(Random.default_rng(), process, t, x)

sampleforward(rng::AbstractRNG, process, t, x) = sampleforward.(rng, process, t, x)
sampleforward(rng::AbstractRNG, process::Process, t, x) = sample(rng, forward(process, x, 0, t))

"""
    samplebackward(guess, process, timesteps, x)

Draw samples backward (i.e., denoise).

# Arguments
- `guess`: a callable object; `guess(X_t, t)` returns the guess (e.g., expectation) of X_0 given X_t at time t
- `process`: a diffusion process (e.g., an `OrnsteinUhlenbeck` process)
- `timesteps`: a vector of positive times (e.g., `5.0 * (1 - 0.05).^(100:-1:0)`)
- `x`: a data point at time `timesteps[end]` (e.g., a sample from the equilibrium distribution)
"""
samplebackward(guess, process, timesteps, x) = samplebackward(Random.default_rng(), guess, process, timesteps, x)

function samplebackward(rng::AbstractRNG, guess, process, timesteps, x)
    checktimesteps(timesteps)
    i = lastindex(timesteps)
    t = timesteps[i]
    x_t = x
    while i > firstindex(timesteps)
        s = timesteps[i-1]
        x_0 = guess(x_t, t)
        prior = forward.(process, x_0, 0, s)
        likelihood = backward.(process, x_t, s, t)
        x_t = sample.(rng, combine.(prior, likelihood))  # sample from the posterior
        t = s
        i -= 1
    end
    return x_t
end

function checktimesteps(timesteps)
    length(timesteps) ≥ 2 || throw(ArgumentError("timesteps must have at least two timesteps"))
    issorted(timesteps) || throw(ArgumentError("timesteps must be decreasing"))
    all(>(0), timesteps) || throw(ArgumentError("all timesteps must be positive"))
end
