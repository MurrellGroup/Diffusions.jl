struct WrappedBrownianMotion{T <: Real} <: Process
    rate::T
end

eq_dist(::WrappedBrownianMotion) = Uniform(-pi, pi)

reangle(x) = mod2pi(x + pi) - pi

# See https://github.com/MurrellGroup/Diffusions.jl/issues/11
# This could be optimized to truncate based on the forward mu and var
# but this is probably fast enough and broad enough for the diffusion scales we'll consider
function wrapped_combine_and_sample(
    rng::AbstractRNG, mu::T, var::T, back_mu::T, back_var::T) where T <: Real
    τ = T(2π)
    t_mu = [back_mu + k * τ for k in -6:6]
    new_var = 1 / (1 / var + 1 / back_var)
    new_means = @. new_var * (mu / var + t_mu / back_var)
    log_norm_consts =
        @. (
            log(τ * (var * back_var / new_var)) +
            (mu^2 / var) +
            (t_mu^2 / back_var) - (new_means^2 / new_var)
        ) / -2
    nw = exp.(log_norm_consts .- maximum(log_norm_consts))
    component = rand(rng, Categorical(nw ./ sum(nw)))
    return reangle(new_means[component] + randn(rng) * sqrt(new_var))
end

sampleforward(rng::AbstractRNG, P::WrappedBrownianMotion{T}, t::Real, X) where T = X .+ reangle.(randn(rng, T, size(X)) .* sqrt(t*P.rate))

endpoint_conditioned_sample(rng::AbstractRNG, P::WrappedBrownianMotion, s::Real, t::Real, x_0, x_t) =
    wrapped_combine_and_sample.(rng, x_0, P.rate * s, x_t, P.rate * (t - s))
