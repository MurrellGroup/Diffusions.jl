Base.broadcastable(x::Process) = Ref(x)

abstract type WrappedDiffusion{T <: Real} <: SimulationProcess end

struct WrappedBrownianMotion{T <: Real} <: WrappedDiffusion{T}
    rate::T
end

struct WrappedInterpolatedBrownianMotion{T <: Real} <: WrappedDiffusion{T}
    rate::T
end

eq_dist(::WrappedDiffusion) = Uniform(-pi, pi)

reangle(x) = mod2pi(x + pi) - pi

function wrapped_combine_and_sample(rng::AbstractRNG, P::WrappedBrownianMotion, mu::T, var::T, back_mu::T, back_var::T) where T
    #Exact bridge, using the mixture trick
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

function wrapped_combine_and_sample(rng::AbstractRNG, P::WrappedInterpolatedBrownianMotion, mu::T, var::T, back_mu::T, back_var::T) where T
    #Approx bridge, using interpolation trick
    B = sampleforward(rng, P, var, mu)
    C = sampleforward(rng, P, back_var, B)
    C = reangle(C)
    back_mu = reangle(back_mu)
    if back_mu > C
        if back_mu - C < pi
            shortestdist = back_mu - C
            dir = 1
        else
            shortestdist = 2pi - (back_mu - C)
            dir = -1
        end
    else
        if C - back_mu < pi
            shortestdist = C - back_mu
            dir = -1
        else
            shortestdist = 2pi - (C - back_mu)
            dir = 1
        end
    end
    return reangle(B + dir * shortestdist * (var / (var + back_var)))
end

sampleforward(rng::AbstractRNG, P::WrappedDiffusion{T}, t::Real, X) where T = X .+ reangle.(randn(rng, T, size(X)) .* sqrt(t*P.rate))

endpoint_conditioned_sample(rng::AbstractRNG, P::WrappedDiffusion, s::Real, t::Real, x_0, x_t) where T =
    wrapped_combine_and_sample.(rng, P, x_0, P.rate * s, x_t, P.rate * (t - s))