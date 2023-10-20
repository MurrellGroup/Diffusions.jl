# OU process
struct OrnsteinUhlenbeckDiffusion{T <: Real} <: GaussianStateProcess
    mean::T
    volatility::T
    reversion::T
end

OrnsteinUhlenbeckDiffusion(mean::Real, volatility::Real, reversion::Real) = OrnsteinUhlenbeckDiffusion(float.(promote(mean, volatility, reversion))...)

OrnsteinUhlenbeckDiffusion(mean::T) where T <: Real = OrnsteinUhlenbeckDiffusion(mean,T(1.0),T(0.5))

var(model::OrnsteinUhlenbeckDiffusion) = (model.volatility^2) / (2 * model.reversion)

eq_dist(model::OrnsteinUhlenbeckDiffusion) = Normal(model.mean,sqrt(var(model)))

# These are for nested broadcasting
elmwiseadd(x, y) = x .+ y
elmwisesub(x, y) = x .- y
elmwisemul(x, y) = x .* y
elmwisediv(x, y) = x ./ y

function forward(process::OrnsteinUhlenbeckDiffusion, x_s::AbstractArray, s::Real, t::Real)
    μ, σ, θ = process.mean, process.volatility, process.reversion
    # exp(-(t - s) * θ) * (x_s - μ) + μ
    mean = elmwiseadd.(elmwisemul.(exp(-(t - s) * θ), elmwisesub.(x_s, μ)), μ)
    var = ((1 - exp(-2(t - s) * θ)) * σ^2) / 2θ
    return GaussianVariables(mean, var)
end

function backward(process::OrnsteinUhlenbeckDiffusion, x_t::AbstractArray, s::Real, t::Real)
    μ, σ, θ = process.mean, process.volatility, process.reversion
    # @. exp((t - s) * θ) * (x_t - μ) + μ
    mean = elmwiseadd.(elmwisemul.(exp((t - s) * θ), elmwisesub.(x_t, μ)), μ)
    var = -(σ^2 / 2θ) + (σ^2 * exp(2(t - s) * θ)) / 2θ
    return (μ = mean, σ² = var)
end

_sampleforward(rng::AbstractRNG, process::OrnsteinUhlenbeckDiffusion, t::Real, x::AbstractArray) =
    sample(rng, forward(process, x, 0, t))

function _endpoint_conditioned_sample(rng::AbstractRNG, process::OrnsteinUhlenbeckDiffusion, s::Real, t::Real, x_0::AbstractArray, x_t::AbstractArray)
    prior = forward(process, x_0, 0, s)
    likelihood = backward(process, x_t, s, t)
    return sample(rng, combine(prior, likelihood))
end
