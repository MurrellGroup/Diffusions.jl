#OU process
struct OrnsteinUhlenbeck{T <: Real} <: GaussianStateProcess
    mean::T
    volatility::T
    reversion::T
end

OrnsteinUhlenbeck(mean::Real, volatility::Real, reversion::Real) = OrnsteinUhlenbeck(float.(promote(mean, volatility, reversion))...)

var(model::OrnsteinUhlenbeck) = (model.volatility^2) / (2 * model.reversion)

eq_dist(model::OrnsteinUhlenbeck) = Normal(model.mean,var(model))

function forward(process::OrnsteinUhlenbeck, x_s::AbstractArray, s::Real, t::Real)
    μ, σ, θ = process.mean, process.volatility, process.reversion
    mean = @. exp(-(t - s) * θ) * (x_s - μ) + μ
    var = similar(mean)
    var .= ((1 - exp(-2(t - s) * θ)) * σ^2) / 2θ
    return GaussianVariables(mean, var)
end

function backward(process::OrnsteinUhlenbeck, x_t::AbstractArray, s::Real, t::Real)
    μ, σ, θ = process.mean, process.volatility, process.reversion
    mean = @. exp((t - s) * θ) * (x_t - μ) + μ
    var = similar(mean)
    var .= -(σ^2 / 2θ) + (σ^2 * exp(2(t - s) * θ)) / 2θ
    return (μ = mean, σ² = var)
end
