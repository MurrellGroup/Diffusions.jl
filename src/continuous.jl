#OU process
struct OrnsteinUhlenbeckDiffusion{T <: Real} <: GaussianStateProcess
    mean::T
    volatility::T
    reversion::T
end

OrnsteinUhlenbeckDiffusion(mean::Real, volatility::Real, reversion::Real) = OrnsteinUhlenbeckDiffusion(float.(promote(mean, volatility, reversion))...)

var(model::OrnsteinUhlenbeckDiffusion) = (model.volatility^2) / (2 * model.reversion)

eq_dist(model::OrnsteinUhlenbeckDiffusion) = Normal(model.mean,var(model))

function forward(process::OrnsteinUhlenbeckDiffusion, x_s::AbstractArray, s::Real, t::Real)
    μ, σ, θ = process.mean, process.volatility, process.reversion
    mean = @. exp(-(t - s) * θ) * (x_s - μ) + μ
    var = similar(mean)
    var .= ((1 - exp(-2(t - s) * θ)) * σ^2) / 2θ
    return GaussianVariables(mean, var)
end

function backward(process::OrnsteinUhlenbeckDiffusion, x_t::AbstractArray, s::Real, t::Real)
    μ, σ, θ = process.mean, process.volatility, process.reversion
    mean = @. exp((t - s) * θ) * (x_t - μ) + μ
    var = similar(mean)
    var .= -(σ^2 / 2θ) + (σ^2 * exp(2(t - s) * θ)) / 2θ
    return (μ = mean, σ² = var)
end
