#OU process
mutable struct OrnsteinUhlenbeck <: GaussianStateProcess
    mean::Float64
    volatility::Float64
    reversion::Float64
    function OrnsteinUhlenbeck(
        mean::Float64,
        volatility::Float64,
        reversion::Float64,
    )
        new(mean, volatility, reversion)
    end
end

var(model::OrnsteinUhlenbeck) = (model.volatility^2) / (2 * model.reversion)

eq_dist(model::OrnsteinUhlenbeck) = Normal(model.mean,var(model))

#MultiGaussianstate
#Geared for DL, this is an arbitrarily shaped Gaussian state. Aim is performant diffusion.
#No normalizing constants!
#Should probably see if MultiGaussianState{T,N} adds speed.
mutable struct MultiGaussianState <: ContinuousState
    mean::Array
    var::Array
end

function combine!(dest::MultiGaussianState, src::MultiGaussianState)
    newvar = 1 ./ (1 ./ dest.var .+ 1 ./ src.var)
    dest.mean .= newvar .* (dest.mean ./ dest.var .+ src.mean ./ src.var)
    dest.var = newvar
end

#Mistake: this dest var ignores the source var.
#We're effectively treating all sources as point masses, which is fine because we only need this for endpoint-conditioned sampling
#But I need to fix the OU process in MolEv!
function forward!(
    dest::MultiGaussianState,
    source::MultiGaussianState,
    process::OrnsteinUhlenbeck,
    t::Float64,
)
    #if source.var > 0.000001
    #    @error "OU forward diffusion only currently implemented from a zero-variance point mass"
    #end
    dest.mean .=
        (exp(-t * process.reversion)) .* (source.mean .- process.mean) .+ process.mean
    dest.var .=
        ((1 - exp(-2 * t * process.reversion)) * process.volatility^2) /
        (2 * process.reversion)
end

function backward!(
    dest::MultiGaussianState,
    source::MultiGaussianState,
    process::OrnsteinUhlenbeck,
    t::Float64,
)
    #if source.var > 0.000001
    #    @error "OU backward diffusion only currently implemented from a zero-variance point mass"
    #end
    theta = process.reversion
    sigma = process.volatility
    dest.mean .=
        (exp(t * theta)) .* (source.mean .- process.mean) .+ process.mean
    dest.var .=
        -((sigma^2)/(2*theta)) + (sigma^2 * exp(2*t*theta))/(2*theta)
end

function sample!(g::MultiGaussianState)
    g.mean .= randn(size(g.mean)) .* sqrt.(g.var) .+ g.mean
    g.var .= 0.0;
end

#This is what gets used for tracking the reverse sampling
values(g::MultiGaussianState) = copy(g.mean)