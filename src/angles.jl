struct WrappedBrownianMotion{T <: Real} <: Process
    rate::T
end

eq_dist(model::WrappedBrownianMotion) = Uniform(-pi,pi)

function rewrap(x,lb,ub)
    mod(x-lb,ub-lb)+lb
end

reangle(x) = rewrap(x,-pi,pi)

#See https://github.com/MurrellGroup/Diffusions.jl/issues/11
    #This could be optimized to truncate based on the forward mu and var
    #but this is probably fast enough and broad enough for the diffusion scales we'll consider
function wrapped_combine_and_sample(rng, mu, var, back_mu, back_var)
    t_mu = [back_mu + k*2*pi for k in -6:1:6]
    new_var = 1 ./ (1 ./ var .+ 1 ./ back_var)
    new_means = new_var .* (mu ./ var .+ t_mu ./ back_var)
    log_norm_consts =
        -0.5 .* (
            log.(2 .* pi .* (var .* back_var ./ new_var)) .+
            (mu^2 ./ var) .+
            (t_mu.^2 ./ back_var) .- (new_means.^2 ./ new_var)
        )
    nw = exp.(log_norm_consts .- maximum(log_norm_consts))
    component = rand(Categorical(nw ./ sum(nw)))
    return reangle(new_means[component] + randn(rng)*sqrt(new_var))
end

sampleforward(rng::AbstractRNG, P::WrappedBrownianMotion{T}, t::Real, X) where T = X .+ reangle.(randn(rng, T, size(X)) .* sqrt(t*P.rate))

function endpoint_conditioned_sample(rng::AbstractRNG, P::WrappedBrownianMotion{T}, s::Real, t::Real, x_0, x_t) where T <: Real
    x_s = similar(x_0)
    for i in eachindex(x_0)
        x_s[i] = wrapped_combine_and_sample(rng, x_0[i], P.rate*s, x_t[i], P.rate*(t-s))
    end
    return x_s
end