using Plots, Distributions, Diffusions

circular_mean(angles, weights = ones(size(angles))) =
    atan(sum(sin.(angles) .* weights), sum(cos.(angles) .* weights))

function wrapped_combine(mu, var, back_mu, back_var)
    t_mu = [back_mu + k*2*pi for k in -10:1:10]
    new_var = 1 ./ (1 ./ var .+ 1 ./ back_var)
    new_means = new_var .* (mu ./ var .+ t_mu ./ back_var)
    log_norm_consts =
        -0.5 .* (
            log.(2 .* pi .* (var .* back_var ./ new_var)) .+
            (mu^2 ./ var) .+
            (t_mu.^2 ./ back_var) .- (new_means.^2 ./ new_var)
        )
    return new_means, log_norm_consts
end

function expec(x_t, t, P, target)
    backvar = P.rate * t
    mus, sigs, weights = target
    ms = Float64[]
    lncs = Float64[]
    for i in eachindex(mus, sigs, weights)
        m, l = wrapped_combine(mus[i], sigs[i]^2, x_t, backvar)
        l .+= log(weights[i])
        append!(ms, m)
        append!(lncs, l)
    end
    ncs = exp.(lncs .- maximum(lncs))
    ncs ./= sum(ncs)
    return circular_mean(ms, ncs)
end

expectation(x_t, t; P = P, target = (mus, sigs, weights)) = [expec(x, t, P, target) for x in x_t]

function wrapped_normal_pdf(mu, sig, x)
    mu = mod2pi(mu + pi) - pi
    d = Normal(mu, sig)
    return sum(pdf.(d, (-20*2pi+x):2pi:(20*2pi+x)))
end

weights = [1/8, 1/8, 1/4, 1/2]
mus, sigs = [pi-pi/10, pi, pi/2, 0], [0.5, 0.2, 0.05, 1.0]
ps = sum([weights[i] .* wrapped_normal_pdf.(mus[i], sigs[i], -pi:pi/240:pi) for i in 1:4])

P = WrappedBrownianDiffusion(1.0)
x_T = rand(eq_dist(P), 20000)
timesteps = timeschedule(exp, 0.0001, 20.0, 200)
@time samp = samplebackward(expectation, P, timesteps, x_T)

#We can see that the target is not perfectly matched for the circular expectation!
plot(-pi:pi/240:pi, ps, label = "Target")
histogram!(samp, bins = -pi:pi/90:pi, normalize=:pdf, label = "Draws", linewidth = 0.0, xlim = (-pi, pi), alpha = 0.5)
