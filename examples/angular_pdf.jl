using Distributions, Plots, Diffusions

# Target distribution - a mixture of von Mises distributions.
target_dists = [VonMises(-pi/2, 50), VonMises(-pi/2+1, 30), VonMises(0, 2), VonMises(pi/2, 30), VonMises(pi/2+0.5, 50), VonMises(pi, 8)]

function target_sample()
    return rewrap(rand(target_dists[rand(1:length(target_dists))]), -pi, pi)
end

target_pdf(x) = sum([wrappedVMpdf(d, x) for d in target_dists]) / length(target_dists)

# Utilities for circular wrapping.
function rewrap(x,lb,ub)
    mod(x-lb,ub-lb)+lb
end

function wrappedVMpdf(VM,x)
    return pdf(VM, rewrap(x,VM.μ-π,VM.μ+π))
end

function circular_mean(angles, weights = ones(length(angles)))
    return atan(sum(sin.(angles) .* weights), sum(cos.(angles) .* weights))
end

# Expectation function. This would likely be a neural network in a real use cases.
function expect!(a0, a2, T; P = P, sample_func = target_sample, n_samples = 500, max_var_step=100)
    a0samples = [MultiAngleState(size(a0.angles)) for i in 1:n_samples]
    for i in 1:n_samples
        forward_sample!(a0samples[i], a2, P, T, max_var_step=max_var_step)
    end
    weights = [target_pdf.(a0samples[i].angles) for i in 1:n_samples]
    for ix in CartesianIndices(a0.angles)
        a0.angles[ix] = circular_mean(map(a -> a.angles[ix], a0samples), map(w -> w[ix], weights))
    end
    return a0samples
end

# Define the diffusion process and the initial state, and initialize each angle to a random value.
P = AngleDiffusionProcess()
d = (100,)
init_state = MultiAngleState(d)
init_state.angles = [(rand()-0.5) * 2pi for i in 1:size(init_state.angles)[1]]

# This is quite slow. For 100 angles it will likely take around 1min.
@time samp, tracking = diffusion_sample(init_state, P, expect!, step_prop = 0.01, steps = 1500, T=15.0, track = true);

# Plot the results
target_dist_samples = [target_sample() for i in 1:d[1]]
pl1 = plot(sin.(target_dist_samples), cos.(target_dist_samples), title="True samples",  seriestype = :scatter, legend = false, xlims = (-1.1, 1.1), ylims = (-1.1, 1.1), color="red")
pl2 = plot(sin.(samp.angles), cos.(samp.angles), title="Diffusion samples", seriestype = :scatter, legend = false, xlims = (-1.1, 1.1), ylims = (-1.1, 1.1), color="blue")
plot(pl1, pl2, layout = (1, 2), size = (600, 300))

# Should use around 10k samples for this. Since that takes a while a precomputed csv with 10k samples is included.
# It was done with the setting expect!(n_samples=200).
using DelimitedFiles
sample10k = vec(readdlm("examples/anglediffusions10k.csv"))
histogram(sample10k, norm=true, bins=-pi:pi/30:2pi, title="Histogram of angles", legend=false)
plot!(target_pdf, -pi, pi, color = :red, label=false)