using Pkg; 
Pkg.add(url="https://github.com/JuliaManifolds/ManifoldMeasures.jl")
Pkg.add(["MeasureBase", "Plots"])
using Manifolds, ManifoldMeasures, MeasureBase, Plots

function expectation(x_t, t; process = process, target_dist = target_dist, n_samples = 1000)
    x0samples = [sampleforward(process, t, x_t) for i in 1:n_samples]
    sampleweights = [mapslices(p -> target_pdf(target_dist, p), x0sample, dims = [1]) for x0sample in x0samples]
    x_0_expectation = similar(x_t)
    for pointindex in CartesianIndices(size(x_t)[2:end])
        x_0_expectation[:, pointindex] = sum( (samp -> samp[:, pointindex]).(x0samples) .* (sampweights -> sampweights[pointindex]).(sampleweights) )
        x_0_expectation[:, pointindex] = project(process.manifold, x_0_expectation[:, pointindex])
    end
    x_0_expectation
end

target_pdf(target_dist, point) = sum( MeasureBase.density_def(dist, point) * weight for (dist, weight) in zip(target_dist.dists, target_dist.weights) )

# N-dimensional sphere - 1 is a circle, 2 is a sphere, 3 is a quaternion, etc.
N = 2
manifold = Sphere(N)
# Distributions on the sphere
dists = [
    ManifoldMeasures.VonMisesFisher(manifold, μ = project(manifold, [1.0, 1.0, 1.0]), κ = 70), 
    ManifoldMeasures.VonMisesFisher(manifold, μ = project(manifold, [-1.0, -1.0, -1.0]), κ = 70),
    ManifoldMeasures.VonMisesFisher(manifold, μ = project(manifold, [0.99995, 9.99934e-5, 0.0]), κ = 1.5)
    ]
unnormalized_weights = [1, 1, 1]
target_dist = (dists = dists, weights = unnormalized_weights ./ sum(unnormalized_weights))

# Diffusion process
process = ManifoldBrownianDiffusion(manifold, 1.0)
d = (1, )
x_T = hcat(rand(uniform_distribution(manifold, zeros(N + 1)), d...)...)
timesteps = timeschedule(exp, log, 0.001, 20, 100)

@time diffusion_samples = samplebackward(expectation, process, timesteps, x_T)

function target_sample(target_dist)
    r = rand()
    for (dist, weight) in zip(target_dist.dists, target_dist.weights)
        r -= weight
        r < 0 && return rand(dist)
    end
end

target_samples = hcat([target_sample(target_dist) for i in eachindex(diffusion_samples)]...)

coordvectors(samples) = [samples[i, :] for i in 1:size(samples)[1]]

pl_S1_diffusion_samples = plot(title = "Diffusion samples", coordvectors(diffusion_samples)..., size=(400, 400), st = :scatter, xlim = (-1.1, 1.1), ylim = (-1.1, 1.1), 
    alpha = 0.3, color = "blue")
pl_S1_target_samples = plot(title = "Target samples", coordvectors(target_samples)..., size=(400, 400), st = :scatter, xlim = (-1.1, 1.1), ylim = (-1.1, 1.1), 
    alpha = 0.3, color="red")
plot(pl_S1_diffusion_samples, pl_S1_target_samples, size = (800, 400))
