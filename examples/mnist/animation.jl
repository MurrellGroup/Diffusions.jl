using Diffusions
using Random
using OneHotArrays
using Printf

using BSON: @load
include("model.jl")
@load "mnist-300.bson" unet

include("plotimages.jl")

θ = 4.0f0  # reversion
σ = √(2θ)  # volatility
diffusion = (OrnsteinUhlenbeckDiffusion(0, σ, θ), UniformDiscreteDiffusion(0.5f0, 10))

timesteps = timeschedule(exp10, 1f-4, 1f+1, 301)

function selfconditioned_guess(x)
    x_0 = zero(x)
    function guess((x_t, y_t), t)
        x̂, ŷ = unet(cat(x_t, x_0, dims = 3), onehotbatch(y_t, 1:10), fill(t, size(y_t, 2)))
        x_0 = sigmoid.(x̂)
        return x_0, randcat(softmax(ŷ, dims = 1))
    end
end

tracker = Diffusions.Tracker()
rng = Xoshiro(1234)
x = randn(rng, Float32, (28, 28, 1, 50))
y = rand(rng, 1:10, 50)
samplebackward(rng, selfconditioned_guess(x), diffusion, timesteps, (x, y); tracker);

mkpath("tmp")
for (i, (time, (x0, y0))) in enumerate(zip(tracker.time, tracker.x0))
    title = @sprintf "Generated samples at time = %.6f" time
    fig = plotimages(sigmoid.(x0), y0 .- 1, title)
    fig.savefig(@sprintf "tmp/%04d.png" i)
    plotclose()
end

run(`bash -c 'convert -delay 10 -loop 0 tmp/*.png x0.gif'`)
