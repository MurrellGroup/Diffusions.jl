using OneHotArrays, Plots, Diffusions, Rotations, Quaternions
using StaticArrays
using Compat: stack

N = 10_000
timesteps = timeschedule(exp, 1.0f-6, 1.0f+1, 100)

plot()

P = RotationDiffusion(1.f0)
losses = []
for t in timesteps
    xzero = [rand(QuatRotation{Float32}) for i in 1:N]
    xend = sampleforward(P, t, xzero)
    push!(losses, standardloss(P, t, rots2flatquats(xend), xzero))
end
plot!(timesteps,losses, label = "Rotation")

#Change this when we update the constructor
P = OrnsteinUhlenbeckDiffusion(0.f0)
losses = []
for t in timesteps
    xzero = [randn(Float32) for i in 1:N]
    xend = sampleforward(P, t, xzero)
    push!(losses, standardloss(P, t, xend, xzero))
end
plot!(timesteps,losses, label = "OU", legend = :topleft)

P = WrappedBrownianDiffusion(1.f0)
losses = []
for t in timesteps
    xzero = [Float32(rand(eq_dist(P))) for i in 1:N]
    xend = sampleforward(P, t, xzero)
    push!(losses, standardloss(P, t, xend, xzero))
end
plot!(timesteps, losses, label = "Wrapped")

k = 2
P = IndependentDiscreteDiffusion(1.f0, ones(SVector{k, Float32}))
losses = []
for t in timesteps
    xzero = Diffusions.onehotsvec.(k, rand(eq_dist(P), N))
    xend = Diffusions.forward(P, xzero, 0.0f0, t)
    push!(losses, standardloss(P, t, log.(stack(xend.p)), xzero))
end
plot!(timesteps, losses, xscale = :log10, yscale = :log10, label = "Discrete", legend = :topleft, xlabel = "t", ylabel = "Loss")
