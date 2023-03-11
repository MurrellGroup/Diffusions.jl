using OneHotArrays, Plots, Diffusions, Rotations, Quaternions

loss_scale = sqrt # x -> x
N = 10000
timesteps = timeschedule(exp,Float32(0.000001),Float32(10.0),100)

plot()

P = RotationDiffusion(1.f0)
losses = []
for t in timesteps
    xzero = [rand(QuatRotation{Float32}) for i in 1:N]
    xend = sampleforward(P, t, xzero)
    push!(losses,standard_loss(P,t,rots2flatquats(xzero),rots2flatquats(xend), loss_scale = loss_scale))
end
plot!(timesteps,losses, label = "Rotation")

#Change this when we update the constructor
P = OrnsteinUhlenbeckDiffusion(0.f0)
losses = []
for t in timesteps
    xzero = [randn(Float32) for i in 1:N]
    xend = sampleforward(P, t, xzero)
    push!(losses,standard_loss(P,t,xzero,xend, loss_scale = loss_scale))
end
plot!(timesteps,losses, label = "OU", legend=:topleft)

P = WrappedBrownianDiffusion(1.f0)
losses = []
for t in timesteps
    xzero = [Float32(rand(eq_dist(P))) for i in 1:N]
    xend = sampleforward(P, t, xzero)
    push!(losses,standard_loss(P,t,xzero,xend, loss_scale = loss_scale))
end
plot!(timesteps,losses, label = "Wrapped")

k = 2
P = IndependentDiscreteDiffusion(1.f0, Float32.(ones(k)./k))
losses = []
for t in timesteps
    xzero = onehotbatch([rand(eq_dist(P)) for i in 1:N],1:k)
    xend = Diffusions.forward(P,  xzero, 0.f0, t)
    push!(losses,standard_loss(P,t,xzero,xend.p, loss_scale = loss_scale))
end
@show losses[end]
plot!(timesteps,losses, xscale = :log10, yscale = :log10, label = "Discrete",
legend=:topleft, xlabel = "t", ylabel = "Loss")