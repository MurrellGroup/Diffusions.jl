using Distributions, Plots, LaTeXStrings, Diffusions, Random, Rotations, Flux

#For plotting rotations
function plot_rot!(R; plotfun = plot!, kwargs...)
    RX = R*[1,0,0]
    RY = R*[0,1,0]
    RZ = R*[0,0,1]
    plotfun([0,RX[1]],[0,RX[2]],[0,RX[3]], color = "red"; kwargs...)
    plotfun([0,RY[1]],[0,RY[2]],[0,RY[3]], color = "green"; kwargs...)
    plotfun([0,RZ[1]],[0,RZ[2]],[0,RZ[3]], color = "blue"; kwargs...)
end

#Setting up the rotation manifold and target distribution
baserot1 = QuatRotation(Float32.([0.99963, -0.0261584, 0.00734308, -0.00115531]))
baserot2 = QuatRotation(Float32.([0.999406, 0.0243732, -0.024322, 0.00167138]))

function manifold_rot(l;baserot1 = baserot1,baserot2 = baserot2)
    r1 = baserot1^Int(floor(l)) * baserot2^Int(floor(l))
    r2 = baserot1^Int(ceil(l)) * baserot2^Int(ceil(l))
    return Diffusions.slerp(r1, r2, l-floor(l))
end

target_sample(;baserot1 = baserot1,baserot2 = baserot2) = manifold_rot(rand(typeof(baserot1.q.s))*200)

#Plotting samples from the target distribution
pl = plot()
for i in 1:500
    r = target_sample()
    plot_rot!(r, label = :none, alpha = 0.2)
    plot_rot!(r, plotfun = scatter!, label = :none,
        markerstrokewidth = 0.0, alpha = 0.2)
end
pl

#Setting up the NN that will learn E[x0|xt]
af = swish
hidden_size = 64

rff = RandomFourierFeatures(hidden_size, 1.0f0)

#Input is a flattened representation of a unit quaternion
embed = Chain(Dense(4,hidden_size,af),Dense(hidden_size,hidden_size,af))
#The fourier embedding of the timestep is added to the input, between embed and decode
decode = Chain(
    Dense(hidden_size, hidden_size),LayerNorm(hidden_size),af,
    Dense(hidden_size, hidden_size),LayerNorm(hidden_size),af,
    Dense(hidden_size, 3)
)
#Outputs 3 values, which get scaled into a unit quaternion

rotmodel(x,T) = bcds2flatquats(decode(embed(x) .+ rff(log(T))))

#The timestep schedule
timesteps = timeschedule(exp, 1.0f-6, 1.0f+1, 100)

#Define the rotational diffusion process
P = RotationDiffusion(1.f0)

#NN training setup
batch_size = 8
ps = Flux.params(embed,decode)
opt = Flux.Optimiser(WeightDecay(0.0001f0), Adam(0.001f0))
cumu_loss = 0.0
training_losses = Float32[];

#Training loop
for iter in 1:250000
    #Generate one batch of training data
    x0 = [target_sample() for i in 1:batch_size]
    #Sample t, then diffuse
    t = rand(timesteps)    
    xt = sampleforward(P, t, x0)

    #Flux doesn't speak Quaternion, so we need to convert to a flat representation
    x0_flatquats = rots2flatquats(x0)
    xt_flatquats = rots2flatquats(xt)
    
    #Loss and gradient
    l,gs = Flux.withgradient(ps) do
        x0hat_flatquats = rotmodel(xt_flatquats,t)
        standardloss(P, t, x0hat_flatquats, x0_flatquats)
    end
    
    #Update the parameters
    Flux.Optimise.update!(opt, ps, gs)

    cumu_loss += l
    if mod(iter,100) == 0
        push!(training_losses, cumu_loss/100)
        cumu_loss = 0.0
        #Plotting loss
        if mod(iter,5000) == 0
            display(plot(training_losses, yscale = :log10, ylabel = "Loss", xlabel = "Iters (x100)", label = :none))
        end
    end
end


#First draw samples from the equilibrium distribution of the process
N = 250
x_T = [rand(QuatRotation{Float32}) for i in 1:N]

pl = plot()
for x in x_T
    plot_rot!(x, label = :none, alpha = 0.2)
    plot_rot!(x, plotfun = scatter!, label = :none,
        markerstrokewidth = 0.0, alpha = 0.2)
end
pl

#Use the NN to get E[x0|xt], flipping between flat and quaternion representations
function xzerohat_NN(x_t, t)
    return flatquats2rots(rotmodel(rots2flatquats(x_t),t))
end

#Run the backward diffution
track = Diffusions.Tracker()
@time x0diff = samplebackward(xzerohat_NN, P, timesteps, x_T, tracker = track);

pl = plot()
for x in x0diff
    plot_rot!(x, label = :none, alpha = 0.2)
    plot_rot!(x, plotfun = scatter!, label = :none,
        markerstrokewidth = 0.0, alpha = 0.2)
end
for l in 0:200
    plot_rot!(manifold_rot(l), plotfun = scatter!, label = :none, markersize = 1.25,
        markerstrokewidth = 0.0, alpha = 0.5, color = "yellow")
end
pl

#Make an animated gif of the rotational diffusion
@gif for x0diff in track.data
    pl = plot()
    for x in x0diff
        plot_rot!(x, label = :none, alpha = 0.2)
        plot_rot!(x, plotfun = scatter!, label = :none,
            markerstrokewidth = 0.0, alpha = 0.2)
    end
    for l in 0:200
        plot_rot!(manifold_rot(l), plotfun = scatter!, label = :none, markersize = 1.25,
            markerstrokewidth = 0.0, alpha = 0.5, color = "yellow")
    end
    pl
end
