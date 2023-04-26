using Distributions, Plots, Flux, LaTeXStrings, Diffusions

#Sample function
function target_sample()
    l = rand()*14
    return Float32.([exp(-l/7)*sin(l) ,exp(-l/7)*cos(l)] * 5.0)
end

#Setting up the NN that will learn E[x0|xt]
af = swish
hidden_size = 64
rff_t = RandomFourierFeatures(hidden_size, 1.0f0)
rff_x1 = RandomFourierFeatures(hidden_size, 1.0f0)
rff_x2 = RandomFourierFeatures(hidden_size, 1.0f0)
embed = Chain(Dense(2,hidden_size,af),Dense(hidden_size,hidden_size,af))
#The fourier embedding of the timestep is added to the input, between embed and decode
decode = Chain(
    Dense(hidden_size, hidden_size),LayerNorm(hidden_size),af,
    Dense(hidden_size, hidden_size),LayerNorm(hidden_size),af,
    Dense(hidden_size, 2)
)
#We use the fourier embedding of the timestep to encode the time
#But also for each dimension of the input
model(x,t) = decode(embed(x) 
.+ rff_x1(x[1,:])
.+ rff_x2(x[2,:])
.+ rff_t(log(t))
)

#The timestep schedule
timesteps = timeschedule(exp, 1.0f-6, 1.0f+1, 100)

#Define the diffusion process
P = OrnsteinUhlenbeckDiffusion(1.f0)

#NN training setup
batch_size = 16
ps = Flux.params(embed,decode)
opt = Flux.Optimiser(WeightDecay(0.0001f0), Adam(0.001f0))
cumu_loss = 0.0
training_losses = Float32[];

#Training loop
for iter in 1:250000
    #Generate one batch of training data
    x0 = hcat([target_sample() for i in 1:batch_size]...)
    #Sample t, then diffuse
    t = rand(timesteps)    
    xt = sampleforward(P, t, x0)
    
    #Loss and gradient
    l,gs = Flux.withgradient(ps) do
        x0hat = model(xt,t)
        standardloss(P, t, x0hat, x0)
    end
    
    #Update the parameters
    Flux.Optimise.update!(opt, ps, gs)

    #Plotting loss
    cumu_loss += l
    if mod(iter,100) == 0
        push!(training_losses, cumu_loss/100)
        cumu_loss = 0.0
        if mod(iter,5000) == 0
            display(plot(training_losses, yscale = :log10, ylabel = "Loss", xlabel = "Iters (x100)", label = :none))
        end
    end
end

N = 1000
xT = hcat([rand(eq_dist(P),2) for i in 1:N]...)

#Using the trained model to set the expectation
xzerohat_NN(x_t, t) = model(x_t,t)

#Run the backward diffution
track = Diffusions.Tracker()
@time x0diff = samplebackward(xzerohat_NN, P, timesteps, xT, tracker = track);

#Plot, alongside target
train = hcat([target_sample() for i in 1:500]...)
pl1 = scatter(train[1,:], train[2,:], markerstrokewidth = 0.0, color = "red", label = "True samples", alpha = 0.55)
pl2 = scatter(x0diff[1,:], x0diff[2,:], markerstrokewidth = 0.0, color = "blue", label = "Diffusion Samples", alpha = 0.55)
pl = plot(pl1, pl2, layout = (1,2), size = (600,300))

#Visualize the diffusion trajectory both for xt, and x0|xt
@gif for i in 1:length(track.time)
    scatter(track.data[i][1,:], track.data[i][2,:], markerstrokewidth = 0.0, axis = ([], false),
    color = "blue", label = L"x_t")
    scatter!(track.x0[i][1,:], track.x0[i][2,:], markerstrokewidth = 0.0, axis = ([], false),
    color = "red", label = L"x_0|x_t", xlim = (-3,4.2), ylim = (-3.5,5.2), alpha = 0.5, 
    markersize = 2.5, legend = :topleft)
end
