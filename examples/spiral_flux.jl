using Distributions, Plots, Flux, LaTeXStrings, Diffusions

function target_sample()
    l = rand()*10
    return [exp(-l/7)*sin(l) ,exp(-l/7)*cos(l)] * 5.0
end

loss(predY,Y) = sum((Y .- predY).^2)/length(Y)

af = swish
hidden_size = 16
model = Chain(
    Dense(3 => hidden_size, af), 
    Dense(hidden_size => 2hidden_size, sin),
    Dense(2hidden_size => 4hidden_size, af),
    Dense(4hidden_size => 2hidden_size, af),
    Dense(2hidden_size => hidden_size, af),
    Dense(hidden_size => 2))

P = OrnsteinUhlenbeck(0.0,1.0,0.5)
maxT = 5.0
batch_size = 16
input = zeros(3,batch_size)
d = (2,batch_size)
init_state = MultiGaussianState(zeros(d),zeros(d))
end_state = MultiGaussianState(zeros(d),zeros(d))


#Training the expectation model
opt = Flux.Optimiser(WeightDecay(0.001),Adam())

#To collect loss over iterations
training_losses = []
cumu_loss = 0.0

ps = Flux.params(model);
for iter in 1:200000
    #Setup one batch of diffused training data
    init_state.mean .= hcat([target_sample() for i in 1:batch_size]...)
    T = exp(rand(Uniform(log(0.0001),log(maxT))))
    forward_sample!(end_state,init_state,P,T)
    scal = sqrt(1 - exp(-T))
    input[1:2,:] .= end_state.mean
    input[3,:] .= T
    
    #Loss and gradient
    l,gs = Flux.withgradient(ps) do
       #This normalization tells the model that it should care about errors
       #proportional to the std of the noise for that T. For an oracle, this shouldn't matter,
       #but can make a difference in practice.
       loss(model(input),init_state.mean)/scal
    end
    cumu_loss += l
    #The give these to the optimizer
    Flux.Optimise.update!(opt, ps, gs)
    
    #Plotting loss
    if mod(iter,2000) == 0
        println(cumu_loss)
        push!(training_losses, cumu_loss/50)
        cumu_loss = 0.0
    end
end


#Using the trained model to set the expectation
function NN_expect!(g0,end_state,T)
    input = zeros(3,size(end_state.mean,2))
    input[1:2,:] .= end_state.mean
    input[3,:] .= T
    g0.mean .= model(input) 
    g0.var .= 0.0
end

#Sampling
d = (2,500)
init_state = MultiGaussianState(zeros(d),zeros(d))
#Initializing at the implied equilibrium
init_state.var .= var(P)
sample!(init_state);

@time samp, tracking = diffusion_sample(init_state,P,NN_expect!, step_prop = 0.01, steps = 1000, track = true);

train = hcat([target_sample() for i in 1:500]...)
pl1 = scatter(train[1,:],train[2,:], markerstrokewidth = 0.0, color = "red", label = "True samples", alpha = 0.55)
pl2 = scatter(samp.mean[1,:],samp.mean[2,:], markerstrokewidth = 0.0, color = "blue", label = "Diffusion Samples", alpha = 0.55)
pl = plot(pl1,pl2, layout = (1,2), size = (600,300))
