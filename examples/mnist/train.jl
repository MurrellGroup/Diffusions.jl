using BSON: @save
using Dates: Dates, now
using Diffusions
using Distributions: Uniform
using Flux.Data: DataLoader
using Flux.Losses: mse, logitcrossentropy
using MLDatasets: MNIST, FashionMNIST
using OneHotArrays: onehotbatch
using Optimisers: Optimisers, WeightDecay, Adam
using Printf: @printf, @sprintf
using Statistics: mean

include("model.jl")

n_epochs = 500
batchsize = 512

# relative weight of the classification loss relative to the reconstruction loss
λ = 0.01f0

# self-conditioned training
self_conditioned = true

# Data loading
if isempty(ARGS) || lowercase(ARGS[1]) == "mnist"
    dataset = MNIST
elseif lowercase(ARGS[1]) == "fashionmnist"
    dataset = FashionMNIST
else
    error("the dataset name must be MNIST or FashionMNIST")
end
preprocess((x, y)) = reshape(x, 28, 28, 1, :), y .+ 1
xtrain, ytrain = preprocess(dataset(:train)[:])
xtest, ytest = preprocess(dataset(:test)[:])
dataloader_train = DataLoader((xtrain, ytrain); batchsize)
dataloader_test = DataLoader((xtest, ytest); batchsize)

# Model and optimizer setup
model = UNet(1 + self_conditioned)
optim = Adam(1f-3)
state = Optimisers.setup(optim, model)

device = gpu
model = device(model)
state = device(state)

# Diffusion scheduling
θ = 4.0f0  # reversion
σ = √(2θ)  # volatility
diffusion = (OrnsteinUhlenbeckDiffusion(0, σ, θ), UniformDiscreteDiffusion(0.5f0, 10))

t_min = 1f-4
t_max = 1f+1
sampletime(n) = Float32.(exp.(rand(Uniform(log(t_min), log(t_max)), n)))

# Diffuse a batch of images and labels up to random times
function diffuse(x, y)
    t = sampletime(size(x, 4))
    x, y = sampleforward(diffusion, t, (x, y))
    x, y, t = device((x, y, t))
    if self_conditioned
        x_0, _ = model(cat(x, zero(x), dims = 3), y, t)
        # nullify images with 50% probability
        x_0[:,:,:,rand(size(x_0, 4)) .< 0.5] .= 0
        x = cat(x, x_0, dims = 3)
    end
    return (;x, y, t)
end

onehotlabel(y) = onehotbatch(y, 1:10)

starttime = now()
for epoch in 1:n_epochs
    loss_train = 0.0
    for (x, y) in dataloader_train
        diffused = diffuse(x, y)
        x, y = device((x, y))
        loss, grads = Flux.withgradient(model) do model
            x̂, ŷ = model(diffused.x, diffused.y, diffused.t)
            reconst = mse(x̂, x)
            class = logitcrossentropy(ŷ, onehotlabel(y))
            return reconst + λ * class
        end
        Optimisers.update!(state, model, grads[1])
        loss_train += loss
    end
    loss_train /= length(dataloader_train)

    loss_reconst = loss_class = 0.0
    for (x, y) in dataloader_test
        diffused = diffuse(x, y)
        x, y = device((x, y))
        x̂, ŷ = model(diffused.x, diffused.y, diffused.t)
        loss_reconst += mse(x̂, x)
        loss_class += logitcrossentropy(ŷ, onehotlabel(y))
    end
    loss_reconst /= length(dataloader_test)
    loss_class /= length(dataloader_test)
    loss_test = loss_reconst + λ * loss_class

    if epoch % 50 == 0
        let unet = cpu(model), name = @sprintf("%s-%03d.bson", lowercase(string(dataset)), epoch)
            @info "Saving weights $(name)"
            @save name unet
        end
    end

    elapsed = Dates.value(now() - starttime) / 1000
    @printf "%d: train=%f test=%f reconst=%f class=%f elapsed=%.1f\n" epoch loss_train loss_test loss_reconst loss_class elapsed
end
