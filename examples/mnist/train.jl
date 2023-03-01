using BSON: @save
using Dates: Dates, now
using Diffusions: OrnsteinUhlenbeck, IJ, sampleforward
using Distributions: Uniform
using Flux.Data: DataLoader
using Flux.Losses: mse, logitcrossentropy
using MLDatasets: MNIST, FashionMNIST
using OneHotArrays: onehotbatch
using Optimisers: Optimisers, WeightDecay, Adam
using Printf: @printf, @sprintf
using Statistics: mean

include("model.jl")

n_epochs = 100
batchsize = 512

# relative weight of the classification loss relative to the reconstruction loss
λ = 1

# self-conditioned training
self_conditioned = true


# Data loading
if isempty(ARGS) || ARGS[1] == "MNIST"
    dataset = MNIST
elseif ARGS[1] == "FashionMNIST"
    dataset = FashionMNIST
else
    error("the dataset name must be MNIST or FashionMNIST")
end
preprocess((x, y)) = reshape(x, 28, 28, 1, :), onehotbatch(y, 0:9)
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
p = ones(Float32, 10) ./ 10  # equilibrium distribution
process = (OrnsteinUhlenbeck(0, 1, 0.5f0), IJ(2.0f0, p))

t_min = 1f-4
t_max = 5.0f0
sampletime() = Float32(exp(rand(Uniform(log(t_min), log(t_max)))))

# Diffuse a batch of images and labels up to random times
function diffuse(x, y)
    n = size(x, 4)
    diffused = (x = similar(x), y = similar(y), t = zeros(Float32, n))
    for i in 1:n
        time = sampletime()
        diffused.t[i] = time
        diffused.x[:,:,:,i], diffused.y[:,i] =
            sampleforward(process, time, (x[:,:,:,i], y[:,i]))
    end
    return diffused
end

function f(model, diffused)
    x = diffused.x
    x_0, _ = model(cat(x, zero(x), dims = 3), diffused.y, diffused.t)
    x_0 = sigmoid.(x_0)
    x_0[:,:,:,rand(size(x_0, 4)) .< 0.5] .= 0  # nullify images with 50% probability
    return (x = cat(x, x_0, dims = 3), y = diffused.y, t = diffused.t)
end

starttime = now()
for epoch in 1:n_epochs
    loss_train = 0.0
    for (x, y) in dataloader_train
        diffused = diffuse(x, y)
        x, y, diffused = device((x, y, diffused))
        if self_conditioned
            diffused = f(model, diffused)
        end
        loss, grads = Flux.withgradient(model) do model
            x̂, ŷ = model(diffused.x, diffused.y, diffused.t)
            reconst = mse(sigmoid.(x̂), x)
            class = logitcrossentropy(ŷ, y)
            return reconst + λ * class
        end
        Optimisers.update!(state, model, grads[1])
        loss_train += loss
    end
    loss_train /= length(dataloader_train)

    loss_reconst = loss_class = 0.0
    for (x, y) in dataloader_test
        diffused = diffuse(x, y)
        x, y, diffused = device((x, y, diffused))
        if self_conditioned
            diffused = f(model, diffused)
        end
        x̂, ŷ = model(diffused.x, diffused.y, diffused.t)
        loss_reconst += mse(sigmoid.(x̂), x)
        loss_class += logitcrossentropy(ŷ, y)
    end
    loss_reconst /= length(dataloader_test)
    loss_class /= length(dataloader_test)
    loss_test = loss_reconst + λ * loss_class

    if epoch % 50 == 0
        let unet = cpu(model), name = @sprintf "mnist-unet-%03d.bson" epoch
            @save name unet
        end
    end

    elapsed = Dates.value(now() - starttime) / 1000
    @printf "%d: train=%f test=%f reconst=%f class=%f elapsed=%.1f\n" epoch loss_train loss_test loss_reconst loss_class elapsed
end
