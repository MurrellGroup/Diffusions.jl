using BSON: @save
using Dates: Dates, now
using Diffusions: OrnsteinUhlenbeck, IJ, sampleforward
using Distributions: Uniform
using Flux.Data: DataLoader
using Flux.Losses: mse, logitbinarycrossentropy, logitcrossentropy
using MLDatasets: MNIST
using OneHotArrays: onehotbatch
using Optimisers: Optimisers, WeightDecay, Adam
using Printf: @printf, @sprintf
using Statistics: mean

include("model.jl")

n_epochs = 50
batchsize = 128

preprocess((x, y)) = reshape(x, 28, 28, 1, :), onehotbatch(y, 0:9)
xtrain, ytrain = preprocess(MNIST(:train)[:])
xtest, ytest = preprocess(MNIST(:test)[:])
dataloader_train = DataLoader((xtrain, ytrain); batchsize)
dataloader_test = DataLoader((xtest, ytest); batchsize)

t_min = 1f-4
t_max = 5.0f0
sampletime() = Float32(exp(rand(Uniform(log(t_min), log(t_max)))))

p = ones(Float32, 10) ./ 10  # equilibrium distribution
process = (OrnsteinUhlenbeck(0, 1, 0.5f0), IJ(2.0f0, p))

model = UNet()
optim = Optimisers.OptimiserChain(WeightDecay(1f-3), Adam(1f-4))
state = Optimisers.setup(optim, model)

device = gpu
model = device(model)
state = device(state)

# relative weight of the classification loss relative to the reconstruction loss
λ = 10.0f0

starttime = now()
for epoch in 1:n_epochs
    agg = sum

    loss_train = 0.0
    for (x, y) in dataloader_train
        b = size(x)[end]
        t = sampletime()
        scale = sqrt(1 - exp(-t))
        diffused = sampleforward(process, t, (x, y))
        t = device(fill(t, b))
        x, y, diffused = device(x), device(y), device(diffused)
        loss, grads = Flux.withgradient(model) do model
            x̂, ŷ = model(diffused, t)
            reconst = mse(sigmoid.(x̂), x; agg)
            class = logitcrossentropy(ŷ, y; agg)
            return (reconst + λ * class) / scale
        end
        isfinite(loss) && Optimisers.update!(state, model, grads[1])
        loss_train += loss
    end

    loss_reconst = loss_class = 0.0
    for (x, y) in dataloader_test
        b = size(x)[end]
        t = sampletime()
        scale = sqrt(1 - exp(-t))
        diffused = sampleforward(process, t, (x, y))
        t = device(fill(t, b))
        x, y, diffused = device(x), device(y), device(diffused)
        x̂, ŷ = model(diffused, t)
        loss_reconst += mse(sigmoid.(x̂), x; agg) / scale
        loss_class += logitcrossentropy(ŷ, y; agg) / scale
        n += 1
    end
    loss_test = loss_reconst + λ * loss_class

    elapsed = Dates.value(now() - starttime) / 1000
    @printf "%d: train=%f test=%f reconst=%f class=%f elapsed=%.1f\n" epoch loss_train loss_test loss_reconst loss_class elapsed

    if epoch % 50 == 0
        let unet = cpu(model), name = @sprintf "mnist-unet-%03d.bson" epoch
            @save name unet
        end
    end
end
