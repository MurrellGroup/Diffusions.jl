#These are a collection of loss functions for use with Diffusions.jl.
#There are arbitrary constants everywhere. These aim to make the loss magnitudes similar across the different processes,
#for the meaningful range of t values. loss_scale = x -> x would balance the loss across values of t, 
#but empirically that doesn't seem to work as well as loss_scale = sqrt, likely because it undervalues the
#importance of high t values for learning to converge to the target distribution, where errors for small t matter
#less. loss_scale = sqrt also lines up with the scaling from the standard diffusion denoising setup, IIUIC, WIMNIIO.

#-----Playing with rotations------
#This gives the square of the expansion of the angle around 1
#Matches acos^2 for values between 0.5 and 1 (where 1 is no angular difference)
#Kinda like huber loss for rotation?
#It avoids the grad singularity when angles are very far apart
rot_ang(x) = oftype(x, 1/18)*(1-x)*(x-13)^2
#rot_ang(x) = acos(x)^2 #Canonical
#rot_ang(x) = -x #Dumb
#rot_ang(x) = -8(x-1) + (4/3)*(x-1)^2 - (16/45)*(x-1)^3 + (4/35)*(x-1)^4 #Longer expansion
#-----Playing with rotations------


# Scale A with s along the last dimension (i.e., batch dimension)
scalebatch(A::AbstractArray, s::Real) = A ./ s
scalebatch(A::AbstractArray, s::AbstractVector{<: Real}) =
    A ./ reshape(s, ntuple(i -> 1, ndims(A) - 1)..., :)

# Calculate the scaled loss (`loss` is an element-wise loss function)
scaledloss(loss, x̂, x::MaskedArray, s) = mean(scalebatch(loss(x̂, x.data), s)[x.indices])
scaledloss(loss, x̂, x::AbstractArray, s) = mean(scalebatch(loss(x̂, x), s))


defaultscaler(p::OrnsteinUhlenbeckDiffusion) = t -> sqrt(1 - exp(-t * p.reversion))

function standardloss(
    p::OrnsteinUhlenbeckDiffusion,
    t::Union{Real, AbstractVector{<: Real}},
    x̂, x;
    scaler = defaultscaler(p)
)
    loss(x̂, x) = abs2.(x̂ .- x)
    return scaledloss(loss, x̂, x, scaler.(t))
end

defaultscaler(p::RotationDiffusion) = t -> sqrt(1 - exp(-t * p.rate * 5))

function standardloss(
    p::RotationDiffusion,
    t::Union{Real, AbstractVector{<: Real}},
    x̂, x;
    scaler = defaultscaler(p)
)
    loss(x̂, x) = rotang.(abs.(sum(x̂ .* x, dims = 1)))
    return scaledloss(loss, x̂, x, scaler.(t)) / size(x, 1)
end

rotang(x) = 1//18 * (1 - x) * (x - 13)^2


defaultscaler(p::WrappedDiffusion) = t -> 2 * sqrt(1 - exp(-t * p.rate / 8))

function standardloss(
    p::WrappedDiffusion,
    t::Union{Real, AbstractVector{<: Real}},
    x̂, x;
    scaler = defaultscaler(p)
)
    loss(x̂, x) = abs2.(minang.(x̂, x))
    return scaledloss(loss, x̂, x, scaler.(t))
end

function minang(x1, x2)
    diff = abs(x1 - x2)
    return min(diff, oftype(diff, 2π) - diff)
end


defaultscaler(p::UniformDiscreteDiffusion) = t -> sqrt(1 - exp(-t * p.rate))

function standardloss(
    p::UniformDiscreteDiffusion,
    t::Union{Real, AbstractVector{<: Real}},
    x̂, x;
    scaler = defaultscaler(p)
)
    loss(x̂, x) = logitcrossentropy(x̂, x)
    return scaledloss(loss, x̂, x, scaler.(t)) / ((p.k - 1) / p.k) * 1.44f0
end

logitcrossentropy(x̂, x; dims = 1) = -sum(x .* logsoftmax(x̂; dims); dims)


defaultscaler(p::IndependentDiscreteDiffusion) = t -> sqrt(1 - exp(-t * p.r))

function standardloss(
    p::IndependentDiscreteDiffusion{K},
    t::Union{Real, AbstractVector{<: Real}},
    x̂, x;
    scaler = defaultscaler(p)
) where K
    loss(x̂, x) = logitcrossentropy(x̂, x)
    return scaledloss(loss, x̂, x, scaler.(t)) / ((K - 1) / K) * 1.44f0
end
