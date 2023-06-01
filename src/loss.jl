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
#=
scalebatch(A::AbstractArray, s::Real) = A ./ s
scalebatch(A::AbstractArray, s::AbstractVector{<: Real}) =
    A ./ reshape(s, ntuple(i -> 1, ndims(A) - 1)..., :)
scalebatch(A::AbstractArray, s::AbstractVector{<: Real}) =
    A ./ repeat(reshape(s, ntuple(i -> 1, ndims(A) - 1)..., :), Base.front(size(A))..., 1)
=#

#scaledloss(loss, indices, s) = mean(scalebatch(loss, s)[indices])
scaledloss(loss, indices, s::Real) = mean(loss[indices] ./ s)
scaledloss(loss, indices, s::AbstractVector{<: Real}) =
    mean(loss[indices] ./ repeat(reshape(s, ntuple(i -> 1, ndims(loss) - 1)..., :), Base.front(size(loss))..., 1)[indices])

maskedindices(x::MaskedArray) = x.indices
maskedindices(x::AbstractArray) = eachindex(x)

defaultscaler(p::OrnsteinUhlenbeckDiffusion, t::Real) = sqrt(1 - exp(-t * p.reversion))

function standardloss(
    p::OrnsteinUhlenbeckDiffusion,
    t::Union{Real,AbstractVector{<:Real}},
    x̂, x;
    scaler=defaultscaler)
    loss(x̂, x) = abs2.(x̂ .- x)
    # ugly syntax but scaler.(p, t) is not differentiable with Zygote.jl for some reason
    return scaledloss(loss(x̂, parent(x)), maskedindices(x), (t -> scaler(p, t)).(t))
end

defaultscaler(p::RotationDiffusion, t::Real) = sqrt(1 - exp(-t * p.rate * 5))

function standardloss(
    p::RotationDiffusion,
    t::Union{Real,AbstractVector{<:Real}},
    x̂, x;
    scaler=defaultscaler)
    loss(x̂, x) = rotang.(abs.(dropdims(sum(x̂ .* x, dims = 1), dims = 1)))
    return scaledloss(loss(x̂, flatquats(parent(x))), maskedindices(x), (t -> scaler(p, t)).(t)) / 4
end

rotang(x) = (1 - x) * (x - 13)^2 / 18


defaultscaler(p::WrappedDiffusion, t::Real) = 2 * sqrt(1 - exp(-t * p.rate / 8))

function standardloss(
    p::WrappedDiffusion,
    t::Union{Real, AbstractVector{<: Real}},
    x̂, x;
    scaler = defaultscaler
)
    loss(x̂, x) = abs2.(minang.(x̂, x))
    return scaledloss(loss(x̂, parent(x)), maskedindices(x), (t -> scaler(p, t)).(t))
end

function minang(x1, x2)
    diff = abs(x1 - x2)
    return min(diff, oftype(diff, 2π) - diff)
end


defaultscaler(p::UniformDiscreteDiffusion, t::Real) = sqrt(1 - exp(-t * p.rate))

function standardloss(
    p::UniformDiscreteDiffusion,
    t::Union{Real,AbstractVector{<:Real}},
    x̂, x;
    scaler=defaultscaler
)
    k = p.k
    loss(x̂, x) = dropdims(logitcrossentropy(x̂, x), dims = 1)
    return scaledloss(loss(x̂, onehotbatch(parent(x), 1:k)), maskedindices(x), (t -> scaler(p, t)).(t)) / ((k - 1) / k) * 1.44f0
end

logitcrossentropy(x̂, x; dims = 1) = -sum(x .* logsoftmax(x̂; dims); dims)


defaultscaler(p::IndependentDiscreteDiffusion, t::Real) = sqrt(1 - exp(-t * p.r))

function standardloss(
    p::IndependentDiscreteDiffusion{K},
    t::Union{Real, AbstractVector{<: Real}},
    x̂, x;
    scaler = defaultscaler
) where K
    loss(x̂, x) = dropdims(logitcrossentropy(x̂, x), dims = 1)
    # TODO: This isn't differentiable on GPU.
    xx = stack([getindex.(parent(x), i) for i in 1:K], dims = 1)
    return scaledloss(loss(x̂, xx), maskedindices(x), (t -> scaler(p, t)).(t)) / ((K - 1) / K) * 1.44f0
end
