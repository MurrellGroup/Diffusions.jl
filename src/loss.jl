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

function standard_loss(
    P::RotationDiffusion{T},
    t::Union{T, AbstractVector{T}},
    x0::AbstractArray{T},
    x0hat::AbstractArray{T}; loss_scale = sqrt
) where T <: Real
    #s = @. loss_scale(1 - exp(-t*P.rate*5))  # this doen't work (https://github.com/FluxML/Zygote.jl/issues/1399)
    s = loss_scale.(1 .- exp.(.-t .* P.rate .* 5))
    return sum(weightbatch(rot_ang.(abs.(sum(x0 .* x0hat, dims = 1))), inv.(s))) / length(x0)
end

function min_ang(x1, x2)
    diff = abs(x1 - x2)
    return min(diff, oftype(diff, 2π) - diff)
end

function standard_loss(
    P::Diffusions.WrappedDiffusion{T},
    t::Union{T, AbstractVector{T}},
    x0::AbstractArray{T},
    x0hat::AbstractArray{T}; loss_scale = sqrt
) where T <: Real
    s = loss_scale.(1 .- exp.(.-t .* P.rate ./ 8))
    return sum(weightbatch(min_ang.(x0, x0hat).^2, inv.(2s))) / length(x0)
end

function standard_loss(
    P::OrnsteinUhlenbeckDiffusion{T},
    t::Union{T, AbstractVector{T}},
    x0::AbstractArray{T},
    x0hat::AbstractArray{T}; loss_scale = sqrt
) where T <: Real
    s = loss_scale.(1 .- exp.(.-t .* P.reversion)) #Need to fix this for non-N(0,1) equilibrium cases
    return sum(weightbatch(abs2.(x0 .- x0hat), inv.(s))) / length(x0)
end

crossent(ŷ,y) = mean(-sum(y .* log.(ŷ .+ 1.0f-10), dims = 1))

function standard_loss(
    P::IndependentDiscreteDiffusion{T},
    t::Union{T, AbstractVector{T}},
    x0,
    x0hat;
    #This "ce" should be logitcrossentropy when Flux is imported
    loss_scale=sqrt, ce=crossent
) where T <: Real
    k = length(P.π)
    s = loss_scale.(1 .- exp.(.-t .* P.r))
    return weightbatch(ce(x0, x0hat), inv.(s)) / (15 * ((k - 1) / k))
end

function standard_loss(
    P::UniformDiscreteDiffusion{T},
    t::Union{T, AbstractVector{T}},
    x0,
    x0hat;
    #This "ce" should be logitcrossentropy when Flux is imported
    loss_scale=sqrt, ce=crossent
) where T <: Real
    k = length(P.π)
    s = loss_scale(1 - exp(-t * P.rate))
    return weightbatch(ce(x0, x0hat), inv.(s)) / (15 * ((k - 1) / k))
end

# Weight A with w along the last dimension (i.e., batch dimension)
weightbatch(A::AbstractArray{T}, w::T) where T <: Real = A .* w
weightbatch(A::AbstractArray{T}, w::AbstractVector{T}) where T <: Real =
    A .* reshape(w, ntuple(i -> 1, ndims(A) - 1)..., :)
