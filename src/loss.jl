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
        t::T,
        x0::Array{T,2},
        x0hat::Array{T,2}; loss_scale = sqrt) where T <: Real
    s = loss_scale(1 - exp(-t*P.rate*5))
    (sum(rot_ang.(abs.(sum(x0 .* x0hat, dims = 1))))/length(x0))/s
end

min_ang(x1,x2) = min(abs(x1-x2),2pi-abs(x1-x2))

function standard_loss(
        P::Diffusions.WrappedDiffusion{T},
        t::T,
        x0::AbstractArray{T},
        x0hat::AbstractArray{T}; loss_scale = sqrt) where T <: Real
    s = loss_scale(1 - exp(-t*P.rate/8))
    (sum(min_ang.(x0,x0hat).^2)/length(x0))/(2s)
end

function standard_loss(
        P::OrnsteinUhlenbeckDiffusion{T},
        t::T,
        x0::AbstractArray{T},
        x0hat::AbstractArray{T}; loss_scale = sqrt) where T <: Real
    
    s = loss_scale(1 - exp(-t*P.reversion)) #Need to fix this for non-N(0,1) equilibrium cases
    (sum(abs2.(x0 .- x0hat))/length(x0))/s
end

crossent(ŷ,y) = mean(-sum(y .* log.(ŷ .+ 1.0f-10), dims = 1))

function standard_loss(
        P::IndependentDiscreteDiffusion{T},
        t::T,
        x0,
        x0hat;
        #This "ce" should be logitcrossentropy when Flux is imported
        loss_scale = sqrt, ce = crossent) where T <: Real
    k = length(P.π)
    s = loss_scale(1 - exp(-t*P.r))
    ce(x0,x0hat)/(s*15*((k-1)/k))
end

function standard_loss(
        P::UniformDiscreteDiffusion{T},
        t::T,
        x0,
        x0hat;
        #This "ce" should be logitcrossentropy when Flux is imported
        loss_scale = sqrt, ce = crossent) where T <: Real
    k = length(P.π)
    s = loss_scale(1 - exp(-t*P.rate))
    ce(x0,x0hat)/(s*15*((k-1)/k))
end