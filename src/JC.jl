struct UniformDiscreteDiffusion{T <: Real} <: SamplingProcess
    rate::T
    k::Int
end

"""
    UniformDiscreteDiffusion(r::Real, k::Integer)

Create a discrete diffusion process with uniform jumps.

This process is a special case of [`IndependentDiscreteDiffusion`](@ref) with
the same rate `r` and `π = ones(k) / k`, but this is faster especially if `k` is
large because sampling can be performed in constant time.
"""
function UniformDiscreteDiffusion(r::Real, k::Integer)
    r = float(r)
    return UniformDiscreteDiffusion{typeof(r)}(r, k)
end

eq_dist(P::UniformDiscreteDiffusion) = Categorical(P.k)

function _sampleforward(rng::AbstractRNG, P::UniformDiscreteDiffusion, t::Real, X0::AbstractArray)
    Xt = copy(X0)
    event = 1-exp(-t*P.rate)
    for i in eachindex(Xt)
        if rand(rng) < event
            Xt[i] = rand(rng, 1:P.k)
        end
    end
    return Xt
end

#pls check for correctness, and make sure types are preserved
#This works by pre-calculating the five possible outcomes of the combine(fwd(X0),back(Xt)) of JC69
#Then only sampling when the random draw suggests the state will differ from Xt.
function _endpoint_conditioned_sample(rng::AbstractRNG, P::UniformDiscreteDiffusion, s::Real, t::Real, X0, Xt)
    Xs_full = copy(Xt)
    Xs = maskedvec(Xs_full)
    X0 = maskedvec(X0)
    Xt = maskedvec(Xt)
    k = P.k
    Fw = 1-exp(-s*P.rate)
    Bk = 1-exp(-(t-s)*P.rate)
    ssF = Fw/k+(1-Fw)
    ssB = Bk/k+(1-Bk)
    A = ssF * (1-ssB)/(k-1) #Same as X0
    B = ssB * (1-ssF)/(k-1) #Same as Xt
    C = (k-2)*((1-ssB)/(k-1))*((1-ssF)/(k-1)) #Different to both
    A,B,C = (A,B,C) ./ (A+B+C)
    D = ssF * ssB #Same as X0 == Xt
    E = (k-1)*((1-ssB)/(k-1))*((1-ssF)/(k-1)) #Different to X0 == Xt
    E = E/(D+E)
    for i in eachindex(Xt)
        if X0[i] == Xt[i]
            if rand(rng) < E
                #Draw from 1:k but excluding Xt[i]
                d = rand(rng,1:(k-1))
                Xs[i] = d>=X0[i] ? d+1 : d
            end
        else
            if rand(rng) < C
                #Draw from 1:k but excluding Xt[i] and X0[i]
                d = rand(rng,1:(k-2))
                if d>=X0[i]
                    d += 1
                end
                if d>=Xt[i]
                    d += 1
                end
                Xs[i] = d
            else
                if rand(rng) < A/(A+B)
                    Xs[i] = X0[i]
                end
            end
        end
    end
    return Xs_full
end

endpoint_conditioned_sample(rng::AbstractRNG, P::UniformDiscreteDiffusion, s::Real, t::Real, X0, Xt) = _endpoint_conditioned_sample(rng, P, s, t, X0, Xt)
endpoint_conditioned_sample(rng::AbstractRNG, P::UniformDiscreteDiffusion, s::Real, t::Real, X0::MaskedArray, Xt::MaskedArray) = _endpoint_conditioned_sample(rng, P, s, t, X0, Xt)
#When only the starting array is masked, we force X0 to inherit its mask, so that the X0 estimating function doesn't have to track the masking.
endpoint_conditioned_sample(rng::AbstractRNG, P::UniformDiscreteDiffusion, s::Real, t::Real, X0, Xt::MaskedArray) = _endpoint_conditioned_sample(rng, P, s, t, MaskedArray(copy(X0), copy(Xt.indices)), Xt)
