function randrot(rng::AbstractRNG, σ²::Real)
    σ = √float(σ²)
    T = typeof(σ)
    return QuatRotation(exp(quat(0, randn(rng, T) * σ, randn(rng, T) * σ, randn(rng, T) * σ)))
end

#T is in units of var
function rotation_diffuse(rng::AbstractRNG, Rstart::QuatRotation, T::Real; max_var_step::Real = oftype(T, 0.05))
    remaining_var = T
    B = Rstart
    for t in max_var_step:max_var_step:T
        B *= randrot(rng, max_var_step)
        remaining_var = T - t
    end
    B *= randrot(rng, remaining_var)
    return B
end

#T is in units of var
function rotation_bridge(
    rng::AbstractRNG,
    Rstart::QuatRotation,
    Rend::QuatRotation,
    eps::Real,
    T::Real;
    max_var_step::Real = oftype(T, 0.05)
)
    B = rotation_diffuse(rng, Rstart, T - eps; max_var_step)
    C = rotation_diffuse(rng, B, eps; max_var_step)
    difference_rot = slerp(C, Rend, (T - eps) / T)
    return B * C' * difference_rot
end


struct RotationDiffusion{T <: Real} <: SamplingProcess
    rate::T
end

RotationDiffusion() = RotationDiffusion(1.0)

sampleforward(rng::AbstractRNG, process::RotationDiffusion, t::Real, x::AbstractArray) =
    rotation_diffuse.(rng, x, t * process.rate)

endpoint_conditioned_sample(rng::AbstractRNG, process::RotationDiffusion, s::Real, t::Real, x_0, x_t) =
    rotation_bridge.(rng, x_0, x_t, (t - s) * process.rate, t * process.rate)

function rots2flatquats(r::AbstractArray{QuatRotation{T}}) where T
    feats = zeros(T, 4, size(r)...)
    for ix in CartesianIndices(r)
        q = r[ix].q
        feats[1,ix] = q.s
        feats[2,ix] = q.v1
        feats[3,ix] = q.v2
        feats[4,ix] = q.v3
    end
    return feats
end

#bcd is the vector component of the quaternion that is a useful NN output.
#bcds2flatquats will convert a 3-by-N to array of bcds to an array representation of N unit quaternions
#This is useful for the rotationally-aware loss function to avoid actual Quaternion types
function bcds2flatquats(bcd::Array{<: Real, 2})
    denom = sqrt.(1 .+ bcd[1,:].^2 .+ bcd[2,:].^2 .+ bcd[3,:].^2)
    return vcat((1 ./ denom)', bcd ./ denom')
end
#Need these to work when there is a batch dim
flatquats2rots(flat::Array{<: Real, 2}) = [QuatRotation(c) for c in eachcol(flat)]
bcds2rots(bcd::Array{<: Real, 2}) = flatquats2rots(bcds2flatquats(bcd))

function flatquats(x)
    s(r) = r.q.s
    v1(r) = r.q.v1
    v2(r) = r.q.v2
    v3(r) = r.q.v3
    return stack([s.(x), v1.(x), v2.(x), v3.(x)], dims = 1)
end
