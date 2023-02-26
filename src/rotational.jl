export
    RotDiffusionProcess,
    MultiRotationState,
    features

function randrot(rng, σsq)
    σ = sqrt(σsq)
    QuatRotation(exp(quat(0, randn(rng) * σ, randn(rng) * σ, randn(rng) * σ)))
end

Quaternions.slerp(qa::QuatRotation{Float64},qb::QuatRotation{Float64},t) = QuatRotation(slerp(qa.q,qb.q,t))

#T is in units of var
function rotation_diffuse(rng, Rstart, T; max_var_step = 0.05)
    remaining_var = T
    B = copy(Rstart)
    for t in max_var_step:max_var_step:T
        B = B * randrot(rng, max_var_step)
        remaining_var = T-t
    end
    B = B * randrot(rng, remaining_var)
    return B
end

#T is in units of var
function rotation_bridge(rng, Rstart, Rend, eps, T; max_var_step = 0.05)
    B = rotation_diffuse(rng, Rstart, T - eps, max_var_step = max_var_step)
    C = rotation_diffuse(rng, B, eps, max_var_step = max_var_step)
    difference_rot = slerp(C, Rend, (T-eps)/T)
    return B * C' * difference_rot
end


mutable struct RotDiffusionProcess <: SimulationProcess
    rate::Float64
    function RotDiffusionProcess(rate)
        new(rate)
    end
    function RotDiffusionProcess()
        new(1.0)
    end
end

mutable struct MultiRotationState <: ContinuousState
    rots::Array{QuatRotation}
    function MultiRotationState(dims...)
        new(ones(QuatRotation,dims...))
    end
end

function forward_sample!(end_state,init_state,P::RotDiffusionProcess,T)
    for ix in CartesianIndices(init_state.rots)
        end_state.rots[ix] = rotation_diffuse(rng, init_state.rots[ix], T*P.rate)
    end
end

function sampleforward(rng::AbstractRNG, process::RotDiffusionProcess, t::Real, x)
    x_t = MultiRotationState(size(x.rots)...)
    for i in eachindex(x.rots)
        x_t.rots[i] = rotation_diffuse(rng, x.rots[i], t * process.rate)
    end
    return x_t
end

function endpoint_conditioned_sample(rng::AbstractRNG, process::RotDiffusionProcess, s::Real, t::Real, x_0, x_t)
    x_s = MultiRotationState(size(x_0.rots)...)
    for i in eachindex(x_0.rots)
        x_s.rots[i] = rotation_bridge(rng, x_0.rots[i], x_t.rots[i], (t - s) * process.rate, t * process.rate)
    end
    return x_s
end

values(r::MultiRotationState) = copy(r.rots)

#In case you want to use the quats as inputs to a NN
flatquat(q::Quaternion) = [q.s, q.v1, q.v2, q.v3]

function features(r::MultiRotationState)
    feats = zeros(4,size(r.rots)...)
    for ix in CartesianIndices(r.rots)
        feats[:,ix] .= flatquat(r.rots[ix].q)
    end
    return feats
end
