export
    RotDiffusionProcess,
    MultiRotationState,
    features

function randrot(σsq)
    σ = sqrt(σsq)
    QuatRotation(exp(quat(0, randn() * σ, randn() * σ, randn() * σ)))
end

Quaternions.slerp(qa::QuatRotation{Float64},qb::QuatRotation{Float64},t) = QuatRotation(slerp(qa.q,qb.q,t))

#T is in units of var
function rotation_diffuse(Rstart, T; max_var_step = 0.05)
    remaining_var = T
    B = copy(Rstart)
    for t in max_var_step:max_var_step:T
        B = B * randrot(max_var_step)
        remaining_var = T-t
    end
    B = B * randrot(remaining_var)
    return B
end

#T is in units of var
function rotation_bridge(Rstart, Rend, eps, T; max_var_step = 0.05)
    B = rotation_diffuse(Rstart, T - eps, max_var_step = max_var_step)
    C = rotation_diffuse(B, eps, max_var_step = max_var_step)
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
        end_state.rots[ix] = rotation_diffuse(init_state.rots[ix], T*P.rate)
    end
end

#We don't use g1B here, but it is present because it saves copying for other kinds of process.
#Could likely be improved
function endpoint_conditioned_sample!(g0::MultiRotationState,g1F::MultiRotationState,g1B::MultiRotationState,g2::MultiRotationState,P::RotDiffusionProcess,T,eps)
    for ix in CartesianIndices(g0.rots)
        g1F.rots[ix] = rotation_bridge(g0.rots[ix], g2.rots[ix], eps*P.rate, T*P.rate)
    end
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
