export 
    AngleDiffusionProcess, 
    MultiAngleState,
    normangle

function rewrap(x,lb,ub)
    mod(x-lb,ub-lb)+lb
end

normangle(v) = rewrap(v, -pi, pi)

function randangle(σsq)
    σ = sqrt(σsq)
    return randn() * σ
end

# T is in units of var
angle_diffuse(A, T) =  A + randangle(T)

# T is in units of var
function angle_bridge(Astart, Aend, eps, T)
    B = angle_diffuse(Astart, T - eps)
    C = angle_diffuse(B, eps)
    C = normangle(C)
    Aend = normangle(Aend)
    if Aend > C
        if Aend - C < pi
            shortestdist = Aend - C
            dir = 1
        else
            shortestdist = 2pi - (Aend - C)
            dir = -1
        end
    else
        if C - Aend < pi
            shortestdist = C - Aend
            dir = -1
        else
            shortestdist = 2pi - (C - Aend)
            dir = 1
        end
    end
    return normangle(B + dir * shortestdist * (T-eps)/T)
end

mutable struct AngleDiffusionProcess <: SimulationProcess
    rate::Float64
    function AngleDiffusionProcess(rate)
        new(rate)
    end
    function AngleDiffusionProcess()
        new(1.0)
    end
end

mutable struct MultiAngleState <: ContinuousState
    angles::Array{Float64}
    function MultiAngleState(dims...)
        new(zeros(Float64,dims...))
    end
end

function forward_sample!(end_state,init_state,P::AngleDiffusionProcess,T; max_var_step = 0.05)
    for ix in CartesianIndices(init_state.angles)
        end_state.angles[ix] = angle_diffuse(init_state.angles[ix], T*P.rate)
    end
end

#We don't use g1B here, but it is present because it saves copying for other kinds of process.
#Could likely be improved
function endpoint_conditioned_sample!(g0::MultiAngleState,g1F::MultiAngleState,g1B::MultiAngleState,g2::MultiAngleState,P::AngleDiffusionProcess,T,eps; max_var_step=0.05)
    for ix in CartesianIndices(g0.angles)
        g1F.angles[ix] = angle_bridge(g0.angles[ix], g2.angles[ix], eps*P.rate, T*P.rate)
    end
end

values(x::MultiAngleState) = copy(x.angles)
