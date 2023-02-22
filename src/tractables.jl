#Used across different state types, where they can handle their variances in closed form (ie. where forward!, backward!, and sample! are implemented)
function endpoint_conditioned_sample!(g0,g1F,g1B,g2,P::DiffusionProcess,T,step)
    #Prop forward, from zero to T-step, from this expectation
    forward!(g1F,g0,P,T-step)

    #Prop backward, step, from T to T-step.
    backward!(g1B,g2,P,step)

    #Combine fwd and back Gaussians, which are both at T-step, and sample
    combine!(g1F,g1B)
    sample!(g1F)
end

function forward_sample!(end_state,init_state,P::DiffusionProcess,T)
    forward!(end_state,init_state,P,T)
    sample!(end_state)
end