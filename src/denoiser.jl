#This is the main denoising sampler.
function diffusion_sample(
    initial_state::State, #A draw from your process equilibrium
    P::Process, #The process itself
    expect!; #expect!(target_state,current_state,current_T) - a function that calculates, in-place, E[x_0|x_t]
    steps = 100, step_prop = 0.1, T = 5.0, final_noiseless_pred = true, track = false
)

    #Re-think these deepcopies. Maybe we implement a copy function that falls back on deepcopy if not implemented?
    g0 = deepcopy(initial_state) #The predicted value at zero
    g1F = deepcopy(initial_state) #The middle value during sampling
    g1B = deepcopy(initial_state) #The middle value during sampling
    g2 = deepcopy(initial_state) #The current value

    #For tracking the diffusion trajectory
    if track
        rev_steps = [T]
        rev_values = [values(g2)]
        target_preds = [values(g0)]
    end

    for i in 1:steps
        step = step_prop*T

        #calculate expectation
        expect!(g0,g2,T)
        
        #Sample one step back
        endpoint_conditioned_sample!(g0,g1F,g1B,g2,P,T,step)
        
        g2 = deepcopy(g1F)
        T = T - step
        if track
            push!(rev_steps,T)
            push!(rev_values,values(g1F))
            push!(target_preds,values(g0))
        end
    end

    #Optionally, one final step without noise added
    if final_noiseless_pred
        expect!(g0,g2,T)
    end

    #Return type changes here - not ideal, but nothing downstream of this will need to be type-stable anyway
    if track
        return g0, (rev_steps,rev_values,target_preds)
    else
        return g0
    end
end