#To-do notes:
#1: Think about whether we can get a perf boost by removing variances from all State types. We only need to endpoint-condition sample, so we don't need to store variances.
#They can be computed temporarily, when needed, for some of the state types. This stops us sharing code between Gaussian and Discrete states, though, so might not be worth it.
#2: Make the dimensionality of the discrete state flexible, in the same way as the MultiGaussianState is. Do we want the first or last dim to be the "states" dim?
#The CTMC prop should be written to do this all in one big matrix/vector op.
#3: Add flexible type sigs to all of these so that they play nice with NN training. We don't need the diffusion to run on the GPU (or do we? Need to think about this), but they should at least not be doing loads of type conversion.

module Diffusions
    using Requires

    using Distributions
    using LinearAlgebra
    using StatsBase

    include("types.jl")
    include("continuous.jl")
    include("discrete.jl")
    include("tractables.jl")
    include("denoiser.jl")

    #Optional dependencies
    function __init__()
        #Have Rotations and Quaternions installed if you want to do rotational diffusion
        @require Rotations = "6038ab10-8711-5258-84ad-4b1120ba62dc" begin
            @require Quaternions = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0" begin
                using .Rotations
                using .Quaternions
                include("rotational.jl")
            end
        end
    end

    export
        #Types
        Process,
        DiffusionProcess,
        SimulationProcess,
        GaussianStateProcess,
        DiscreteStateProcess,
        State,
        ContinuousState,
        DiscreteState,
        #Continuous
        OrnsteinUhlenbeck,
        MultiGaussianState,
        #Discrete
        IJ,
        DiscreteState,
        #functions
        diffusion_sample,
        endpoint_conditioned_sample!,
        forward!,
        backward!,
        combine!,
        forward_sample!,
        sample!,
        eq_dist,
        values,
        var

end
