#To-do notes:
#1: Think about whether we can get a perf boost by removing variances from all State types. We only need to endpoint-condition sample, so we don't need to store variances.
#They can be computed temporarily, when needed, for some of the state types. This stops us sharing code between Gaussian and Discrete states, though, so might not be worth it.
#2: Make the dimensionality of the discrete state flexible, in the same way as the MultiGaussianState is. Do we want the first or last dim to be the "states" dim?
#The CTMC prop should be written to do this all in one big matrix/vector op.
#3: Add flexible type sigs to all of these so that they play nice with NN training. We don't need the diffusion to run on the GPU (or do we? Need to think about this), but they should at least not be doing loads of type conversion.

module Diffusions
    using Distributions
    using LinearAlgebra
    using StatsBase
    using OneHotArrays: onehotbatch
    using Random: Random, AbstractRNG
    using Rotations
    using Quaternions
    using Functors: @functor

    include("types.jl")
    include("randomvariable.jl")
    include("continuous.jl")
    include("discrete.jl")
    include("tractables.jl")
    include("JC.jl")
    include("rotational.jl")
    include("denoiser.jl")
    include("tracker.jl")
    include("utils.jl")
    include("randomfourierfeatures.jl")
    include("interface.jl")

    export
        #Processes
        OrnsteinUhlenbeck,
        IJ,
        UniformDiscreteDiffusion,
        RotDiffusionProcess,
        #functions
        eq_dist,
        sampleforward,
        samplebackward,
        randcat,
        rotation_features,
        #utils
        RandomFourierFeatures,
        sqrt_schedule,
        log_schedule

end
