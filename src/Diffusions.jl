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
    include("angles.jl")
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
        MultiGaussianState,
        WrappedBrownianMotion,
        WrappedInterpolatedBrownianMotion,
        IJ,
        UniformDiscreteDiffusion,
        RotDiffusionProcess,
        #Diffusion functions
        sampleforward,
        samplebackward,
        eq_dist,
        #utils
        randcat,
        rotation_features,
        RandomFourierFeatures,
        sqrt_schedule,
        log_schedule,
        reangle

end
