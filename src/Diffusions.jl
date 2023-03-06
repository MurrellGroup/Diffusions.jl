module Diffusions
    using Distributions
    using LinearAlgebra
    using StatsBase
    using OneHotArrays: onehotbatch
    using Random: Random, AbstractRNG
    using Rotations
    using Quaternions

    include("types.jl")
    include("randomvariable.jl")
    include("continuous.jl")
    include("discrete.jl")
    include("angles.jl")
    include("JC.jl")
    include("rotational.jl")
    include("tracker.jl")
    include("utils.jl")
    include("randomfourierfeatures.jl")
    include("interface.jl")

    export
        #Processes
        OrnsteinUhlenbeckDiffusion,
        MultiGaussianState,
        WrappedBrownianDiffusion,
        WrappedInterpolatedBrownianDiffusion,
        IndependentDiscreteDiffusion,
        UniformDiscreteDiffusion,
        RotationDiffusion,
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
