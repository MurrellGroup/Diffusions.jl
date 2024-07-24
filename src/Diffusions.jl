module Diffusions
    using Distributions
    using LinearAlgebra
    using OneHotArrays: onehotbatch
    using Random: Random, AbstractRNG
    using Rotations
    using Quaternions
    using InverseFunctions: inverse, square
    using StaticArrays: SVector
    using NNlib: logsoftmax
    using Adapt: Adapt
    using MLUtils: flatten

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
    include("maskedarrays.jl")
    include("interface.jl")
    include("loss.jl")

    export
        #Processes
        OrnsteinUhlenbeckDiffusion,
        MultiGaussianState,
        WrappedBrownianDiffusion,
        WrappedInterpolatedBrownianDiffusion,
        IndependentDiscreteDiffusion,
        UniformDiscreteDiffusion,
        RotationDiffusion,
        MaskedDiscreteDiffusion,
        #Diffusion functions
        sampleforward,
        samplebackward,
        eq_dist,
        #utils
        randcat,
        RandomFourierFeatures,
        timeschedule,
        square,
        Tracker,
        reangle,
        standardloss,
        rots2flatquats,
        bcds2flatquats,
        bcds2rots,
        flatquats2rots


end
