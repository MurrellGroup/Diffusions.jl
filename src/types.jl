#Type system
abstract type Process end

Base.broadcastable(x::Process) = Ref(x)

abstract type DiffusionProcess <: Process end #Can propogate uncertainty with forward! and backward!
abstract type SimulationProcess <: Process end #Only deals with point masses. We might switch to everything using point masses, so we might not need this layer

#Where the state, at any point, is a gaussian
abstract type GaussianStateProcess <: DiffusionProcess end
abstract type DiscreteStateProcess <: DiffusionProcess end

abstract type State end
abstract type ContinuousState <: State end
abstract type DiscreteState <: State end