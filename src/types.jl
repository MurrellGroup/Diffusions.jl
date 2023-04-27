#Type system
abstract type Process end

Base.broadcastable(x::Process) = Ref(x)

abstract type TractableProcess <: Process end #Tractable uncertainty propogation
abstract type SamplingProcess <: Process end #Only deals with point masses and sampling

abstract type GaussianStateProcess <: TractableProcess end
abstract type DiscreteStateProcess <: TractableProcess end