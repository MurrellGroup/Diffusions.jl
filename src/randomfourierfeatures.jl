struct RandomFourierFeatures{T <: Real, A <: AbstractVector{T}}
    w::A
end

@functor RandomFourierFeatures

"""
    RandomFourierFeatures(d::Integer, σ::Real)

Create a generator of `d`-dimensional random Fourier features with scale `σ`.

Tancik, Matthew, et al. "Fourier features let networks learn high frequency
functions in low dimensional domains." Advances in Neural Information Processing
Systems 33 (2020): 7537-7547.
"""
RandomFourierFeatures(d::Integer, σ::Real) = RandomFourierFeatures(d, float(σ))

function RandomFourierFeatures(d::Integer, σ::AbstractFloat)
    iseven(d) || throw(ArgumentError("dimension must be even"))
    isfinite(σ) && σ > 0 || throw(ArgumentError("scale must be finite and positive"))
    return RandomFourierFeatures(randn(typeof(σ), d ÷ 2) * σ)
end

function (rff::RandomFourierFeatures{T})(t::AbstractVector) where T
    wt = T(2π) .* rff.w .* t'
    return [cos.(wt); sin.(wt)]
end
