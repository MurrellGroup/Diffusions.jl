struct RandomFourierFeatures{T <: Real, A <: AbstractVector{T}}
    w::A
end

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

(rff::RandomFourierFeatures{T})(t::Union{Real, AbstractVector{<: Real}}) where T = rff(convert.(T, t))

function (rff::RandomFourierFeatures{T})(t::Union{T, AbstractVector{T}}) where T <: Real
    wt = T(2π) .* rff.w .* t'
    return [cos.(wt); sin.(wt)]
end
