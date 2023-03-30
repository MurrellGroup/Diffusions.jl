struct MaskedArray{T, A <: AbstractArray, N} <: AbstractArray{T, N}
    data::A
    mask::BitArray{N}

    function MaskedArray(data::AbstractArray{T, N}, mask::BitArray{N}) where {T, N}
        size(data) == size(mask) || throw(ArgumentError("data and mask must have the same size"))
        return new{T, typeof(data), N}(data, mask)
    end
end

Base.size(A::MaskedArray) = size(A.data)
Base.copy(A::MaskedArray) = MaskedArray(copy(A.data), copy(A.mask))
Base.getindex(A::MaskedArray, i...) = A.data[i...]

function Base.setindex!(A::MaskedArray, val, i...)
    A.data[i...] = val
    return A
end

maskedvec(A::MaskedArray) = A.data[A.mask]

function updatemasked!(A::MaskedArray, vals::AbstractVector)
    A.data[A.mask] .= vals
    return A
end

"""
    mask(data, mask)

Create a masked array.

`data` and `mask` must have the same size.
"""
mask(data::AbstractArray, mask::AbstractArray{Bool}) = MaskedArray(data, convert(BitArray, mask)) 

# avoid nesting masked arrays
mask(masked::MaskedArray, mask::AbstractArray{Bool}) = MaskedArray(masked.data, mask)
