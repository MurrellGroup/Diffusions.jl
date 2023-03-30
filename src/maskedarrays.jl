struct MaskedArray{T, A <: AbstractArray, N} <: AbstractArray{T, N}
    data::A
    indices::Vector{Int}

    function MaskedArray(data::AbstractArray{T, N}, indices::Vector{Int}) where {T, N}
        # TODO: check integrity of indices (sorted, not duplicated, and more?)
        return new{T, typeof(data), N}(data, indices)
    end
end

Base.size(A::MaskedArray) = size(A.data)
Base.copy(A::MaskedArray) = MaskedArray(copy(A.data), copy(A.indices))
Base.getindex(A::MaskedArray, i...) = A.data[i...]

function Base.setindex!(A::MaskedArray, val, i...)
    A.data[i...] = val
    return A
end

"""
    namsked(A)

Return the number of masked elements.
"""
nmasked(A::MaskedArray) = length(A.indices)

"""
    maskedvec(A)

Return a view of masked elements as a vector.
"""
maskedvec(A::MaskedArray) = view(A.data, A.indices)

"""
    mask(data, mask)

Create a masked array.

`data` and `mask` must have the same size.
"""
mask(data::AbstractArray, mask::AbstractArray{Bool}) = MaskedArray(data, findall(vec(mask)))

# this is to avoid nesting masked arrays
mask(masked::MaskedArray, mask::AbstractArray{Bool}) = MaskedArray(masked.data, findall(vec(mask)))