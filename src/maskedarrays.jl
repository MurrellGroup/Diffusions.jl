struct MaskedArray{T, N, A <: AbstractArray, I <: AbstractVector{<: Integer}} <: AbstractArray{T, N}
    data::A
    indices::I

    function MaskedArray(data::AbstractArray{T, N}, indices::AbstractVector{<: Integer}) where {T, N}
        # TODO: check integrity of indices (sorted, not duplicated, and more?)
        return new{T, N, typeof(data), typeof(indices)}(data, indices)
    end
end

Base.size(A::MaskedArray) = size(A.data)
Base.copy(A::MaskedArray) = MaskedArray(copy(A.data), copy(A.indices))
Base.getindex(A::MaskedArray, i...) = A.data[i...]
Base.parent(A::MaskedArray) = A.data

function Base.setindex!(A::MaskedArray, val, i...)
    A.data[i...] = val
    return A
end

Adapt.adapt_structure(to, A::MaskedArray) = MaskedArray(Adapt.adapt(to, A.data), Adapt.adapt(to, A.indices))

"""
    namsked(A)

Return the number of masked elements.
"""
nmasked(A::MaskedArray) = length(A.indices)
nmasked(A::AbstractArray) = length(A)

"""
    maskedvec(A)

Return a view of masked elements as a vector.
"""
maskedvec(A::MaskedArray) = view(A.data, A.indices)
maskedvec(A::AbstractArray) = view(A, :)

"""
    mask(data, mask)

Create a masked array.

`data` and `mask` must have the same size.
"""
mask(data::AbstractArray, mask::AbstractArray{Bool}) = MaskedArray(data, findall(vec(mask)))

# this is to avoid nesting masked arrays
mask(masked::MaskedArray, mask::AbstractArray{Bool}) = MaskedArray(masked.data, findall(vec(mask)))
