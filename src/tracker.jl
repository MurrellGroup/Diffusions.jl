struct NullTracker end

track!(::NullTracker, t, x) = nothing


struct Tracker
    time::Vector
    data::Vector
end

Tracker() = Tracker([], [])

function track!(tracker::Tracker, t, x)
    push!(tracker.time, t)
    push!(tracker.data, x)
    return nothing
end
