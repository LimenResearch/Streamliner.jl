struct Sequential
    layers::Vector
end

function Sequential(paths::Dict, metadata::Dict)
    input_size = metadata[:data][:input_size]
    return to_chain(paths, :sequential, input_size)
end
