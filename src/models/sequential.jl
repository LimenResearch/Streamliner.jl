struct Sequential
    layers::Vector
end

function Sequential(paths::Dict, metadata::Dict)
    input_size = metadata[:data][:input_size]
    layer_params = parse_architecture(paths, "sequential")
    return to_chain(layer_params, input_size)
end
