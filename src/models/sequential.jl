struct Sequential
    layers::Vector
end

function Sequential(paths::Dict, metadata::Dict)
    input_size = metadata[:data][:input_size]
    println(input_size)
    layer_params = parse_architecture(paths, :sequential)
    println(layer_params)
    return to_chain(layer_params, input_size)
end
