function build_dense!(f::String, l_params::Dict, input_size::Union{Tuple,Vector})
    layer = string_to_layer[f]
    out = pop!(l_params, "out")
    σ = string_to_sigma[pop!(l_params,"sigma")]
    return layer(input_size[1], out, σ; _makesymbol(l_params)...)
end

function build_conv_like!(f::String, l_params::Dict, input_size::Union{Tuple,Vector})
    layer = string_to_layer[f]
    in_ch = last(input_size)
    out_ch = pop!(l_params, "out")
    filter = Tuple(Integer(k) for k in pop!(l_params, "filter"))
    σ = string_to_sigma[pop!(l_params,"sigma")]
    return layer(filter, in_ch => out_ch, σ; _makesymbol(l_params)...)
end

function build_rnn!(f::String, l_params::Dict, input_size::Union{Tuple,Vector})
    layer = string_to_layer[f]
    out = pop!(l_params, "out")
    σ = string_to_sigma[pop!(l_params,"sigma")]
    return layer(input_size[1], out, σ; _makesymbol(l_params)...)
end

function build_lstm_like!(f::String, l_params::Dict, input_size::Union{Tuple,Vector})
    layer = string_to_layer[f]
    out = pop!(l_params, "out")
    return layer(input_size[1], out)
end

function get_output_size(layer, input_size::Union{Vector, Tuple})
    dummy = rand(Float32, tuple(input_size...,1)...)
    output_size = layer(dummy) |> size
    output_size = output_size[1:lastindex(output_size)-1] 
end