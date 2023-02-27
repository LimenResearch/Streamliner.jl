function build_dense!(l_params::Dict, input_size::Union{Tuple,Vector})
    out = pop!(l_params, "out")
    σ = string_to_sigma[pop!(l_params,"sigma")]
    return Flux.Dense(input_size[1], out, σ; _makesymbol(l_params)...)
end

function build_convlike!(l_params::Dict, input_size::Union{Tuple,Vector}; layer)
    in_ch = last(input_size)
    out_ch = pop!(l_params, "out")
    filter = Tuple(Integer(k) for k in pop!(l_params, "filter"))
    σ = string_to_sigma[pop!(l_params,"sigma")]
    return layer(filter, in_ch => out_ch, σ; _makesymbol(l_params)...)
end

function build_conv!(l_params::Dict, input_size::Union{Tuple,Vector})
    build_convlike!(l_params::Dict, input_size::Union{Tuple,Vector}; layer=Flux.Conv)
end

function build_conv_t!(l_params::Dict, input_size::Union{Tuple,Vector})
    build_convlike!(l_params::Dict, input_size::Union{Tuple,Vector}; layer=Flux.ConvTranspose)
end

function build_rnn!(l_params::Dict, input_size::Union{Tuple,Vector})
    out = pop!(l_params, "out")
    σ = string_to_sigma[pop!(l_params,"sigma")]
    return Flux.RNN(input_size[1], out, σ; _makesymbol(l_params)...)
end

function build_lstmlike!(l_params::Dict, input_size::Union{Tuple,Vector}; layer)
    out = pop!(l_params, "out")
    return layer(input_size[1], out)
end

function build_lstm!(l_params::Dict, input_size::Union{Tuple,Vector})
    build_lstmlike!(l_params::Dict, input_size::Union{Tuple,Vector}; layer=Flux.LSTM)
end

function build_gru!(l_params::Dict, input_size::Union{Tuple,Vector})
    build_lstmlike!(l_params::Dict, input_size::Union{Tuple,Vector}; layer=Flux.GRU)
end

function get_output_size(layer, input_size::Union{Vector, Tuple})
    dummy = rand(Float32, tuple(input_size...,1)...)
    output_size = layer(dummy) |> size
    output_size = output_size[1:lastindex(output_size)-1] 
end