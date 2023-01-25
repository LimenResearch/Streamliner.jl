function consume!(::Type{T}, l_params::Dict, input_size::Union{Tuple,Vector}) where {T<:Dense}
    out = pop!(l_params, "out")
    σ = string_to_sigma[pop!(l_params,"sigma")]
    return T(input_size[1], out, σ; _makesymbol(l_params)...)
end

function consume!(::Type{T}, l_params::Dict, input_size::Union{Tuple,Vector}) where {T<:Union{Conv,ConvTranspose}}
    in_ch = last(input_size)
    out_ch = pop!(l_params, "out")
    filter = Tuple(Integer(k) for k in pop!(l_params, "filter"))
    σ = string_to_sigma[pop!(l_params,"sigma")]
    return T(filter, in_ch => out_ch, σ; _makesymbol(l_params)...)
end

function consume!(::Type{T}, l_params::Dict, input_size::Union{Tuple,Vector}) where {T<:RNN}
    out = pop!(l_params, "out")
    σ = string_to_sigma[pop!(l_params,"sigma")]
    return T(input_size[1], out, σ; _makesymbol(l_params)...)
end

function consume!(::Type{T}, l_params::Dict, input_size::Union{Tuple,Vector}) where {T<:Union{LSTM,GRU}}
    out = pop!(l_params, "out")
    return T(input_size[1], out)
end

# function Flux.Dense(l_params::Dict, input_size::Union{Tuple,Vector})
#     out = l_params["out"]
#     delete!(l_params, "out")
#     σ = string_to_sigma[l_params["sigma"]]
#     delete!(l_params, "sigma")
#     layer = Dense(input_size[1], out, σ; _makesymbol(l_params)...)
# end

# function Flux.Conv(l_params::Dict, input_size::Union{Tuple,Vector})
#     """l_params can heave keys:
#     filter, out, σ = identity, stride = 1, pad = 0, dilation = 1, groups = 1,
#     bias=true, init=glorot_uniform
#     """
#     in_ch = last(input_size)
#     out_ch = l_params["out"]
#     delete!(l_params, "out")
#     filter = Tuple(Integer(k) for k in l_params["filter"])
#     delete!(l_params, "filter")
#     σ = string_to_sigma[l_params["sigma"]]
#     delete!(l_params, "sigma")
#     layer = Conv(filter, in_ch => out_ch, σ; _makesymbol(l_params)...)
# end

# function Flux.ConvTranspose(l_params::Dict, input_size::Union{Tuple,Vector})
#     """l_params can heave keys:
#     filter, out, σ = identity, stride = 1, pad = 0, dilation = 1, groups = 1,
#     bias=true, init=glorot_uniform
#     """
#     in_ch = last(input_size)
#     out_ch = l_params["out"]
#     delete!(l_params, "out")
#     filter = Tuple(Integer(k) for k in l_params["filter"])
#     delete!(l_params, "filter")
#     σ = string_to_sigma[l_params["sigma"]]
#     delete!(l_params, "sigma")
#     layer = ConvTranspose(filter, in_ch => out_ch, σ; _makesymbol(l_params)...)
# end

# function Flux.RNN(l_params::Dict, input_size::Union{Tuple,Vector})
#     # TODO: This constructor behaves exactly as dense. We could implement a DenseLike
#     # constructor.
#     out = l_params["out"]
#     delete!(l_params, "out")
#     σ = string_to_sigma[l_params["sigma"]]
#     delete!(l_params, "sigma")
#     layer = RNN(input_size[1], out, σ; _makesymbol(l_params)...)
# end

# function Flux.LSTM(l_params::Dict, input_size::Union{Tuple,Vector})
#     layer = LSTM(input_size[1], l_params["out"])
# end

# function Flux.GRU(l_params::Dict, input_size::Union{Tuple,Vector})
#     layer = GRU(input_size[1], l_params["out"])
# end

function get_output_size(layer, input_size::Union{Vector, Tuple})
    dummy = rand(Float32, tuple(input_size...,1)...)
    output_size = layer(dummy) |> size
    output_size = output_size[1:lastindex(output_size)-1] 
end