using TOML
using Flux
using IterTools

include("./constants.jl")
include("./chainer.jl")
"""
When reading TOML files it seems dictionary keys are strings, but Symbols are 
needed to initialize structures from dict splat. Here is what I found people do.

TODO: Rewrite this in a better way (?)
"""
_makesymbol(x) = x   # default 
_makesymbol(p::Pair) = (Symbol(p.first) => _makesymbol(p.second))
_makesymbol(D::Dict) = Dict(_makesymbol.([D...])...)

function Flux.Dense(l_params::Dict, input_size::Union{Tuple,Vector})
    out = l_params["out"]
    delete!(l_params, "out")
    layer = Dense(input_size[1], out; _makesymbol(l_params)...)
end

get_output_size(layer::Dense) = [size(layer.weight)[1]]
get_output_size(layer::Flux.flatten, input_size=Union{Vector, Tuple}) = prod(input_size) 

const layer_to_constructor = Dict("dense" => Flux.Dense, "conv" => Flux.Conv)

Base.@kwdef struct Architecture
    name::String
    input_size::Union{Vector, Array, Tuple}
    layers::Vector{Any} # TODO specify entry type (Flux?)
end

function Architecture(path::String)
    d = TOML.parsefile(path)
    input_size = d["architecture"]["input_size"]

    layers = []
    prev_f = missing
    for layer in d["architecture"]["layers"]
        f = layer["f"]
        constructor = layer_to_constructor[f]
        delete!(layer, "f")
        layer = constructor(layer, input_size)
        push!(layers, layer)
        input_size = get_output_size(layer)
        if reshape_layers[(prev_f, f)] !== missing
            layer = reshape_layers[(prev_f, f)]
            push!(layers, layer)
            input_size = get_output_size(layer, input_size)
        end
        prev_f = f
    end

    d["architecture"]["layers"] = layers
    architecture = Architecture(; _makesymbol(d["architecture"])...)
end

get_model(architecture::Architecture) = Flux.Chain(architecture.layers...)