# When reading TOML files it seems dictionary keys are strings, but Symbols are 
# needed to initialize structures from dict splat. Here is what I found people do.
# TODO: Rewrite this in a better way (?)

_makesymbol(x) = x   # default 
_makesymbol(p::Pair) = (Symbol(p.first) => _makesymbol(p.second))
_makesymbol(D::Dict) = Dict(_makesymbol.([D...])...)

Base.@kwdef struct Architecture
    name::String
    input_size::Union{Vector, Array, Tuple}
    layers::Vector{Any} # TODO specify entry type (Flux?)
    num_epochs:: Union{Integer,Missing} = missing
    optimizer:: Union{String,Missing} = missing
    optimizer_params:: Union{Dict,Missing} = missing
    loss:: Union{String,Missing} = missing
    loss_params:: Union{Dict,Missing} = missing
    batch_size:: Union{Integer,Missing} = missing
    num_classes::Union{Integer,Missing} = missing
    is_supervised:: Union{Bool,Missing} = missing
end
function Architecture(path::String)
    d = TOML.parsefile(path)
    input_size = d["architecture"]["input_size"]
    layer_params = d["architecture"]["layers"]
    num_classes = get(d["architecture"], "num_classes", missing)
    if num_classes !== missing && last(layer_params)["out"] != num_classes
        @warn ("The output size of the last layer will be set to $num_classes 
                to match the number of classes provided in the configuration file.")
        last(layer_params)["out"] = num_classes
    end
    layers = build_layers(layer_params, input_size)
    d["architecture"]["layers"] = layers
    architecture = Architecture(; _makesymbol(d["architecture"])..., _makesymbol(d["training"])...)
end

function build_layers(layer_params::Vector, input_size::Union{Vector, Array, Tuple})
    layers = []
    prev_f = missing

    for l_params in layer_params
        f = l_params["f"]
        delete!(l_params, "f")
        if prev_f !== missing && reshape_layers[reduce_to_dense([prev_f, f])...] !== missing
            layer = reshape_layers[reduce_to_dense([prev_f, f])...]
            push!(layers, layer)
            input_size = get_output_size(layer, input_size)
        end
        prev_f = f
        layer = layer_to_constructor[f](l_params, input_size)
        push!(layers, layer)
        input_size = get_output_size(layer, input_size)
    end
    return layers
end

get_model(architecture::Architecture) = Flux.Chain(architecture.layers...)
function get_optimizer(architecture::Architecture)
    # !!! TODO parameters must be ordered: this could make it difficult to
    # generate cards automatially. Should we write kwargs-based constructors?
    opt = string_to_optim[architecture.optimizer](architecture.optimizer_params[:lr])
end

function get_loss(architecture::Architecture)
    # !!! TODO find a way to relax TOML's keys to accept greek characters.
    # This would be useful for the nonlinearity (σ) and the loss' parameters.
    model = get_model(architecture)
    if architecture.is_supervised
        loss_fun(x, y) = string_to_loss[architecture.loss](model(x), y)
    else
        loss_fun(x) = string_to_loss[architecture.loss](model(x))
    end
end
