using Flux: @functor

_makesymbol(x) = x
_makesymbol(p::Pair) = (Symbol(p.first) => _makesymbol(p.second))
_makesymbol(D::Dict) = Dict(_makesymbol.([D...])...)

abstract type AbstractEnrichedModel end

struct EnrichedModel <: AbstractEnrichedModel
    metadata::Dict # info
    optimizer::Any
    loss::Any
    model::Any
end

@functor EnrichedModel (model,)
(m::EnrichedModel)(x) = m.model(x)

function EnrichedModel(path::String)
    info = _makesymbol(TOML.parsefile(path))
    optimizer = get_optimizer(info[:training][:optimizer])
    loss = get_loss(info[:training][:loss])
    model = model_to_constructor[info[:model][:type]](info[:model][:paths], info[:training])
    return EnrichedModel(info, optimizer, loss, model)
end

function get_optimizer(opt_data::Dict)
    opt = string_to_optim[opt_data[:name]](opt_data[:params][:lr])
end

function get_loss(loss_data::Dict)
    # !!! TODO find a way to relax TOML's keys to accept greek characters.
    # This would be useful for the nonlinearity (σ) and the loss' parameters.
    loss_fun(args...) = string_to_loss[loss_data[:name]](args...; loss_data[:params]...)
    return loss_fun # for clarity, not necessary
end

function build_layers(layer_params::Vector, input_size::Union{Vector,Tuple};
                      prev_f = missing, last_layer_info=false,
                      out_size::Union{Vector,Tuple,Missing}=missing)
    layers = []
    cur_size = input_size

    for l_params in layer_params
        f = pop!(l_params, "f")
        if prev_f !== missing 
            # TODO rewrite the following if elseif horror
            if (reshape_layers[reduce_to_dense([prev_f, f])...] !== missing)
                layer = reshape_layers[reduce_to_dense([prev_f, f])...]
                push!(layers, layer)
                cur_size = get_output_size(layer, cur_size)
            elseif (reshape_layers[reduce_to_conv([prev_f, f])...] !== missing)
                layer = reshape_layers[reduce_to_conv([prev_f, f])...]
                push!(layers, layer)
                cur_size = get_output_size(layer, cur_size)
            end
        end
        prev_f = f
        layer = string_to_constructor[f](l_params, cur_size)
        push!(layers, layer)
        cur_size = get_output_size(layer, cur_size)
    end
    if out_size !== missing && cur_size !== out_size
        @warn "The last layer size ($cur_size) does not match the provided out_size $out_size. A resampling layer shall be added"
        push!(layers, Flux.Upsample(:nearest, size=out_size[1:lastindex(out_size)-1]))
    end
    if last_layer_info
        return layers, cur_size, prev_f
    end
    return layers
end

parse_architecture(path_dict::Dict, key::Union{String,Symbol}) = 
    _makesymbol(TOML.parsefile(path_dict[key]))

to_chain(params::Dict, input_size::Union{Vector,Tuple};
         prev_f = missing, last_layer_info=false, out_size=missing) =
    Flux.Chain(build_layers(params[:layers], input_size;
               last_layer_info=last_layer_info, prev_f = prev_f, out_size=out_size))

function to_chain(path_dict::Dict, key::Union{String, Symbol}, input_size::Union{Vector,Tuple};
                  prev_f = missing, last_layer_info=false, out_size=missing)
    params = parse_architecture(path_dict, key)
    println("In to chain with parsing. Parsed params are $params")
    return to_chain(params, input_size; last_layer_info=last_layer_info, prev_f = prev_f, out_size=out_size)
end
