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
    loss = get_loss(info[:training][:loss], info[:training][:task])
    model = model_to_constructor[info[:model][:type]](info[:model][:paths], info[:training])
    return EnrichedModel(info, optimizer, loss, model)
end

function get_optimizer(opt_data::Dict)
    opt = string_to_optim[opt_data[:name]](opt_data[:params][:lr])
end

function get_loss(loss_data::Dict, task_data::Dict)
    # !!! TODO find a way to relax TOML's keys to accept greek characters.
    # This would be useful for the nonlinearity (σ) and the loss' parameters.
    if task_data[:supervised]
        loss_fun(ŷ, y) = string_to_loss[loss_data[:name]](ŷ, y; loss_data[:params]...)
    else
        if occursin("vae", loss_data[:name])
           loss_fun(x, x̂, μ, logvar) = string_to_loss[loss_data[:name]](x, x̂, μ, logvar; loss_data[:params])
        else
            loss_fun(ŷ) = string_to_loss[loss_data[:name]](ŷ; loss_data[:params]...)
        end
    end
end

function build_layers(layer_params::Vector, input_size::Union{Vector,Tuple};
                      prev_f = missing, last_layer_info=false,
                      out_size::Union{Vector,Tuple}=missing)
    layers = []
    cur_size = input_size

    for l_params in layer_params
        f = l_params["f"]
        delete!(l_params, "f")
        if prev_f !== missing 
            if (reshape_layers[reduce_to_dense([prev_f, f])...] !== missing)
                layer = reshape_layers[reduce_to_dense([prev_f, f])...]
            else (reshape_layers[reduce_to_conv([prev_f, f])...] !== missing)
                layer = reshape_layers[reduce_to_conv([prev_f, f])...]
            end
            push!(layers, layer)
            cur_size = get_output_size(layer, cur_size)
        end
        prev_f = f
        layer = layer_to_constructor[f](l_params, cur_size)
        push!(layers, layer)
        cur_size = get_output_size(layer, cur_size)
    end
    if out_size !== missing && cur_size !== out_size
        @warn "The last layer size ($cur_size) does not match the provided out_size $out_size. A resampling layer shall be added"
        push(layers, Flux.Upsample(:nearest, size=out_size[1:lastindex(out_size)-1]))
    end
    if last_layer_info
        return layers, cur_size, f
    end
    return layers
end

parse_architecture(path_dict::Dict, key::String) = 
    _makesymbol(TOML.parsefile(path_dict[key]))

to_chain(params::Dict, input_size::Union{Vector,Tuple};
         prev_f = missing, last_layer_info=false) =
    Flux.Chain(build_layers(params[:layers], input_size;
               last_layer_info=last_layer_info, prev_f = prev_f)...)

function to_chain(path_dict::Dict, key::String, input_size::Union{Vector,Tuple};
                  prev_f = missing, last_layer_info=false)
    params = parse_architecture(path_dict, key)
    return to_chain(params, input_size; last_layer_info=last_layer_info, prev_f = prev_f)
end
