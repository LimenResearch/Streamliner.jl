using Flux: @functor

_makesymbol(x) = x
_makesymbol(p::Pair) = (Symbol(p.first) => _makesymbol(p.second))
_makesymbol(D::Dict) = Dict(_makesymbol.([D...])...)

struct EnrichedModel
    metadata::Dict # training info
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
    model = model_to_constructor[info[:model][:type]](info[:model][:paths],
                                 info[:training])
    return EnrichedModel(info, optimizer, loss, model)
end

function get_optimizer(opt_data::Dict)
    opt = string_to_optim[opt_data[:name]](opt_data[:params][:lr])
end

function get_loss(loss_data::Dict, task_data::Dict)
    # !!! TODO find a way to relax TOML's keys to accept greek characters.
    # This would be useful for the nonlinearity (σ) and the loss' parameters.
    if task_data[:supervised]
        loss_fun(ŷ, y) = string_to_loss[loss_data[:loss]](ŷ, y; loss_data[:params]...)
    else
        loss_fun(ŷ) = string_to_loss[loss_data[:loss]](ŷ; loss_data[:params]...)
    end
end

function build_layers(layer_params::Vector, input_size::Union{Vector,Array,Tuple})
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

struct Sequential
    layers::Vector
end

function Sequential(paths::Vector, metadata::Dict)
    input_size = metadata[:data][:input_size]
    layer_params = _makesymbol(TOML.parsefile(paths[1]["layers"]))
    return Flux.Chain(build_layers(layer_params[:layers], input_size)...)
end

const model_to_constructor = Dict("sequential" => Sequential)
