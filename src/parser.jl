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
    num_epochs::Integer
    batch_size::Integer
end

@functor EnrichedModel (model,)
(m::EnrichedModel)(x) = m.model(x)

function EnrichedModel(path::String)
    info = _makesymbol(TOML.parsefile(path))
    optimizer = get_optimizer(info[:training][:optimizer])
    loss = get_loss(info[:training][:loss])
    num_epochs = info[:training][:data][:num_epochs]
    batch_size = info[:training][:data][:batch_size]
    model = model_to_constructor[info[:model][:type]](info[:model][:paths], info[:training])
    return EnrichedModel(info, optimizer, loss, model, num_epochs, batch_size)
end

function get_optimizer(opt_data::Dict)
    # TODO: this needs refactoring to include all the optimizers introduced with Optimizers.jl
    opt = string_to_optim[opt_data[:name]](opt_data[:params]...)
end

"""
    translate(params::AbstractDict, translators::AbstractDict)

For every `key => value` pair in `params`, if `key` is present in replace `value`
with `translators[key][value]`.
"""
function translate(params::AbstractDict, translators::AbstractDict)
    d = Dict()
    for (k, v) in pairs(params)
        # FIXME: why do we get a mismatch `String` vs `Symbol` without this?
        translator = get(translators, string(k), nothing)
        d[k] = isnothing(translator) ? v : translator[v]
    end
    return d
end

# Simple callable structure that also holds parameters.
# The main purpose is to avoid storing closures in the model.
# This way is a bit easier for debugging.
struct ParametricFunction{F, NT<:NamedTuple}
    f::F
    params::NT
end

ParametricFunction(f; params...) = ParametricFunction(f, values(params))

(p::ParametricFunction)(args...) = p.f(args...; p.params...)

function get_loss(loss_data::Dict)
    # !!! TODO find a way to relax TOML's keys to accept greek characters.
    # This would be useful for the nonlinearity (σ) and the loss' parameters.
    params = translate(loss_data[:params], loss_params_translators)
    loss = string_to_loss[loss_data[:name]]
    return ParametricFunction(loss; params...)
end

function build_layers(layer_params::Vector, input_size::Union{Vector,Tuple};
                      prev_f = nothing, last_layer_info=false,
                      out_size::Union{Vector,Tuple,Nothing}=nothing)
    layers = []
    cur_size = input_size

    for l_params in layer_params
        f = pop!(l_params, "f")
        if prev_f !== nothing 
            reshaper = get(reshape_layers, reduce_layer([prev_f, f]), nothing)
            if !isnothing(reshaper)
                layer = reshaper
                push!(layers, layer)
                cur_size = get_output_size(layer, cur_size)
            end
        end
        prev_f = f
        layer = string_to_constructor[f](l_params, cur_size)
        push!(layers, layer)
        cur_size = get_output_size(layer, cur_size)
    end
    if !isnothing(out_size) && cur_size !== out_size
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
         prev_f=nothing, last_layer_info=false, out_size=nothing) =
    Flux.Chain(build_layers(params[:layers], input_size;
               last_layer_info=last_layer_info, prev_f = prev_f, out_size=out_size))

function to_chain(path_dict::Dict, key::Union{String, Symbol}, input_size::Union{Vector,Tuple};
                  prev_f=nothing, last_layer_info=false, out_size=nothing)
    params = parse_architecture(path_dict, key)
    println("In to chain with parsing. Parsed params are $params")
    return to_chain(params, input_size; last_layer_info=last_layer_info, prev_f = prev_f, out_size=out_size)
end
