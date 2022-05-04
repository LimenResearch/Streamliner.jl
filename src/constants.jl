using Flux

const layer_to_constructor = Dict("dense" => Flux.Dense, "conv" => Flux.Conv)

const reshape_layers = Dict(
    ("dense", "dense") => missing,
    ("conv", "conv") => missing,
    ("conv", "dense") => Flux.flatten,
)

const string_to_sigma = Dict(
    "identity" => Flux.identity,
    "celu" => Flux.celu,
    "elu" => Flux.elu,
    "gelu" => Flux.gelu,
    "hardsigmoid" => Flux.hardsigmoid,
    "hardtanh" => Flux.hardtanh,
    "leakyrelu" => Flux.leakyrelu,
    "lisht" => Flux.lisht,
    "logcosh" => Flux.logcosh,
    "logsigmoid" => Flux.logsigmoid,
    "relu" => Flux.relu,
    "relu6" => Flux.relu6,
    "rrelu" => Flux.rrelu,
    "selu" => Flux.selu,
    "sigmoid" => Flux.sigmoid,
    "softplus" => Flux.softplus,
    "softshrink" => Flux.softshrink,
    "softsign" => Flux.softsign,
    "swish" => Flux.swish,
    "tanhshrink" => Flux.tanhshrink,
    "trelu" => Flux.trelu,
)

const string_to_sigma_array = Dict(
    "softmax" => Flux.softmax,
    "logsoftmax" => Flux.logsoftmax,
    "logsumexp" => Flux.logsumexp,
)