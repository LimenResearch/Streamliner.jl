using Flux
include("./resize_helper.jl")

const layer_to_constructor = Dict(
    "dense" => Flux.Dense,
    "conv" => Flux.Conv,
    "rnn" => Flux.RNN,
    "lstm" => Flux.LSTM,
    "gru" => Flux.GRU,
    )

const dense_like = ["dense", "rnn", "lstm", "gru"]

reduce_to_dense(name::String) = ifelse(name in dense_like, "dense", name)
reduce_to_dense(list::Vector) = reduce_to_dense.(list)

const reshape_layers = Dict(
    ("dense", "dense") => missing,
    ("conv", "conv") => missing,
    ("conv", "dense") => Flux.flatten,
    ("dense", "conv") => dense_to_conv
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
    "tanh" => Flux.tanh,
    "tanhshrink" => Flux.tanhshrink,
    "trelu" => Flux.trelu,
)

const string_to_sigma_array = Dict(
    "softmax" => Flux.softmax,
    "logsoftmax" => Flux.logsoftmax,
    "logsumexp" => Flux.logsumexp,
)