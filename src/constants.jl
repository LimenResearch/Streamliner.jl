const model_to_constructor = Dict(
    "sequential" => Sequential,
    "vae"=> VAE)

const string_to_constructor = Dict(
    "dense" => build_dense!,
    "conv" => build_conv!,
    "conv_t" => build_conv_t!,
    "rnn" => build_rnn!,
    "lstm" => build_lstm!,
    "gru" => build_gru!,
    "dpm" => build_densemachine!,
    "cpm" => build_convmachine!,
    "tpm" => build_timemachine!
)

const dense_like = ["dense", "rnn", "lstm", "gru", "dpm", "rpm"]
const conv_like = ["conv", "conv_t", "cpm"]

function reduce_layer(name::String)
    if name in dense_like
        return "dense"
    elseif name in conv_like
        return "conv"
    end
    return name
end

reduce_layer(list::Vector) = Tuple(reduce_layer.(list))

const reshape_layers = Dict(
    ("dense", "dense") => nothing,
    ("conv", "conv") => nothing,
    ("conv_t", "conv_t") => nothing,
    ("conv", "dense") => Flux.flatten,
    ("dense", "conv") => dense_to_conv,
    ("conv_t", "dense") => Flux.flatten,
    ("dense", "conv_t") => dense_to_conv,
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

const string_to_loss = Dict(
    "mae" => Flux.Losses.mae,
    "mse" => Flux.Losses.mse,
    "msle" => Flux.Losses.msle,
    "huber_loss" => Flux.Losses.huber_loss,
    "label_smoothing" => Flux.Losses.label_smoothing,
    "crossentropy" => Flux.Losses.crossentropy,
    "logitcrossentropy" => Flux.Losses.logitcrossentropy,
    "binarycrossentropy" => Flux.Losses.binarycrossentropy,
    "logitbinarycrossentropy" => Flux.Losses.logitbinarycrossentropy,
    "kldivergence" => Flux.Losses.kldivergence,
    "poisson_loss" => Flux.Losses.poisson_loss,
    "hinge_loss" => Flux.Losses.hinge_loss,
    "squared_hinge_loss" => Flux.Losses.squared_hinge_loss,
    "dice_coeff_loss" => Flux.Losses.dice_coeff_loss,
    "tversky_loss" => Flux.Losses.tversky_loss,
    "binary_focal_loss" => Flux.Losses.binary_focal_loss,
    "focal_loss" => Flux.Losses.focal_loss,
    "siamese_contrastive_loss" => Flux.Losses.siamese_contrastive_loss,
    "vae_loss" => vae_loss
)

const string_to_optim = Dict(
    "Descent" => Flux.Optimise.Descent,
    "Momentum" => Flux.Optimise.Momentum,
    "Nesterov" => Flux.Optimise.Nesterov,
    "RMSProp" => Flux.Optimise.RMSProp,
    "ADAM" => Flux.Optimise.ADAM,
    "RADAM" => Flux.Optimise.RADAM,
    "AdaMax" => Flux.Optimise.AdaMax,
    "ADAGrad" => Flux.Optimise.ADAGrad,
    "ADADelta" => Flux.Optimise.ADADelta,
    "AMSGrad" => Flux.Optimise.AMSGrad,
    "NADAM" => Flux.Optimise.NADAM,
    "ADAMW" => Flux.Optimise.ADAMW,
    "OADAM" => Flux.Optimise.OADAM,
    "AdaBelief" => Flux.Optimise.AdaBelief,
    "BFGS" => Optim.BFGS,
    "LBFGS" => Optim.LBFGS
)