using Flux

const layer_to_constructor = Dict("dense" => Flux.Dense, "conv" => Flux.Conv)

const reshape_layers = Dict(
    ("dense", "dense") => missing,
    ("conv", "conv") => missing,
    ("dense", "conv") => Flux.flatten,
)

