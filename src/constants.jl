using Flux

name_to_layer = Dict("dense" => Dense, "conv" => Conv)
reshape_layers = Dict(
    ("dense", "dense") => missing,
    ("conv", "conv") => missing,
    ("dense", "conv") => Flux.flatten,
)

