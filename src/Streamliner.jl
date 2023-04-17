module Streamliner

using TOML
using Flux
using IterTools
using Primes
using Optimization, OptimizationOptimisers
using ParametricMachinesDemos

# include("model_config.jl")
include("models/sequential.jl")
include("models/vae.jl")
include("parser.jl")
include("layer_constructors.jl")
include("resize_helper.jl")
include("constants.jl")
# include("trainer.jl")
# include("model_warehouse.jl")


end
