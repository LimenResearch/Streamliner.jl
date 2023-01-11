using Streamliner
using Test

@testset "Streamliner.jl" begin
    include("./mlp.jl")
    include("./conv.jl")
end
