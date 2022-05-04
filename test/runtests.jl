using DlWrappers
using Test

@testset "DlWrappers.jl" begin
    include("./mlp.jl")
    include("./conv.jl")
end
