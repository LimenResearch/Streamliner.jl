using DlWrappers: Architecture, get_model
using Flux

@testset "MLP test" begin
    mlp_arch = Architecture("../static/dense.toml")
    @test size(mlp_arch.layers) == (2,)
    @test mlp_arch.layers[1] isa Dense && mlp_arch.layers[1] isa Dense
    t = rand(10, 5)
    mlp = get_model(mlp_arch)
    @test size(mlp(t)) == (10, 5)
end
