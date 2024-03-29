using Streamliner: Architecture, get_model
using Flux

@testset "MLP test" begin
    mlp_arch = Architecture("../static/mlp.toml")
    @test size(mlp_arch.layers) == (2,)
    @test mlp_arch.layers[1] isa Dense && mlp_arch.layers[2] isa Dense
    t = Float32.(rand(50, 5))
    mlp = get_model(mlp_arch)
    @test size(mlp(t)) == (10, 5)
end
