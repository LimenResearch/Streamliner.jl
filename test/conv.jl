using Streamliner: Architecture, get_model
using Flux

@testset "Conv test" begin
    conv_arch = Architecture("../static/conv.toml")
    @test size(conv_arch.layers) == (4,)
    @test conv_arch.layers[1] isa Conv && conv_arch.layers[2] isa Conv
    t = Float32.(rand(28, 28, 1, 10))
    conv = get_model(conv_arch)
    @test size(conv(t)) == (10, 10)
end

