using Streamliner: Architecture, get_model
using Flux

@testset "Conv test" begin
    conv_arch = Architecture("../static/conv.toml")
    @test size(conv_arch.layers) == (2,)
    @test conv_arch.layers[1] isa Conv && conv_arch.layers[1] isa Conv
    t = rand([10, 10, 3, 10])
    conv = get_model(conv_arch)
    @test size(conv(t)) == (10, 10, 3, 10)
end

@testset "Conv classifier test" begin
    conv_class_arch = Architecture("../static/conv_classigier.toml")
    @test size(conv_class_arch.layers) == (3,)
    t = rand([10, 10, 3, 10])
    conv_class = get_model(conv_class_arch)
    @test size(conv_class(t)) == (10, 10)
end
