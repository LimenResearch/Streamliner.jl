using DlWrappers
using MLDatasets: MNIST
using Flux.Data: DataLoader
using Flux: onehotbatch

function get_mnist()
    train_x, train_y = MNIST.traindata(Float32)
    test_x, test_y = MNIST.testdata(Float32)
    train_x = reshape(train_x, 28, 28, 1, :)
    test_x = reshape(test_x, 28, 28, 1, :)
    train_y, test_y = onehotbatch(train_y, 0:9), onehotbatch(test_y, 0:9)
    return train_x, train_y, test_x, test_y
end

architecture = DlWrappers.Architecture(joinpath(@__DIR__, "static/conv_classifier.toml"));
train_x, train_y, test_x, test_y = get_mnist()
train_loader = DataLoader((train_x, train_y); batchsize=architecture.batch_size, shuffle=true)
DlWrappers.train(architecture, train_loader, test_x, test_y)
println("done")