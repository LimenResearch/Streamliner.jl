using Flux: gradient, params, update!
using Flux: @epochs


function train(architecture::Architecture, train_loader, test_x, test_y)
    loss = get_loss(architecture)
    opt = get_optimizer(architecture)
    m = get_model(architecture)
    evalcallback() = @show(loss(test_x, test_y))
    @epochs architecture.num_epochs Flux.train!(
        loss,
        Flux.params(m),
        train_loader,
        opt,
        cb = Flux.throttle(evalcallback, 5),
    )
    
    # epochs = architecture.num_epochs
    # for epoch = 1:epochs
    #     for d in train_loader
    #         gs = gradient(Flux.params(m)) do
    #         l = loss(d...)
    #         println(l)
    #         end
    #         update!(opt, Flux.params(m), gs)
    #     end
    # # @show accuracy(valX, valY)
    # end
end

