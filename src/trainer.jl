function train(architecture::Architecture, data, test_data)
    objective = get_loss(architecture)
    opt = get_optimizer(architecture)
    if architecture.is_supervised
        evalcb() = @show(loss(test_data...))
    else
        evalcb() = @show(loss(test_data))
    end
    @epochs architecture.num_epochs Flux.train!(
        objective,
        data,
        opt,
        cb = throttle(evalcb, 5),
    )
end
