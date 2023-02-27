using Flux: gradient, params, update!, throttle, train!
using Flux: @epochs


function train(em::EnrichedModel, train_loader, test_x, test_y)
    loss, opt = em.loss, em.opt
    evalcallback() = @show(loss(test_x, test_y))
    @epochs em.num_epochs train!(
        loss,
        params(em),
        train_loader,
        opt,
        cb = throttle(evalcallback, 5),
    )
end

