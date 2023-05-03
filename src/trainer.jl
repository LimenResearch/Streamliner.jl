using IterTools: ncycle

function train(em::EnrichedModel, train_loader; cb=default_callback)
    loss, opt = em.loss, em.optimizer
    # Turn model into a pair consisting of
    # - a flat vector of parameters `θ` and
    # - an anonymous function to reconstruct the model from the parameters.
    θ, reconstruct = Flux.destructure(em.model)
    # `OptimizationFunction` expects a function `f` that takes several arguments
    # The first argument, `θ`, represents the variable to be optimized.
    # The second argument, `p`, represents some hyperparameters (we do not need it).
    # The remaining arguments can be used to pass data.
    function f(θ, p, x, y)
        # reconstruct the model from the parameters
        m = reconstruct(θ)
        # compute the estimate of `y`
        ŷ = m(x)
        # compute the loss
        return loss(ŷ, y)
    end
    optfun = OptimizationFunction(f, Optimization.AutoZygote())
    optprob = OptimizationProblem(optfun, θ)
    res = solve(optprob, opt, ncycle(train_loader, em.num_epochs), callback = cb)
end


# callback = function (p, l, pred; doplot = false) #callback function to observe training
#     display(l)
#     # plot current prediction against data
#     if doplot
#         pl = scatter(t, ode_data[1, :], label = "data")
#         scatter!(pl, t, pred[1, :], label = "prediction")
#         display(plot(pl))
#     end
#     return false
# end