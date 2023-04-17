using IterTools: ncycle

function train(em::EnrichedModel, train_loader, test_x, test_y; cb=nothing)
    loss, opt = em.loss, em.opt
    pp, re = Flux.destructure(em)
    optfun = OptimizationFunction(loss, Optimization.AutoZygote())
    optprob = OptimizationProblem(optfun, pp)
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