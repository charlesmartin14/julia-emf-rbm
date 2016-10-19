
using Boltzmann
using Distributions
using Base.Test

X = rand(1000, 200)


function brbm_smoke_test()
    model = BernoulliRBM(1000, 500, (1000,1))
    fit(model, X)
end

function grbm_smoke_test()
    model = GRBM(1000, 500, (1000,1))
    fit(model, X)
end

function conf_smoke_test()
    model = RBM(Normal, Normal, 1000, 500,(1000,1))
    fit(model, X)
end

function init_smoke_test()
    model = RBM(Normal, Normal, 1000, 500, (1000,1);dataset=X)
    fit(model, X)
end

info("brbm_smoke_test()")
brbm_smoke_test()
info("grbm_smoke_test()")
grbm_smoke_test()
info("conf_smoke_test()")
conf_smoke_test()
info("init_smoke_test()")
init_smoke_test()
