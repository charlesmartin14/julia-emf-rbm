using Colors
using Images
using Boltzmann
using MNIST
using ImageView
using Base.Test

function run_mnist()
    X, y = testdata()  # test data is smaller, no need to downsample
    normalize_samples!(X)
    binarize!(X;threhold=0.01)

    TrainSet = X[:,1:10000]
    ValidSet = []
    HiddenUnits = 256;
    Epochs = 15;
    Gibbs = 1;
    LearnRate = 0.05

    m = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.0)
    mwdQuad = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.0)
    mwdLin = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.0)
    
    # Attempt without weight decay
    info("Running Without Weight Decay")
    m, mon = fit(m, TrainSet;n_iter=Epochs,lr=LearnRate,validation=ValidSet,n_gibbs=Gibbs)
    SaveMonitor(m,mon,"test_nodecay.pdf")

    # Attempt with L2 weight decay
    info("Running With L2-Decay")
    mwQuad, mon = fit(mwdQuad, TrainSet;n_iter=Epochs,weight_decay="l2",decay_magnitude=0.05,lr=LearnRate,validation=ValidSet,n_gibbs=Gibbs)
    SaveMonitor(mwQuad,mon,"test_l2decay.pdf")

    # Attempt with L1 weight decay
    info("Running With L1-Decay")
    mwLin, mon = fit(mwdLin, TrainSet;n_iter=Epochs,weight_decay="l1",decay_magnitude=0.05,lr=LearnRate,validation=ValidSet,n_gibbs=Gibbs)
    SaveMonitor(mwLin,mon,"test_l1decay.pdf")

    return m
end

run_mnist()


