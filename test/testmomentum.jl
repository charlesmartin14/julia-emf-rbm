using Colors
using Images
using Boltzmann
using MNIST
using ImageView
using Base.Test

function run_mnist()
    X, y = testdata()  # test data is smaller, no need to downsample
    binarize!(X;threhold=0.01)

    HiddenUnits = 256;
    Epochs = 10;
    LearningRate = 0.1;

    m = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.0, dataset=X)
    mM = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.5, dataset=X)
    
    # Attempt without Momentum
    info("Running Without Momentum")
    fit(m, X;n_iter=Epochs,lr=LearningRate)

    # Attempt with Momentum
    info("Running With Momentum")
    fit(mM, X;n_iter=Epochs,lr=LearningRate)

    return m
end

run_mnist()