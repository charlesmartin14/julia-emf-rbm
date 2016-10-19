using Boltzmann
using MNIST
using Base.Test

function run_mnist()
    X, y = testdata()  # test data is smaller, no need to downsample
    binarize!(X;threhold=0.01)

    TrainSet = X[:,1:7000]
    ValidSet = X[:,7001:end]

    HiddenUnits = 100;
    Epochs = 10;
    LearningRate = 0.1;

    m_valid = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.5, dataset=TrainSet)
    m = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.5, dataset=TrainSet)
    
    # Fit using Validation Set
    fit(m_valid, TrainSet;n_iter=Epochs,lr=LearningRate,validation=ValidSet)

    # Fit without Validation Set
    fit(m, TrainSet;n_iter=Epochs,lr=LearningRate)

    return m
end

run_mnist()



