using Colors
using Images
using Boltzmann
using MNIST
using ImageView
using Base.Test

function run_mnist()
    X, y = testdata()  # test data is smaller, no need to downsample
    normalize_samples!(X)
    binarize!(X;threshold=0.01)

    TrainSet = X[:,1:9000]
    ValidSet = X[:,9001:end]
    HiddenUnits = 100;
    Epochs = 10;
    Gibbs = 1;
    LearnRate = 0.05
    MonitorEvery=1

    rbm = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.0)

    # Attempt with L2 weight decay
    finalrbm,monitor = fit(rbm, TrainSet;n_iter=Epochs,
                          weight_decay="l2",
                          decay_magnitude=0.001,
                          lr=LearnRate,
                          validation=ValidSet,
                          NormalizationApproxIter=Gibbs,
                          monitor_every=MonitorEvery,
                          monitor_vis=true)
    
    info("==========monitor==============")
    println(monitor)
    info("===============================")

    WriteMonitorChartPDF(finalrbm,monitor,X,"testmonitor.pdf")
    SaveMonitorHDF5(monitor,"testmonitor.h5")
end

run_mnist()