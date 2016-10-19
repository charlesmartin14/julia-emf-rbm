using Boltzmann
using MNIST
using Base.Test
using HDF5

function run_mnist()
    X, y = traindata()  # test data is smaller, no need to downsample
    normalize_samples!(X)
    binarize!(X;threshold=0.01)

    TrainSet = X
    ValidSet = []
    HiddenUnits = 500;
    Epochs = 10;
    MCMCIter = 1;
    EMFIter = 3
    LearnRate = 0.005
    MonitorEvery=2
    EMFPersistStart=5

    rbm1 = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.5, TrainData=TrainSet, sigma = 0.01)
    rbm2 = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.5, TrainData=TrainSet, sigma = 0.01)
    rbm3 = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.5, TrainData=TrainSet, sigma = 0.01)


    finalrbmtap2,monitor = fit(rbm1, TrainSet;n_iter=Epochs,
                          weight_decay="l2",
                          decay_magnitude=0.001,
                          lr=LearnRate,
                          persistent=true,
                          validation=ValidSet,
                          NormalizationApproxIter=EMFIter,
                          monitor_every=MonitorEvery,
                          monitor_vis=true,
                          approx="tap2",
                          persistent_start=EMFPersistStart)

    WriteMonitorChartPDF(finalrbmtap2,monitor,X,"testmonitor_tap2.pdf")
    SaveMonitorHDF5(monitor,"testmonitor_tap2.h5")

    finalrbmnaive,monitor = fit(rbm2, TrainSet;n_iter=Epochs,
                          weight_decay="l2",
                          decay_magnitude=0.001,
                          lr=LearnRate,
                          persistent=true,
                          validation=ValidSet,
                          NormalizationApproxIter=EMFIter,
                          monitor_every=MonitorEvery,
                          monitor_vis=true,
                          approx="naive",
                          persistent_start=EMFPersistStart)

    WriteMonitorChartPDF(finalrbmnaive,monitor,X,"testmonitor_naive.pdf")
    SaveMonitorHDF5(monitor,"testmonitor_naive.h5")

    finalrbmCD,monitor = fit(rbm3, TrainSet;n_iter=Epochs,
                          weight_decay="l2",
                          decay_magnitude=0.001,
                          lr=LearnRate,
                          persistent=true,                          
                          validation=ValidSet,
                          NormalizationApproxIter=MCMCIter,
                          monitor_every=MonitorEvery,
                          monitor_vis=true,
                          approx="CD")

    WriteMonitorChartPDF(finalrbmCD,monitor,X,"testmonitor_CD.pdf")
    SaveMonitorHDF5(monitor,"testmonitor_CD.h5")
end

run_mnist()