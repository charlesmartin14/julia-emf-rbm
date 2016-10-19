
Boltzmann.jl
============

[![Build Status](https://travis-ci.org/dfdx/Boltzmann.jl.svg)](https://travis-ci.org/dfdx/Boltzmann.jl)

Restricted Boltzmann machines and deep belief networks in Julia.
This particular package is a fork of [dfdx/Boltzmann.jl](https://github.com/dfdx/Boltzmann.jl) 
with modifications made by the SPHINX Team @ ENS Paris.


Installation
------------
Currently, this package is unregistered with the Julia package manager. 
Once the modifications here are feature complete, we can either make the fork permanent or request a merge back into the [dfdx/Boltzmann.jl](https://github.com/dfdx/Boltzmann.jl) package. For now, installation should be accomplished via:

```julia
    Pkg.clone("https://github.com/sphinxteam/Boltzmann.jl")
```

RBM Basic Usage
---------------

Below, we show a basic script to train a binary RBM on random training data. For this example, persistent contrastive divergence with one step on the MCMC sampling chain (PCD-1) is used. Finally, we also point out the monitoring and charting functionality which is passed as an optional argument to the `fit` procedure.

```julia
    using Boltzmann

    # Experimental parameters for this smoke test
    NFeatures    = 100
    FeatureShape = (10,10)
    NSamples     = 2000
    NHidden      = 50

    # Generate a random test set in [0,1]
    X = rand(NFeatures, NSamples)    
    binarize!(X;threshold=0.5)                        

    # Initialize the RBM Model
    rbm = BernoulliRBM(NFeatures, NHidden, FeatureShape)

    # Run CD-1 Training with persistence
    rbm = fit(rbm,X; n_iter        = 30,      # Training Epochs
                     batch_size    = 50,      # Samples per minibatch
                     persistent    = true,    # Use persistent chains
                     approx        = "CD",    # Use CD (MCMC) Sampling
                     monitor_every = 1,       # Epochs between scoring
                     monitor_vis   = true)    # Show live charts
```

Extended Mean-Field (EMF) Approximation
---------------------------------------

Besides the use of the sampling-based default CD RBM training, we have also implemented the extended mean-field approach of

> M. Gabrié, E. W. Tramel, F. Krzakala, ``Training restricted Boltzmann machines via the Thouless-Andreson-Palmer free energy,'' in Proc. Conf. on Neural Info. Processing Sys. (NIPS), Montreal, Canada, June 2015.

In this approach, rather than using MCMC to produce a number of independent samples used to collect the statistics in the negative training phase, 1st, 2nd, and 3rd order mean-field approximations are used to estimate equilibrium magnetizations on both the visible and hidden units. These real-valued magnetizations are then used in lieu of binary particles.

```julia
    ApproxIter = 3      # How many fixed-point EMF steps to take

    # ...etc...

    # Train using 1st order mean-field (naïve mean field)
    fit(rbm1,TrainData; approx="naive", NormalizationApproxIter=ApproxIter)

    # Train using 2nd order mean-field
    fit(rbm2,TrainData; approx="tap2", NormalizationApproxIter=ApproxIter)

    # Train using 3rd order mean-field
    fit(rbm3,TrainData; approx="tap3", NormalizationApproxIter=ApproxIter)
```

MNIST Example
-------------

One can find the script for this example inside the `/examples` directory [of the repository](https://github.com/sphinxteam/Boltzmann.jl/blob/master/examples/mnistexample.jl).

Sampling
--------

After training an RBM, one can generate samples from the distribution it has been trained to model. To start the sampling chain, one needs to provide an initialization to the visible layer. This can be either a sample from the training set or some random initialization, depending on the task to be accomplished. Below we see a short script to accomplish this sampling. 

```julia
# Experimental Parameters
    NFeatures = 100

    # Generate a random binary initialization
    vis_init = rand(NFeatures,1)
    binarize!(vis_init;threshold=0.5)

    # Obtain the number of desired samples
    vis_samples = generate(rbm,       # Trained RBM Model to sample from
                           vis_init,   # Starting point for sampling chain
                           "CD",       # Sampling method, here, MCMC/Gibbs
                           100)        # Number of steps to take on sampling chain
```


RBM Variants
------------

Currently, this version of the Boltzmann package only provides support for the following RBM variants:

 - `BernoulliRBM`: RBM with binary visible and hidden units.

Support for real valued visibile units is still in progress. Some basic functionality for this feature was provided in limited, though unverified way, in the [upstream repository of this fork](https://https://github.com/dfdx/Boltzmann.jl). We suggest waiting until a verified implementation of the G-RBM is provided, here.

Integration with Mocha
----------------------

[Mocha.jl](https://github.com/pluskid/Mocha.jl) is an excellent deep learning framework implementing auto-encoders and a number of fine-tuning algorithms. Boltzmann.jl allows to save pretrained model in a Mocha-compatible file format to be used later on for supervised learning. Below is a snippet of the essential API, while complete code is available in [Mocha Export Example](https://github.com/dfdx/Boltzmann.jl/blob/master/examples/mocha_export_example.jl):

```julia
    # pretraining and exporting in Boltzmann.jl
    dbn_layers = [("vis", GRBM(100, 50)),
                  ("hid1", BernoulliRBM(50, 25)),
                  ("hid2", BernoulliRBM(25, 20))]
    dbn = DBN(dbn_layers)
    fit(dbn, X)
    save_params(DBN_PATH, dbn)

    # loading in Mocha.jl
    backend = CPUBackend()
    data = MemoryDataLayer(tops=[:data, :label], batch_size=500, data=Array[X, y])
    vis = InnerProductLayer(name="vis", output_dim=50, tops=[:vis], bottoms=[:data])
    hid1 = InnerProductLayer(name="hid1", output_dim=25, tops=[:hid1], bottoms=[:vis])
    hid2 = InnerProductLayer(name="hid2", output_dim=20, tops=[:hid2], bottoms=[:hid1])
    loss = SoftmaxLossLayer(name="loss",bottoms=[:hid2, :label])
    net = Net("TEST", backend, [data, vis, hid1, hid2])

    h5open(DBN_PATH) do h5
        load_network(h5, net)
    end
```


