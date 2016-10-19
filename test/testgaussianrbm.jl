using Colors
using Images
using Boltzmann
using MNIST
using ImageView
using Devectorize

function run_mnist()
    HiddenUnits = 100;
    Epochs = 20;

    X, y = testdata()  # test data is smaller, no need to downsample
    X = X ./ (maximum(X) - minimum(X))

    Nfeatures = size(X,1)
    Nsamples = size(X,2)

    # Mean subtraction
    mu = mean(X,2)
    X = broadcast(-,X,mu)

    # Scaling to unit variance
    Xvars = sqrt(sum(X.^2,1))    
    X = sqrt(Nfeatures) * broadcast(/,X,Xvars)

    m = GRBM(28*28, HiddenUnits, (28,28); momentum=0.0, dataset=X)
    
    # Attempt without weight decay
    info("Running With Gaussian Visible Units")
    fit(m, X;n_iter=Epochs,lr=0.000005,persistent=false,weight_decay="l2",decay_magnitude=0.05)
    chart_weights(m.W, (28, 28); annotation="Gaussian Visible")
    chart_weights_distribution(m.W;filename="grbm_distribution.pdf",bincount=200)    

    return m
end

run_mnist()

println("Press RETURN when ready")
readline(STDIN)

