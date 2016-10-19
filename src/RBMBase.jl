using Distributions
using Base.LinAlg.BLAS
using Devectorize
import Base.getindex

typealias Gaussian Normal
abstract AbstractRBM

"""
    # Boltzmann.RBM{V,H} (RBMBase.jl)
    ## Description
        A structure for containing all of the restricted Boltzmann Machine (RBM)
        model parameters. Besides just the model parameters (couplings, biases),
        the structure also contains variables which are pertinent to the RBM training
        procedure.

    ## Structure
        - `W::Matrix{Float64}`:       The matrix of coupling parameters (RBM model parameter)
        - `W2::Matrix{Float64}`:      The square of `W` (used for EMF learning)
        - `W3::Matrix{Float64}`:      The cube of `W` (used for EMF learning)
        - `vbias::Vector{Float64}`:   The visible unit biases (RBM model parameter)
        - `hbias::Vector{Float64}`:   The hidden unit biases (RBM model parameter)
        - `dW::Matrix{Float64}`:      The current gradient on the coupling parameters (used for RBM training)
        - `dW_prev::Matrix{Float64}`: The last gradient on the coupling parmaters (used for RBM training)
        - `persistent_chain_vis::Matrix{Float64}`: Visible fantasy particles (used for RBM persistent mode training)
        - `persistent_chain_hid::Matrix{Float64}`: Hidden fantasy particles (used for RBM persistent mode training)
        - `momentum::Float64`:        Amount of last gradient to add to the current gradient (used for RBM training)
        - `VisShape::Tuple{Int,Int}`: Final output shape of the visible units
"""
type RBM{V,H} <: AbstractRBM
    W::Matrix{Float64}
    W2::Matrix{Float64}
    W3::Matrix{Float64}
    vbias::Vector{Float64}
    hbias::Vector{Float64}
    dW::Matrix{Float64}
    dW_prev::Matrix{Float64}
    persistent_chain_vis::Matrix{Float64}
    persistent_chain_hid::Matrix{Float64}
    momentum::Float64
    VisShape::Tuple{Int,Int}
end

function RBM(V::Type, H::Type, n_vis::Int, n_hid::Int, visshape::Tuple{Int,Int}; sigma=0.01, momentum=0.0, TrainData=[])
    # Some "tiny" value, used to enforce min/max boundary conditions
    eps = 1e-8

    # Initialize the weighting matrix by drawing from an iid Gaussian 
    # of the specified standard deviation.
    
    W = rand(Normal(0, sigma), (n_hid, n_vis))

    # If the user specifies the training dataset, it can be useful to
    # initialize the visibile biases according to the empirical expected
    # feature values of the training data.
    #
    # TODO: Generalize this biasing. Currently, the biasing is only written for 
    #       the case of binary RBMs.
    InitialVisBias = zeros(n_vis,1)
    if !isempty(TrainData)
        
        x1 = TrainData[:,1]
        
        ProbVis = mean(TrainData,2)             # Mean across  samples
        
        ProbVis = max(ProbVis,eps)              # Some regularization (avoid Inf/NaN)
        ProbVis = min(ProbVis,1 - eps)          # ''
        @devec InitialVisBias = log(ProbVis ./ (1-ProbVis)) # Biasing as the log-proportion
        
        
    end

    RBM{V,H}(W,                                             # W
             W.*W,                                          # W2
             W.*W.*W,                                       # W3
             vec(InitialVisBias),                           # vbias
             zeros(n_hid),                                  # hbias
             zeros(n_hid, n_vis),                           # dW
             zeros(n_hid, n_vis),                           # dW_prev
             Array(Float64, 0, 0),                          # persistent_chain_vis
             Array(Float64, 0, 0),                          # persistent_chain_hid
             momentum,                                      # momentum
             visshape)                                      # Shape of the visible units (for display)
end


function Base.show{V,H}(io::IO, rbm::RBM{V,H})
    n_vis = size(rbm.vbias, 1)
    n_hid = size(rbm.hbias, 1)
end


typealias BernoulliRBM RBM{Bernoulli, Bernoulli}
BernoulliRBM(n_vis::Int, n_hid::Int, visshape::Tuple{Int,Int}; sigma=0.1, momentum=0.0, TrainData=[]) =
    RBM(Bernoulli, Bernoulli, n_vis, n_hid, visshape; sigma=sigma, momentum=momentum, TrainData=TrainData)

typealias GRBM RBM{Gaussian, Bernoulli}
GRBM(n_vis::Int, n_hid::Int, visshape::Tuple{Int,Int}; sigma=0.1, momentum=0.0, TrainData=[]) =
    RBM(Gaussian, Bernoulli, n_vis, n_hid, visshape; sigma=sigma, momentum=momentum, TrainData=TrainData)


"""
    # Boltzmann.PassHidToVis  (RBMBase.jl)
"""
function PassHidToVis(rbm::RBM, hid::Mat{Float64})
    return rbm.W' * hid .+ rbm.vbias
end

"""
    # Boltzmann.PassVisToHid  (RBMBase.jl)
"""
function PassVisToHid(rbm::RBM, vis::Mat{Float64})
    p =  rbm.W * vis .+ rbm.hbias
    return p
end

### These functions need to be generalized to detect the Distribution on 
### the hidden and visible variables.
"""
    # Boltzmann.ProbHidCondOnVis  (RBMBase.jl)
"""
function ProbHidCondOnVis(rbm::RBM, vis::Mat{Float64})
    return logsig(PassVisToHid(rbm,vis))
end

"""
    # Boltzmann.ProbVisCondOnHid  (RBMBase.jl)
"""
function ProbVisCondOnHid(rbm::RBM, hid::Mat{Float64})
    return logsig(PassHidToVis(rbm,hid))
end

"""
    # Boltzmann.PinField! (RBMBase.jl)
"""
function PinField!(rbm::RBM,pinning_field::Vec{Float64})
    pos_inf_locations = pinning_field > 0
    neg_inf_locations = pinning_field < 0

    rbm.vbias(pos_inf_locations) =  Inf
    rbm.vbias(neg_inf_locations) = -Inf
end
