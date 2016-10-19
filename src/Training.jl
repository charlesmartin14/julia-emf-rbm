
using Distributions
using ProgressMeter
using Base.LinAlg.BLAS
using Compat
using Devectorize
using HDF5
using PyCall
@pyimport matplotlib.pyplot as plt
@pyimport numpy as np

import StatsBase.fit

function calculate_weight_gradient!(rbm::RBM, h_pos::Mat{Float64}, v_pos::Mat{Float64}, h_neg::Mat{Float64}, v_neg::Mat{Float64}, lr::Float64; approx="CD")
    info("calculate_weight_gradient")
    ## Load step buffer with negative-phase    
    gemm!('N', 'T', lr, h_neg, v_neg, 0.0, rbm.dW)          # dW <- LearRate*<h_neg,v_neg>
    ## Stubtract step buffer from positive-phase to get gradient    
    gemm!('N', 'T', lr, h_pos, v_pos, -1.0, rbm.dW)         # dW <- LearnRate*<h_pos,v_pos> - dW

    info("dW naive: ",norm(rbm.dW)/lr)
    ## Second-Order EMF Correction (for EMF-TAP2, EMF-TAP3)
    if contains(approx,"tap") 
        buf2 = gemm('N', 'T', h_neg-abs2(h_neg), v_neg-abs2(v_neg)) .* rbm.W  
        axpy!(-lr, buf2, rbm.dW)
    end

    info("dW  tap2: ",norm(rbm.dW/lr),"dW buf2: ",norm(buf2))
    
    ## Third-Order EMF Correction (for EMF-TAP3)
    if approx == "tap3"
        buf3 = gemm('N','T', (h_neg-abs2(h_neg)) .* (0.5-h_neg), (v_neg-abs2(v_neg)) .* (0.5-v_neg)) .* rbm.W2
        axpy!(-2.0*lr, buf3, rbm.dW)  
    end    
    ## Apply Momentum (adding last gradient to this one)    
    axpy!(rbm.momentum, rbm.dW_prev, rbm.dW)    # rbm.dW += rbm.momentum * rbm.dW_prev
    info("dW, past dW_prev mom: ",norm(rbm.dW/lr), "   ",norm(rbm.dW_prev/lr))
end

function update_weights!(rbm::RBM,approx::AbstractString)
    axpy!(1.0,rbm.dW,rbm.W)             # Take step: W = W + dW
    copy!(rbm.dW_prev, rbm.dW)          # Save the current step for future use
    info("lr * dW, curr dW_prev mom: ",norm(rbm.dW), "   ",norm(rbm.dW_prev))
    # if contains(approx,"tap")
    rbm.W2 = rbm.W  .* rbm. W       # Update Square [for EMF-TAP2]
    # end
    if approx == "tap3"
        rbm.W3 = rbm.W2 .* rbm.W        # Update Cube   [for EMF-TAP3]
    end
 end

function regularize_weight_gradient!(rbm::RBM,LearnRate::Float64;L2Penalty::Float64=NaN,L1Penalty::Float64=NaN,DropOutRate::Float64=NaN)
    ## Quadratic penalty on weights (Energy shrinkage)
    if !isnan(L2Penalty)
        axpy!(-LearnRate*L2Penalty,rbm.W,rbm.dW)
    end
    ## Linear penalty on weights (Sparsifying)
    if !isnan(L1Penalty)
        info("L1 reg ",-LearnRate, " ",L1Penalty, " ",norm(sign(rbm.W)), "  ", norm(rbm.W))
        axpy!(-LearnRate*L1Penalty,sign(rbm.W),rbm.dW)
        info("L1 reg ",-LearnRate, " ",L1Penalty, " ",norm(sign(rbm.W)), "  ", norm(rbm.W))
    end
    ## Dropout Regularization (restricted set of updates)
    if !isnan(DropOutRate)
        # Not yet implemented, so we do nothing.
        # TODO: Implement Drop-out, here.
    end
end

function get_negative_samples(rbm::RBM,vis_init::Mat{Float64},hid_init::Mat{Float64},approx::AbstractString, iterations::Int)
    if approx=="naive" || contains(approx,"tap")
        info("equlibirating")
        v_neg, h_neg = equilibrate(rbm,vis_init,hid_init; iterations=iterations, approx=approx)
    end

    if approx=="CD"
        # In the case of Gibbs/MCMC sampling, we will take the binary visible samples as the negative
        # visible samples, and the expectation (means) for the negative hidden samples.
        v_neg, _, _, h_neg = MCMC(rbm, hid_init; iterations=iterations, StartMode="hidden")
    end

    return v_neg, h_neg
end

function generate(rbm::RBM,vis_init::Mat{Float64},approx::AbstractString,SamplingIterations::Int)
    Nsamples = size(vis_init,2)
    Nhid     = size(rbm.hbias,1)
    h_init  = zeros(Nsamples,Nhid)

    if approx=="naive" || contains(approx,"tap")
        _, hid_mag = equilibrate(rbm,vis_init,hid_init; iterations=SamplingIterations, approx=approx)
    end

    if approx=="CD"
        _, hid_mag, _, _ = MCMC(rbm, vis_init; iterations=SamplingIterations, StartMode="visible")
    end

    samples,_ = sample_visibles(rbm,hid_mag)

    return reshape(samples,rbm.VisShape)
end

function fit_batch!(rbm::RBM, vis::Mat{Float64};
                    persistent=true, lr=0.1, NormalizationApproxIter=1,
                    weight_decay="none",decay_magnitude=0.01, approx="CD")

    # Determine how to acquire the positive samples based upon the persistence mode.
    info("-----fit batch------")
    v_pos = vis
    info("size of batch  ",size(vis))
    info("norm of batch  ",norm(vis))

    info("start batch norm of W, hb, vb  ",norm(rbm.W), " ",norm(rbm.hbias)," ",norm(rbm.vbias))

    h_samples, h_pos = sample_hiddens(rbm,v_pos)

    info("norm sampled hpos  ",norm(h_pos))
    # Set starting points in the case of persistence
    if persistent
        if approx=="naive" || contains(approx,"tap")
            v_init = copy(rbm.persistent_chain_vis)      
            h_init = copy(rbm.persistent_chain_hid)       
        end
        if approx=="CD" 
            v_init = vis               # A dummy setting
            h_init,_ = sample_hiddens(rbm,rbm.persistent_chain_vis)
        end
    else
        if approx=="naive" || contains(approx,"tap")
            info("init tap v,h")
            v_init = vis
            h_init = h_pos
            # is the diff just noise in W?
        end
        if approx=="CD"
            v_init = vis               # A dummy setting
            h_init = h_samples
        end
    end

    info("v, h init [0]",norm(v_init)," ",norm(h_init))
    
    # Calculate the negative samples according to the desired approximation mode
    v_neg, h_neg = get_negative_samples(rbm,v_init,h_init,approx,NormalizationApproxIter)
    info("v, h neg ",norm(v_neg)," ",norm(h_neg))
    
    # If we are in persistent mode, update the chain accordingly
    if persistent
        copy!(rbm.persistent_chain_vis,v_neg)
        copy!(rbm.persistent_chain_hid,h_neg)
    end

    # Update on weights
    calculate_weight_gradient!(rbm,h_pos,v_pos,h_neg,v_neg,lr,approx=approx)
    info("dW norm ",norm(rbm.dW/lr))

    if weight_decay == "l2"
        regularize_weight_gradient!(rbm,lr;L2Penalty=decay_magnitude)
    end
    if weight_decay == "l1"
        regularize_weight_gradient!(rbm,lr;L1Penalty=decay_magnitude)
        info("reg L1 ",norm(rbm.dW/lr))
    end
    info(" BUW dW norm ",norm(rbm.dW/lr))
    # dW should not change here, but it does !  not sure why
    update_weights!(rbm,approx)
    info("prev dW updated ",norm(rbm.dW_prev/lr))

    # Gradient update on biases
    rbm.hbias += vec(lr * (sum(h_pos, 2) - sum(h_neg, 2)))
    rbm.vbias += vec(lr * (sum(v_pos, 2) - sum(v_neg, 2)))

    println("updated h_bias ", norm(rbm.hbias))
    println("updated v_bias ", norm(rbm.vbias))

    pl = mean(score_samples(rbm, vis))    
    tap_free_energy = mean(score_samples_TAP(rbm, vis))
    entropy = mean(score_entropy_TAP(rbm, vis))
    u_naive = mean(score_U_naive(rbm, vis))
    free_energy = mean(score_free_energy(rbm, vis))
    println("end batch norm of W, hb, vb  ",norm(rbm.W), " ",norm(rbm.hbias)," ",norm(rbm.vbias))
    println("pseudo l-hood: ",pl,"\n")
    println("entropy: ",entropy,"\n")
    println("TAP free_energy: ",tap_free_energy,"\n")
    println("U naive: ",u_naive,"\n")
    println("free energy: ",free_energy,"\n")
    flush(STDOUT)
    return rbm
end


"""
    # Boltzmann.fit (training.jl)
    ## Function Call
        `fit(rbm::RBM, X::Mat{Float64}[, persistent, lr, batch_size, NormalizationApproxIter, weight_decay, 
                                         decay_magnitude, validation,monitor_ever, monitor_vis,
                                         approx, persistent_start, save_params])`
    ## Description
    The core RBM training function. Learns the weights and biasings using 
    either standard Contrastive Divergence (CD) or Persistent CD, depending on
    the user options. 
    
    - *rbm:* RBM object, initialized by `RBM()`/`GRBM()`
    - *X:* Set of training vectors

    ### Optional Inputs
     - *persistent:* Whether or not to use persistent-CD [default=true]
     - *persistent_start:* At which epoch to start using the persistent chains. Only
                           applicable for the case that `persistent=true`.
                           [default=1]
     - *lr:* Learning rate [default=0.1]
     - *n_iter:* Number of training epochs [default=10]
     - *batch_size:* Minibatch size [default=100]
     - *NormalizationApproxIter:* Number of Gibbs sampling steps on the Markov Chain [default=1]
     - *weight_decay:* A string value representing the regularization to add to apply to the 
                       weight magnitude during training {"none","l1","l2"}. [default="none"]
     - *decay_magnitude:* Relative importance assigned to the weight regularization. Smaller
                          values represent less regularization. Should be in range (0,1). 
                          [default=0.01]
     - *validation:* An array of validation samples, e.g. a held out set of training data.
                     If passed, `fit` will also track generalization progress during training.
                     [default=empty-set]
     - *score_every:* Controls at which epoch the progress of the fit is monitored. Useful to 
                      speed up the fit procedure if detailed progress monitoring is not required.
                      [default=5]
     - *save_progress:* Controls the saving of RBM parameters throughout the course of the training.
                     Should be passed as a tuple in the following manner:
                        (::AbstractString,::Int)                      
                     where the first field is the filename for the HDF5 used to save results and
                     the second field specifies how often to write the parameters to disk. All
                     results will be stored in the specified HDF5 file under the root headings
                        `Epochxxxx___weight`
                        `Epochxxxx___vbias`
                        `Epochxxxx___bias`
                     where `xxxx` specifies the epoch number in the `%04d` format.   
                     [default=nothing]    

    ## Returns
     - *::RBM* -- A trained RBM model.
     - *::Monitor* -- A Monitor structure containing information on the training progress over
                      epochs.
"""
function fit(rbm::RBM, X::Mat{Float64};
             persistent=true, lr=0.1, n_iter=10, batch_size=20, NormalizationApproxIter=1,
             weight_decay="none",decay_magnitude=0.01,validation=[],
             monitor_every=5,monitor_vis=false, approx="CD",
             persistent_start=1, save_progress=nothing)

    # TODO: This line needs to be changed to accomodate real-valued units
    @assert minimum(X) >= 0 && maximum(X) <= 1

    info("RBM v bias",norm(rbm.vbias))
    n_valid=0
    n_features = size(X, 1)
    n_samples = size(X, 2)
    n_hidden = size(rbm.W,1)
    n_batches = @compat Int(ceil(n_samples / batch_size))
    N = n_hidden+n_features

    # Check for the existence of a validation set
    flag_use_validation=false
    if length(validation)!=0
        flag_use_validation=true
        n_valid=size(validation,2)        
    end

    # Create the historical monitor
    ProgressMonitor = Monitor(n_iter,monitor_every;monitor_vis=monitor_vis,
                                                   validation=flag_use_validation)

    # Print info to user
    m_ = rbm.momentum
    info("=====================================")
    info("RBM Training")
    info("=====================================")
    info("  + Training Samples:     $n_samples")
    info("  + Features:             $n_features")
    info("  + Hidden Units:         $n_hidden")
    info("  + Epochs to run:        $n_iter")
    info("  + Persistent ?:         $persistent")
    info("  + Training approx:      $approx")
    info("  + Momentum:             $m_")
    info("  + Learning rate:        $lr")
    info("  + Norm. Approx. Iters:  $NormalizationApproxIter")   
    info("  + Weight Decay?:        $weight_decay") 
    info("  + Weight Decay Mag.:    $decay_magnitude")
    info("  + Validation Set?:      $flag_use_validation")    
    info("  + Validation Samples:   $n_valid")   
    info("=====================================")

    # Scale the learning rate by the batch size
    lr=lr/batch_size

    # Random initialization of the persistent chains
    rbm.persistent_chain_vis,_ = random_columns(X,batch_size)
    rbm.persistent_chain_hid = ProbHidCondOnVis(rbm, rbm.persistent_chain_vis)


    
    use_persistent = false
    for itr=1:n_iter
        # Check to see if we can use persistence at this epoch
        use_persistent = itr>=persistent_start ? persistent : false

        tic()

        # Mini-batch fitting loop. 
        @showprogress 1 "Fitting Batches..." for i=1:n_batches
            batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
            info("batch info",i,size(batch), norm(batch))
            batch = full(batch)

            info("norm W vb hb ",norm(rbm.W)," ",norm(rbm.vbias)," ",norm(rbm.hbias))
            fit_batch!(rbm, batch; persistent=use_persistent, 
                                   NormalizationApproxIter=NormalizationApproxIter,
                                   weight_decay=weight_decay,
                                   decay_magnitude=decay_magnitude,
                                   lr=lr, approx=approx)
            
        end
        
        # Get the average wall-time in µs
        walltime_µs=(toq()/n_batches/N)*1e6
        
#        UpdateMonitor!(rbm,ProgressMonitor,X,itr;bt=walltime_µs,validation=validation,lr=lr,mo=rbm.momentum)
#        ShowMonitor(rbm,ProgressMonitor,X,itr)

        # Attempt to save parameters if need be
        if save_progress != nothing 
            if itr%save_progress[2]==0
                rootName = @sprintf("Epoch%04d",itr)
                if isfile(save_progress[1])
                    info("Appending Params...")
                    append_params(save_progress[1],rbm,rootName)
                else
                    info("Creating file and saving params...")
                    save_params(save_progress[1],rbm,rootName)
                end
            end
        end
    end

#    if monitor_vis 
#        plt.close()
#    end 
    
    return rbm, ProgressMonitor
end
