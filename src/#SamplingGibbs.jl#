using Distributions
using Base.LinAlg.BLAS

function sample(::Type{Bernoulli}, means::Mat{Float64})
    s = zeros(means)
    r = rand(size(means))
    @simd for i=1:length(means)
        @inbounds s[i] = r[i] < means[i] ? 1.0 : 0.0
    end

    return s
end

function sample(::Type{Gaussian}, means::Mat{Float64})
    sigma2 = 1                   # using fixed standard diviation
    samples = zeros(size(means))
    for j=1:size(means, 2), i=1:size(means, 1)
        samples[i, j] = rand(Normal(means[i, j], sigma2))
    end
    return samples
end
    
function sample_hiddens{V,H}(rbm::RBM{V,H}, vis::Mat{Float64})
    means = ProbHidCondOnVis(rbm, vis)
    return sample(H, means), means
end

function sample_visibles{V,H}(rbm::RBM{V,H}, hid::Mat{Float64})
    means = ProbVisCondOnHid(rbm, hid)
    return sample(V, means), means
end


function gibbs(rbm::RBM, vis::Mat{Float64}; n_times=1)
    v_pos = vis
    h_samp, h_pos = sample_hiddens(rbm, v_pos)
    h_neg = Array(Float64,0,0)::Mat{Float64}
    v_neg = Array(Float64,0,0)::Mat{Float64}
    if n_times > 0
    # Save computation by setting `n_times=0` in the case
    # of persistent CD.
        v_neg = sample_visibles(rbm, h_samp)
        h_samp, h_neg = sample_hiddens(rbm, v_neg)
        for i=1:n_times-1
            v_neg = sample_visibles(rbm, h_samp)
            h_samp, h_neg = sample_hiddens(rbm, v_neg)
        end
    end
    return v_pos, h_pos, v_neg, h_neg
end

function MCMC(rbm::RBM, init::Mat{Float64}; iterations=1, StartMode="visible")    
    if StartMode == "visible"
    # In this first mode we assume that we are starting from the visible samples. E.g. in
    # the case of binary RBM, we should be starting with binary samples.
        vis_samples = copy(init)                                    # Start from the visible samples
        vis_means   = copy(init)                                    # Giving a starting point for the means
        hid_samples, hid_means = sample_hiddens(rbm,vis_samples)     # Get the first hidden means [NMF-ish]
    end

    if StartMode == "hidden"
    # In this second mode we assume that we are starting from a set of hidden
    # samples. Because of this, we increment the iteration count by 1
        hid_samples = copy(init)
        hid_means = copy(init)
        iterations+=1
    end
    
    for i=1:iterations-1
        vis_samples, vis_means = sample_visibles(rbm,hid_samples)          # Sample the visible units from true distribution
        hid_samples, hid_means = sample_hiddens(rbm,vis_samples)           # Update the hidden unit means, a NMF-ish approach
    end

    return vis_samples, hid_samples, vis_means, hid_means
end
