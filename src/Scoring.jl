using Base.LinAlg.BLAS

function free_energy(rbm::RBM, vis::Mat{Float64})
    vb = sum(vis .* rbm.vbias, 1)
    Wx_b_log = sum(log(1 + exp(rbm.W * vis .+ rbm.hbias)), 1)
    return - vb - Wx_b_log
end

function score_samples(rbm::RBM, vis::Mat{Float64}; sample_size=10000)
    if issparse(vis)
        # sparse matrices may be infeasible for this operation
        # so using only little sample
        cols = sample(1:size(vis, 2), sample_size)
        vis = full(vis[:, cols])
    end
    n_feat, n_samples = size(vis)
    vis_corrupted = copy(vis)
    idxs = rand(1:n_feat, n_samples)
    for (i, j) in zip(idxs, 1:n_samples)
        vis_corrupted[i, j] = 1 - vis_corrupted[i, j]
    end
    fe = free_energy(rbm, vis)
    fe_corrupted = free_energy(rbm, vis_corrupted)
    return n_feat * log(logsig(fe_corrupted - fe))
end

function recon_error(rbm::RBM, vis::Mat{Float64})
    # Fully forward MF operation to get back to visible samples
    vis_rec = ProbVisCondOnHid(rbm,ProbHidCondOnVis(rbm,vis))
    # Get the total error over the whole tested visible set,
    # here, as MSE
    dif = vis_rec - vis
    mse = mean(dif.*dif)
    return mse
end

function score_samples_TAP(rbm::RBM, vis::Mat{Float64}; n_iter=5)
    _, _, m_vis, m_hid = iter_mag(rbm, vis; n_times=n_iter, approx="tap2")
    eps=1e-8
    m_vis = max(m_vis, eps)
    m_vis = min(m_vis, 1.0-eps)
    m_hid = max(m_hid, eps)
    m_hid = min(m_hid, 1.0-eps)

    m_vis2 = abs2(m_vis)
    m_hid2 = abs2(m_hid)

    S = - sum(m_vis.*log(m_vis)+(1.0-m_vis).*log(1.0-m_vis),1) - sum(m_hid.*log(m_hid)+(1.0-m_hid).*log(1.0-m_hid),1)
    U_naive = - gemv('T',m_vis,rbm.vbias)' - gemv('T',m_hid,rbm.hbias)' - sum(gemm('N','N',rbm.W,m_vis).*m_hid,1)
    # why is this not 2 blass calls ?
    Onsager = - 0.5 * sum(gemm('N','N',rbm.W2,m_vis-m_vis2).*(m_hid-m_hid2),1)
    
    
    fe_tap = U_naive + Onsager - S
    fe = free_energy(rbm, vis)
    return fe_tap - fe
end

function score_entropy_TAP(rbm::RBM, vis::Mat{Float64}; n_iter=3)
    _, _, m_vis, m_hid = iter_mag(rbm, vis; n_times=n_iter, approx="tap2")
    eps=1e-8 # why was this 10-6 ? 
    m_vis = max(m_vis, eps)
    m_vis = min(m_vis, 1.0-eps)
    m_hid = max(m_hid, eps)
    m_hid = min(m_hid, 1.0-eps)

    m_vis2 = abs2(m_vis)
    m_hid2 = abs2(m_hid)
    
    

    
    
    S = - sum(m_vis.*log(m_vis)+(1.0-m_vis).*log(1.0-m_vis),1) - sum(m_hid.*log(m_hid)+(1.0-m_hid).*log(1.0-m_hid),1)
    
    return S
end

function score_free_energy(rbm::RBM, vis::Mat{Float64})
    return sum(free_energy(rbm,vis))
end

function score_U_naive(rbm::RBM, vis::Mat{Float64}; n_iter=5)
    _, _, m_vis, m_hid = iter_mag(rbm, vis; n_times=n_iter, approx="tap2")
    eps=1e-8
    m_vis = max(m_vis, eps)
    m_vis = min(m_vis, 1.0-eps)
    m_hid = max(m_hid, eps)
    m_hid = min(m_hid, 1.0-eps)

    U_naive = - gemv('T',m_vis,rbm.vbias)' - gemv('T',m_hid,rbm.hbias)' - sum(gemm('N','N',rbm.W,m_vis).*m_hid,1)
    return U_naive
end 
