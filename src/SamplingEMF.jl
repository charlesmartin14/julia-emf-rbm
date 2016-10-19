using Distributions
using Base.LinAlg.BLAS

### Base MF definitions
#### Naive mean field
function mag_vis_naive(rbm::RBM, m_hid::Mat{Float64}) ## to be constrained to being only Bernoulli
    buf = gemm('T', 'N', rbm.W, m_hid) .+ rbm.vbias
    return logsig(buf)
end    
# Defining a method with same arguments as other mean field approxiamtions
mag_vis_naive(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64})=mag_vis_naive(rbm, m_hid) 

function mag_hid_naive(rbm::RBM, m_vis::Mat{Float64}) 
    buf = gemm('N', 'N', rbm.W, m_vis) .+ rbm.hbias
    return logsig(buf)
end    

mag_hid_naive(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64})=mag_hid_naive(rbm, m_vis) 
  
#### Second order development
function mag_vis_tap2(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64}) ## to be constrained to being only Bernoulli
    buf = gemm('T', 'N', rbm.W, m_hid) .+ rbm.vbias
    
    
    
    dv = 0.5-m_vis
    
    ggg =  gemm('T', 'N', rbm.W2, m_hid-abs2(m_hid))
    
    
    
    second_order = gemm('T', 'N', rbm.W2, m_hid-abs2(m_hid)).*(0.5-m_vis)
    axpy!(1.0, second_order, buf)
    return logsig(buf)
end  

function mag_hid_tap2(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64})
    buf = gemm('N', 'N', rbm.W, m_vis) .+ rbm.hbias
    second_order = gemm('N', 'N', rbm.W2, m_vis-abs2(m_vis)).*(0.5-m_hid)
    axpy!(1.0, second_order, buf)
    return logsig(buf)
end

#### Third order development

function mag_vis_tap3(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64}) ## to be constrained to being only Bernoulli
    buf = gemm('T', 'N', rbm.W, m_hid) .+ rbm.vbias
    second_order = gemm('T', 'N', rbm.W2, m_hid-abs2(m_hid)).*(0.5-m_vis)
    third_order = gemm('T', 'N', rbm.W3, abs2(m_hid).*(1.-m_hid)).*(1/3-2*(m_vis-abs2(m_vis)))
    axpy!(1.0, second_order, buf)
    axpy!(1.0, third_order, buf)
    return logsig(buf)
end  

function mag_hid_tap3(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64})
    buf = gemm('N', 'N', rbm.W, m_vis) .+ rbm.hbias
    second_order = gemm('N', 'N', rbm.W2, m_vis-abs2(m_vis)).*(0.5-m_hid)
    third_order = gemm('N', 'N', rbm.W3, abs2(m_vis).*(1.-m_vis)).*(1/3-2*(m_hid-abs2(m_hid)))
    axpy!(1.0, second_order, buf)
    axpy!(1.0, third_order, buf)
    return logsig(buf)
end

function equilibrate(rbm::RBM, vis_init::Mat{Float64}, hid_init::Mat{Float64}; iterations=3, approx="tap2", damp=0.5)
    # Redefine names for clarity
    m_vis = copy(vis_init)
    m_hid = copy(hid_init)
    
    # Set the proper iteration based on the approximation type
    if approx == "naive"
        mag_vis = mag_vis_naive
        mag_hid = mag_hid_naive
    elseif approx == "tap3"
        mag_vis = mag_vis_tap3
        mag_hid = mag_hid_tap3
    else    
        mag_vis = mag_vis_tap2
        mag_hid = mag_hid_tap2
    end
    
    
    
    # Take the desired number of steps
    for i=1:iterations
       m_vis = 0.5 * mag_vis(rbm, m_vis, m_hid) + 0.5 * m_vis
       m_hid = 0.5 * mag_hid(rbm, m_vis, m_hid) + 0.5 * m_hid
       
    end

    return m_vis, m_hid
end

##-----------------------------------------------------------------------------##
### To be deprecated
function iter_mag(rbm::RBM, vis::Mat{Float64}; n_times=3, approx="tap2")
    v_pos = vis
    h_pos = ProbHidCondOnVis(rbm, v_pos)

    if approx == "naive"
        mag_vis = mag_vis_naive
        mag_hid = mag_hid_naive
    elseif approx == "tap3"
        mag_vis = mag_vis_tap3
        mag_hid = mag_hid_tap3
    else    
        mag_vis = mag_vis_tap2
        mag_hid = mag_hid_tap2
    end    

    
    m_vis = 0.5 * mag_vis(rbm, vis, h_pos) + 0.5 * vis
    m_hid = 0.5 * mag_hid(rbm, m_vis, h_pos) + 0.5 * h_pos
    for i=1:2
       
       m_vis = 0.5 * mag_vis(rbm, m_vis, m_hid) + 0.5 * m_vis
       m_hid = 0.5 * mag_hid(rbm, m_vis, m_hid) + 0.5 * m_hid
    end
    

    return v_pos, h_pos, m_vis, m_hid
end    

function iter_mag_persist!(rbm::RBM, vis::Mat{Float64}; n_times=3, approx="tap2")
    v_pos = vis
    h_pos = ProbHidCondOnVis(rbm, v_pos)
    
    if approx == "naive"
        mag_vis = mag_vis_naive
        mag_hid = mag_hid_naive
    elseif approx == "tap3"
        mag_vis = mag_vis_tap3
        mag_hid = mag_hid_tap3
    else    
        mag_vis = mag_vis_tap2
        mag_hid = mag_hid_tap2
    end    

    m_vis = rbm.persistent_chain_vis
    m_hid = rbm.persistent_chain_hid

    for i=1:n_times
       m_vis = 0.5 * mag_vis(rbm, m_vis, m_hid) + 0.5 * m_vis
       m_hid = 0.5 * mag_hid(rbm, m_vis, m_hid) + 0.5 * m_hid
    end

    rbm.persistent_chain_vis = m_vis
    rbm.persistent_chain_hid = m_hid

    return v_pos, h_pos, m_vis, m_hid
end 
