
module Boltzmann

export RBM,
       BernoulliRBM,
       Monitor,
       Update!,
       GRBM,
       DBN,
       DAE,
       fit,
       transform,
       sample_hiddens,
       sample_visibles,
       mag_vis_tap2,
       mag_hid_tap2,
       generate,
       components,
       features,
       unroll,
       save_params,
       load_params,
       chart_weights,
       chart_weights_distribution,
       chart_activation_distribution,
       binarize,
       binarize!,       
       normalize,
       normalize!,
       normalize_samples,
       normalize_samples!,
       ShowMonitor,
       WriteMonitorChartPDF,
       SaveMonitorHDF5,
       plot_scores,
       plot_evolution,
       plot_rf,
       plot_chain,
       plot_vbias,
       plot_weightdist,
       chart_likelihood_evolution,
       PinField!

include("core.jl")

end