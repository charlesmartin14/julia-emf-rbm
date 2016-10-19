using PyCall
@pyimport matplotlib.pyplot as plt

function chart_weights(W, imsize; padding=0, annotation="", filename="", noshow=false, ordering=true)
    h, w = imsize
    n = size(W, 1)
    rows = round(Int,floor(sqrt(n)))
    cols = round(Int,ceil(n / rows))
    halfpad = div(padding, 2)
    dat = zeros(rows * (h + padding), cols * (w + padding))

    # Sort receptive fields by energy
    if ordering
        p = sum(W.^2,2)
        p = sortperm(vec(p);rev=true)
        W = W[p,:]
    end

    for i=1:n
        wt = W[i, :]
        wim = reshape(wt, imsize)
        r = div(i - 1, cols) + 1
        c = rem(i - 1, cols) + 1
        dat[(r-1)*(h+padding)+halfpad+1 : r*(h+padding)-halfpad,
            (c-1)*(w+padding)+halfpad+1 : c*(w+padding)-halfpad] = wim
    end

    normalize!(dat)

    return dat
end

function plot_hidden_activations(rbm::RBM,X::Mat{Float64})
    max_samples = 100
    n_samples = min(size(X,2),max_samples)
    x,_ = random_columns(X,n_samples)

    # Get all hidden activations for batch    
    act = ProbHidCondOnVis(rbm,x)
    # Show this matrix of activations
    plt.imshow(act;interpolation="Nearest")
    plt.title("Hidden Unit Activations")
    plt.xlabel("Random Samples")
    plt.ylabel("Hidden Unit Index")
    plt.gray()

end

function plot_scores(mon::Monitor)
    ax_pl = plt.gca()
    ax_re = ax_pl[:twinx]()
    
    hpl = ax_pl[:plot](mon.Epochs,mon.PseudoLikelihood,"b^-",label="Pseudo-Likelihood")
    htl = ax_pl[:plot](mon.Epochs,mon.TAPLikelihood,"g^-",label="Tap-Likelihood")
    if mon.UseValidation
        hvpl = ax_pl[:plot](mon.Epochs,mon.ValidationPseudoLikelihood,"b^:",label="Pseudo-Likelihood (Validation)")
        hvtl = ax_pl[:plot](mon.Epochs,mon.ValidationTAPLikelihood,"g^:",label="Tap-Likelihood (Validation)")
    end
    ax_pl[:set_ylabel]("Normalized Likelihood")
    ax_pl[:set_ylim]((-0.3,0.0))
    
    hre = ax_re[:plot](mon.Epochs,mon.ReconError,"-*r",label="Recon. Error")
    if mon.UseValidation
        hvre = ax_re[:plot](mon.Epochs,mon.ValidationReconError,":*r",label="Recon. Error (Validation)")
    end
    ax_re[:set_ylabel]("Value")            
    ax_re[:set_yscale]("log")

    plt.title("Scoring")
    plt.xlabel("Training Epoch")
    plt.xlim((1,mon.Epochs[mon.LastIndex]))        
    plt.grid("on")        
    if mon.UseValidation
        plt.legend(handles=[hpl;hvpl;htl;hvtl;hre;hvre],loc=3,fontsize=10)
    else
        plt.legend(handles=[hpl;htl;hre],loc=3,fontsize=10)
    end
end

function plot_evolution(mon::Monitor)
    hbt = plt.plot(mon.Epochs,mon.BatchTime_µs,"-k*",label="Norm. Batch time (µs)")

    plt.legend(handles=hbt,loc=1)
    plt.title("Evolution")
    plt.xlabel("Training Epoch")
    plt.xlim((1,mon.Epochs[mon.LastIndex]))        
    plt.grid("on")  
end

function plot_rf(rbm::RBM)
    # TODO: Implement RF display in the case of 1D signals
    rf = chart_weights(rbm.W,rbm.VisShape; padding=0,noshow=true)    
    plt.imshow(rf;interpolation="Nearest")
    plt.title("Receptive Fields")
    plt.gray()
end

function plot_chain(rbm::RBM)
    # TODO: Implement Chain display in the case of 1D signals
    pc = chart_weights(rbm.persistent_chain_vis',rbm.VisShape; padding=0,noshow=true,ordering=false)    
    plt.imshow(pc;interpolation="Nearest")
    plt.title("Visible Chain")
    plt.gray()
end

function plot_vbias(rbm::RBM)
    vectorMode = minimum(rbm.VisShape)==1 ? true : false

    if vectorMode
        plt.plot(rbm.vbias)
        plt.grid("on")        
    else
        plt.imshow(reshape(rbm.vbias,rbm.VisShape);interpolation="Nearest")
    end
    plt.title("Visible Biasing")
    plt.gray()
end

function plot_weightdist(rbm::RBM)
    plt.hist(vec(rbm.W);bins=100)
    plt.title("Weight Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
end

function figure_refresh(figureHandle)
    figureHandle[:canvas][:draw]()
    plt.show(block=false)
    plt.pause(0.0001)
end


function WriteMonitorChartPDF(rbm::RBM,mon::Monitor,X::Mat{Float64},filename::AbstractString)
    savefig = plt.figure(5;figsize=(12,15))
    # Show Per-Epoch Progres
    savefig[:add_subplot](231)
        plot_scores(mon)
        
    savefig[:add_subplot](232)
        plot_evolution(mon)      

    # Show Receptive fields
    savefig[:add_subplot](233)
        plot_rf(rbm)

    # Show the Visible chains/fantasy particle
    savefig[:add_subplot](234)
        plot_chain(rbm)

    # Show the current visible biasing
    savefig[:add_subplot](235)
        # plot_vbias(rbm)
        plot_hidden_activations(rbm,X)

    # Show the distribution of weight values
    savefig[:add_subplot](236)
        plot_weightdist(rbm)

    plt.savefig(filename;transparent=true,format="pdf",papertype="a4",frameon=true,dpi=300)
    plt.close()
end




function ShowMonitor(rbm::RBM,mon::Monitor,X::Mat{Float64},itr::Int;filename=[])
    fig = mon.FigureHandle

    if mon.MonitorVisual && itr%mon.MonitorEvery==0
        # Wipe out the figure
        fig[:clf]()

        # Show Per-Epoch Progres
        fig[:add_subplot](231)
            plot_scores(mon)
            
        fig[:add_subplot](232)
            plot_evolution(mon)      

        # Show Receptive fields
        fig[:add_subplot](233)
            plot_rf(rbm)

        # Show the Visible chains/fantasy particle
        fig[:add_subplot](234)
            plot_chain(rbm)

        # Show the current visible biasing
        fig[:add_subplot](235)
            # plot_vbias(rbm)
            plot_hidden_activations(rbm,X)

        # Show the distribution of weight values
        fig[:add_subplot](236)
            plot_weightdist(rbm)

        figure_refresh(fig)  
    end

    if mon.MonitorText && itr%mon.MonitorEvery==0
        li = mon.LastIndex
        ce = mon.Epochs[li]
        if mon.UseValidation
            @printf("[Epoch %04d] Train(pl : %0.3f, tl : %0.3f), Valid(pl : %0.3f, tl : %0.3f)  [%0.3f µsec/batch/unit]\n",ce,
                                                                                                   mon.TAPPseudoLikelihood[li],
                                                                                                   mon.TAPLikelihood[li],
                                                                                                   mon.ValidationPseudoLikelihood[li],
                                                                                                   mon.ValidationTAPLikelihood[li],
                                                                                                   mon.BatchTime_µs[li])
        else
            @printf("[Epoch %04d] Train(pl : %0.3f, tl : %0.3f)  [%0.3f µsec/batch]\n",ce,
                                                                           mon.TAPPseudoLikelihood[li],
                                                                           mon.TAPLikelihood[li],
                                                                           mon.BatchTime_µs[li])
        end
    end
end

# function chart_likelihood_evolution(pseudo, tap; filename="")

#     if length(filename) > 0
#         # Write to file if filename specified
#         LikelihoodPlot = plot(  
#                                 layer(x=1:length(pseudo),y=pseudo,Geom.point, Geom.line),
#                                 layer(x=1:length(tap),y=tap, Geom.point, Geom.line, Theme(default_color=colorant"green")),
#                                 Guide.xlabel("epochs"),Guide.ylabel("Likelihood"),Guide.title("Evolution of likelihood for training set")
#                             )
#         draw(PDF(filename, 10inch, 6inch), LikelihoodPlot)
#     else
#         # Draw plot if no filename given
#        plot(  
#                                 layer(x=1:length(pseudo),y=pseudo,Geom.point, Geom.line),
#                                 layer(x=1:length(tap),y=tap, Geom.point, Geom.line, Theme(default_color=colorant"green")),
#                                 Guide.xlabel("epochs"),Guide.ylabel("Likelihood"),Guide.title("Evolution of likelihood for training set")
#                             )
#     end
# end
