
using HDF5


function save_params(file::HDF5File, rbm::RBM, name::AbstractString)
    write(file, "$(name)___weight", rbm.W')
    write(file, "$(name)___vbias", rbm.vbias)
    write(file, "$(name)___bias", rbm.hbias)
end

function save_params(file::HDF5File, rbm::RBM, name::AbstractString)
    write(file, "$(name)___weight", rbm.W')
    write(file, "$(name)___vbias", rbm.vbias)
    write(file, "$(name)___bias", rbm.hbias)
end

function save_params(filename::AbstractString,rbm::RBM,name::AbstractString)
    h5open(filename,"w") do file
        save_params(file,rbm,name)
    end
end

function append_params(filename::AbstractString,rbm::RBM,name::AbstractString)
    h5open(filename,"r+") do file
        save_params(file,rbm,name)
    end
end

function load_params(file::HDF5File, rbm::RBM, name::AbstractString)
    rbm.W = read(file, "$(name)___weight")'
    rbm.vbias = read(file, "$(name)___vbias")
    rbm.hbias = read(file, "$(name)___bias")
end

function save_params(file::HDF5File, net::Net)
    for i=1:length(net)
        save_params(file, net[i], getname(net, i))
    end
end
save_params(path::AbstractString, net::Net) = h5open(path, "w") do h5
    save_params(h5, net)
end


function load_params(file, net::Net)
    for i=1:length(net)
        load_params(file, net[i], getname(net, i))
    end
end
load_params(path::AbstractString, net::Net) = h5open(path) do h5
    load_params(h5, net)
end

function SaveMonitorHDF5(mon::Monitor,filename::AbstractString)
    h5open(filename , "w") do file
        # write(file, "LastIndex", mon.LastIndex)
        # write(file, "UseValidation", mon.UseValidation)
        write(file, "MonitorEvery", mon.MonitorEvery)
        # write(file, "MonitorVisual", mon.MonitorVisual)
        # write(file, "MonitorText", mon.MonitorText)
        write(file, "Epochs", mon.Epochs)
        write(file, "LearnRate", mon.LearnRate)
        write(file, "Momentum", mon.Momentum)
        write(file, "PseudoLikelihood", mon.PseudoLikelihood)
        write(file, "TAPLikelihood", mon.TAPLikelihood)
        write(file, "ValidationPseudoLikelihood", mon.ValidationPseudoLikelihood)
        write(file, "ValidationTAPLikelihood", mon.ValidationTAPLikelihood)
        write(file, "ReconError", mon.ReconError)
        write(file, "ValidationReconError", mon.ValidationReconError)
        write(file, "BatchTime_µs", mon.BatchTime_µs)
        # write(file, "FigureHandle", mon.FigureHandle)
    end 
end  