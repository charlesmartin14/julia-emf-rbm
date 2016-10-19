
using Compat

abstract Net

immutable DBN <: Net
    layers::Vector{RBM}
    layernames::Vector{AbstractString}
end

DBN{T<:@compat(Tuple{AbstractString,RBM})}(namedlayers::Vector{T}) =
    DBN(map(p -> p[2], namedlayers), map(p -> p[1], namedlayers))
    

immutable DAE <: Net
    layers::Vector{RBM}
    layernames::Vector{AbstractString}
end


function Base.show(io::IO, net::Net)
    nettype = string(typeof(net))
    layer_str = join(net.layernames, ",")
    print(io, "$nettype($layer_str)")
end

# DBN fields may change in the future, so it's worth to work through accessors
getname(net::Net, k::Int) = net.layernames[k]
getmodel(net::Net, k::Int) = net.layers[k]
function getmodel(net::Net, name::AbstractString)
    k = findfirst(lname -> lname == name, net.layernames)
    return (k != 0) ? net.layers[k] : error("No layer named '$name'")
end

# short syntax for accessing stored RBMs
getindex(net::Net, k::Int) = getmodel(net, k)
getindex(net::Net, name::AbstractString) = getmodel(net, name)
Base.length(net::Net) = length(net.layers)
Base.endof(net::Net) = length(net)


function mh_at_layer(net::Net, batch::Array{Float64, 2}, layer::Int)
    hiddens = Array(Array{Float64, 2}, layer)
    hiddens[1] = hid_means(net[1], batch)
    for k=2:layer
        hiddens[k] = hid_means(net[k], hiddens[k-1])
    end
    hiddens[end]
end


function transform(net::Net, X::Mat{Float64})
    return mh_at_layer(net, X, length(net))
end


function fit(dbn::DBN, X::Mat{Float64};
             lr=0.1, n_iter=10, batch_size=100, n_gibbs=1)
    @assert minimum(X) >= 0 && maximum(X) <= 1
    n_samples = size(X,2)
    n_batches = round(Int, ceil(n_samples / batch_size))
    for k = 1:length(dbn.layers)
        w_buf = zeros(size(dbn[k].W))
        for itr=1:n_iter
            info("Layer $(k), iteration $(itr)")
            for i=1:n_batches
                batch = X[:, ((i-1)*batch_size + 1):min(i*batch_size, end)]
                input = k == 1 ? batch : mh_at_layer(dbn, batch, k-1)
                fit_batch!(dbn[k], input, buf=w_buf, n_gibbs=n_gibbs)
            end
        end
    end
end


function invert(rbm::RBM)
    irbm = deepcopy(rbm)
    irbm.W = rbm.W'
    irbm.vbias = rbm.hbias
    irbm.hbias = rbm.vbias
    return irbm
end


function unroll(dbn::DBN)
    n = length(dbn)
    layers = Array(RBM, 2n)
    layernames = Array(AbstractString, 2n)
    layers[1:n] = dbn.layers
    layernames[1:n] = dbn.layernames
    for i=1:n
        layers[n+i] = invert(dbn[n-i+1])
        layernames[n+i] = getname(dbn, n-i+1) * "_inv"
    end
    return DAE(layers, layernames)
end

