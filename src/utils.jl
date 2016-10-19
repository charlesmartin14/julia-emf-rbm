using Devectorize

if !isdefined(:__EXPRESSION_HASHES__)
    __EXPRESSION_HASHES__ = Set{UInt64}()
end

macro runonce(expr)
    h = hash(expr)
    return esc(quote
        if !in($h, __EXPRESSION_HASHES__)
            push!(__EXPRESSION_HASHES__, $h)
            $expr
        end
    end)
end

"""
  Defines the type `Mat{T}` as a convenience wrapper around the 
  `AbstractArray{T,2}` type. 
"""
typealias Mat{T} AbstractArray{T, 2}

"""
  Defines the type `Vec{T}` as a convenience wrapper around the 
  `AbstractArray{T,1}` type. 
"""
typealias Vec{T} AbstractArray{T, 1}

""" 
  # Boltzmann.normalize_samples (utils.jl)
  ## Function Calls
    `normalize_samples(X::Mat{Float64})`
  
  ## Description
    Given a matrix, `X`, assume that each column represents a different
    data sample and that each row represents a different feature. In this
    case, `normalize_samples` will normalize each individual sample to
    the range `[0,1]` according to the minimum and maximum features in the
    sample.

  ## Returns
    1. `::Mat{Float64}`, The normalized dataset.

  ### See also...
    `normalize_samples!`
"""
function normalize_samples(x::Mat{Float64})
    samples = size(x,2)
    y = zeros(x)    

    for i=1:samples
      s = x[:,i]
      mins = minimum(s)
      maxs = maximum(s)
      rans = maxs-mins
      y[:,i] = (s-mins)/rans
    end

    return y
end

"""
  # Boltzmann.normalize_samples! (utils.jl)
  ## Function Calls
    `normalize_samples!(x::Mat{Float64})`
  
  ## Description
    Given a matrix, `x`, assume that each column represents a different
    data sample and that each row represents a different feature. In this
    case, `normalize_samples` will normalize each individual sample to
    the range `[0,1]` according to the minimum and maximum features in the
    sample. 

  ## Returns
    Nothing, modifies `x` in place.

  ### See also...
    `normalize_samples`
"""
function normalize_samples!(x::Mat{Float64})
    samples = size(x,2)

    for i=1:samples
      s = x[:,i]
      mins = minimum(s)
      maxs = maximum(s)
      rans = maxs-mins
      x[:,i] = (s-mins)/rans
    end
end

"""
  # Boltzmann.normalize (utils.jl)
  ## Function Calls
    `normalize(x::Mat{Float64})`
    `normalize(x::Vec{Float64})`
  
  ## Description
    Given an array, `x`, normalize the entire array to the range
    `[0,1]` according to the maximum and miniumum values of the array.

  ## Returns
    1. `::Mat{Float64}` *or* `::Vec{Float64}` depending on the input.

  ### See also...
    `normalize!`
"""
function normalize(x::Mat{Float64})
    y = zeros(x)
    minx = minimum(x)
    maxx = maximum(x)
    ranx = maxx-minx

    # We can index a two-dimension array as if it were a vector.
    @simd for i=1:length(x)
      @inbounds y[i] = (x[i]-minx) / ranx
    end

    return y
end
function normalize(x::Vec{Float64})
    y = zeros(x)
    minx = minimum(x)
    maxx = maximum(x)
    ranx = maxx-minx

    @simd for i=1:length(x)
      @inbounds y[i] = (x[i]-minx) / ranx
    end

    return y
end

"""
  # Boltzmann.normalize! (utils.jl)
  ## Function Calls
    `normalize!(x::Mat{Float64})`
    `normalize!(x::Vec{Float64})`
  
  ## Description
    Given an array, `x`, normalize the entire array to the range
    `[0,1]` according to the maximum and miniumum values of the array.

  ## Returns
    Nothing. Modifies `x` in place.

  ### See also...
    `normalize`
"""
function normalize!(x::Mat{Float64})
    minx = minimum(x)
    maxx = maximum(x)
    ranx = maxx-minx

    @simd for i=1:length(x)
      @inbounds x[i] = (x[i]-minx) / ranx
    end
end
function normalize!(x::Vec{Float64})
    minx = minimum(x)
    maxx = maximum(x)
    ranx = maxx-minx

    @simd for i=1:length(x)
      @inbounds x[i] = (x[i]-minx) / ranx
    end
end


"""
  # Boltzmann.binarize (utils.jl)
  ## Function Calls
    `binarize(x::Mat{Float64}[,threshold=0.0])`
    `binarize(x::Vec{Float64}[,threshold=0.0])`
  
  ## Description
    Given an array, `x`, assign each element of the array to 
    either `0` or `1` depending on specified value of `threshold`.
    This is done according to the following rule...
    ```
        if element <= threshold: element = 0
        if element >  threshold: element = 1
    ```

  ## Returns
    1. `::Mat{Float64}` *or* `::Vec{Float64}`, depending on input.

  ### See also...
    `binarize!`
"""
function binarize(x::Mat{Float64};threshold=0.0)
  s = copy(x)
  @simd for i=1:length(x)
    @inbounds s[i] = x[i] > threshold ? 1.0 : 0.0
  end
  return s
end
function binarize(x::Vec{Float64};threshold=0.0)
  s = copy(x)
  @simd for i=1:length(x)
    @inbounds s[i] = x[i] > threshold ? 1.0 : 0.0
  end
  return s
end

"""
  # Boltzmann.binarize! (utils.jl)
  ## Function Calls
    `binarize!(x::Mat{Float64}[,threshold=0.0])`
    `binarize!(x::Vec{Float64}[,threshold=0.0])`
  
  ## Description
    Given an array, `x`, assign each element of the array to 
    either `0` or `1` depending on specified value of `threshold`.
    This is done according to the following rule...
    ```
        if element <= threshold: element = 0
        if element >  threshold: element = 1
    ```

  ## Returns
    Nothing. Modifies `x` in place.

  ### See also...
    `binarize`
"""
function binarize!(x::Mat{Float64};threshold=0.0)
  @simd for i=1:length(x)
    @inbounds x[i] = x[i] > threshold ? 1.0 : 0.0
  end
end
function binarize!(x::Vec{Float64};threshold=0.0)
  @simd for i=1:length(x)
    @inbounds x[i] = x[i] > threshold ? 1.0 : 0.0
  end
end

"""
  # Boltzmann.logsig (utils.jl)
  ## Function Calls
    `logsig(x::Mat{Float64})`
    `logsig(x::Vec{Float64})`
  
  ## Description
    Perform the logistic sigmoid element-by-element for the entries
    of `x`. Here, the logistic sigmoid is defined as
                      `y  =  1 / (1+e^-x)`.
    
  ## Returns
    1. `::Mat{Float64}` *or* `::Vec{Float64}` (depending on input).

  ### See also...
    `logsig!`
"""
function logsig(x::Mat{Float64})
    @devec s = 1 ./ (1 + exp(-x))
    return s
end
function logsig(x::Vec{Float64})
    @devec s = 1 ./ (1 + exp(-x))
    return s
end


"""
  # Boltzmann.logsig! (utils.jl)
  ## Function Calls
    `logsig!(x::Mat{Float64})`
    `logsig!(x::Vec{Float64})`
  
  ## Description
    Perform the logistic sigmoid element-by-element for the entries
    of `x`. Here, the logistic sigmoid is defined as
                      `y  =  1 / (1+e^-x)`.
    
  ## Returns
    Nothing. Modifies `x` in place.

  ### See also...
    `logsig`
"""
function logsig!(x::Mat{Float64})
  # We would like to have used `@devec`, here, but this macro by default
  # makes a new instantiation in memory and thus the result is distinct 
  # from the original.
  @simd for i=1:length(x)
    @inbounds x[i] = 1 ./ (1 + exp(-x[i]))
  end
end
function logsig!(x::Vec{Float64})
  @simd for i=1:length(x)
    @inbounds x[i] = 1 ./ (1 + exp(-x[i]))
  end
end

"""
  # Boltzmann.random_samplse (utils.jl)
  ## Function Calls
    `random_columns(x::Mat{Float64})`
  ## Description
    Select a random permutation of the columsn of the specified 
    matrix, returning the permuted sub-matrix as well as the 
    column indices selected from the original matrix.

  ## Returns
    1. `::Mat{Float64}`, the random sub-matrix.
    2. `::Vec{Int}`, the array of selected column indices
"""
function random_columns(x::Mat{Float64},n_select::Int)
    n_features = size(x,1)
    n_columns = size(x,2)
    perm = shuffle!(collect(1:n_columns))[1:n_select]
    y = x[:,perm]

    return y, perm
end