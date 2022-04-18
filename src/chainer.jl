using Flux


get_output_shape(layer::Dense) = size(layer.weight)[1]

function conv_output(in_size, padding, dilation, kernel, stride)
    floor((in_size + 2 * padding - dilation * (kernel - 1) - 1) / stride)
end

function to_list(param, ndims_ker)
    p_type = typeof(param)
    if p_type == Tuple
        @assert(ndims(param) == ndims_ker)
    elseif p_type == Integer
        param = [param for _ = 1 : ndims_ker]
    else
        error("Parameters should be either Int or Tuple, not $p_type")
    end
end

function get_output_shape(layer::Conv, input_size::Tuple)
    ind = lastindex(size(layer.weight))
    kernel = size(layer.weight)[1:ind-2]
    out_ch = last(size(layer.weight))
    dilation = layer.dilation
    pad = layer.pad
    stride = layer.stride
    params = zip(input_size, pad, dilation, kernel, stride)
    vcat([conv_output(in, p, d, k, s) for (in, p, d, k, s) in params], [out_ch])
end

