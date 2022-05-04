using Flux
using Primes

function dense_to_conv(x::AbstractArray)
    factors = []

    for (p, k) in factor(size(x)[1])
        factors = vcat(factors, [p for _ in range(1, k)])
    end
    if length(factors) > 2
        new_size = (factors[1], factors[2], prod(factors[3:end]))
    else
        new_size = (factors[1], factors[2], 1)
    end
    return reshape(x, new_size..., size(x)[end])
  end