struct VAE
    encoder_μ::Vector
    encoder_logvar::Vector
    latent::Vector
    decoder::Vector
    beta::Number
end

function VAE(paths::Vector, metadata::Dict)
    input_size = metadata[:data][:input_size]
    e, out_size, ll = to_chain(paths, "encoder", input_size; last_layer_info=true)
    l, out_size, ll = to_chain(paths, "latent", out_size; prev_f=ll; last_layer_info=true)
    d, out_size, ll = to_chain(paths, "decoder", out_size; prev_f=ll;
                               last_layer_info=true, out_size=input_size)
    return VAE(e, e, l, d, metadata[:loss][:beta])
end

function(m::VAE)(x)
    μ = m.encoder_μ(x)
    logvar = m.encoder_logvar(x)
    σ = exp(0.5 * logvar)
    ϵ = randn(size(σ))
    z = ϵ * σ + μ
    x̂ = m.decoder(z)
    return x̂, μ, σ
end