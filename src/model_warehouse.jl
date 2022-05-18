using BSON: @save, @load

function makedirs(path)
    if not isdir(dir)
        @info "Creating $dir"
        mkpath(dir)
    end
end

function save(path::AbstractString, model)
    dir = dirname(path)
    makedirs(dir)
    @save path model
end

function load(weights_path::AbstractString)
    @load weights_path, model
end