# Streamliner

## Local installation instructions

The environment in `app` was generated as follows.

From the top-level folder, start julia with `julia --project=app`. Then, in the Pkg REPL, do

```julia
(app) pkg> add https://github.com/LimenResearch/ParametricMachinesDemos.jl
(app) pkg> dev .
```

The first line adds the unregistered dependency ParametricMachinesDemos, whereas the second line develops the local folder. Note `dev` instead of `add` to ensure that local changes in Streamliner are picked up in the `app` environment.

For future uses, once the Manifest is present, it should be sufficient to `instantiate` the `app` environment.
