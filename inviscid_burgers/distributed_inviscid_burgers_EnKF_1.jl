# -*- coding: utf-8 -*-
using ClusterManagers, Distributed

# +
# addprocs(40)

# +
# Add in the cores allocated by the scheduler as workers
@show nworkers()

addprocs(SlurmManager(parse(Int, ENV["SLURM_NTASKS"]) - 1))

print("Added workers: ")

@show nworkers()
# -

pwd()

@everywhere begin
    using Pkg
    path_to_library = "/home/gridsan/mleprovost/julia/HierarchicalDA.jl"
    path_to_save = "/home/gridsan/mleprovost/julia/HierarchicalDA.jl/notebooks/inviscid_burgers/data/"

    Pkg.activate(path_to_library)
end

# +
@everywhere begin
    using LinearAlgebra
    using HierarchicalDA
    import TransportBasedInference: StateSpace, AdditiveInflation, Model, SyntheticData,
        Localization, MultiplicativeInflation, MultiAddInflation,
        periodicmetric!, rmse, spread, metric_hist
    using Statistics
    using Distributions
    using JLD
    using FileIO
    using ProgressMeter
    using SparseArrays


    polydeg = 3
    Ncells = 100

    Nx = (polydeg + 1) * Ncells
    Δ = 20 # Space between observations
    Ny = ceil(Int64, Nx / Δ)

    # Define Trixi system for inviscid Burgers equation
    sys_burgers = setup_burgers(polydeg, Ncells)
    xgrid = vec(sys_burgers.mesh.md.xq)

    order_PA = 3
    Ns = Nx - 2 * ceil(Int64, order_PA / 2)

    PA = PolyAnnil(xgrid, order_PA; istruncated=true)

    @assert size(PA.P) == (Ns, Nx)

    S = LinearMaps.FunctionMap{Float64,true}((s, x) -> mul!(s, PA.P, x), (x, s) -> mul!(x, PA.P', s),
        Ns, Nx; issymmetric=false, isposdef=false)

    xs = xgrid[ceil(Int64, order_PA / 2)+1:end-ceil(Int64, order_PA / 2)]


    Δtdyn = 0.002
    Δtobs = 0.002

    # Set up time
    t0 = 0.0
    Tf = 2000
    tf = Tf * Δtobs

    Tmetric = 1000
    tmetric = Tmetric * Δtobs

    π0 = MvNormal(zeros(Nx), Matrix(1.0 * I, Nx, Nx))

    σx_true = 1e-6
    σx = 0.01


    σy = 0.2

    ϵx_true = AdditiveInflation(Nx, zeros(Nx), σx_true)
    ϵx = AdditiveInflation(Nx, zeros(Nx), σx)

    ϵy = AdditiveInflation(Ny, zeros(Ny), σy)


    h(x, t) = x[1:Δ:end]
    H = LinearMap(sparse(Matrix(1.0 * I, Nx, Nx)[1:Δ:end, :]))
    F = StateSpace(x -> x, h)

    model = Model(Nx, Ny, Δtdyn, Δtobs, ϵx_true, ϵy, π0, 0, 0, 0, F)

    # Define function class for the initial condition
    αk = 0.7
    f0 = SmoothPeriodic(xgrid, αk; L=2.0)
end

x0 = vec(1 / 2 .+ 0.5 * sin.(3 * π * sys_burgers.mesh.md.xq));

data = generate_data_trixi(model, x0, Tf, sys_burgers)

save(path_to_save * "benchmark_inviscid_burgers_enkf_1.jld", "data", data)
@show "data saved"

@everywhere begin
    data = load(path_to_save * "benchmark_inviscid_burgers_enkf_1.jld", "data")

    data = SyntheticData(data.tt, data.Δt, data.x0, data.xt, data.yt)

    @show "data loaded"

    Nrun = 1

    struct Light_Metric
        Ne::Int64
        RMSE_MED::Float64
        RMSE_MEAN::Float64
        SPREAD_MED::Float64
        SPREAD_MEAN::Float64
        # COV_MEAN::Float64
        # CRPS_MEAN::Float64
    end

    function custom_output_metrics(data::SyntheticData, model::Model, Tbegin::Int64, Tend::Int64, statehist::Array{Array{Float64,2},1})
        Ne = size(statehist[1], 2)

        # Post_process compute the statistics (mean, median, and
        # standard deviation) of the RMSE, spread and coverage
        # probability over (J-T_BurnIn) assimilation steps.

        # enshist contains the initial condition, so one more element
        idx_xt = Tbegin+1:Tend

        idx_ens = Tbegin+2:Tend+1

        # Compute root mean square error statistics
        Rmse, Rmse_med, Rmse_mean, Rmse_std = metric_hist(rmse, data.xt[:, idx_xt], statehist[idx_ens])
        # Compute ensemble spread statistics
        Spread, Spread_med, Spread_mean, Spread_std = metric_hist(spread, statehist[idx_ens])


        metric = Light_Metric(Ne, Rmse_med, Rmse_mean, Spread_med, Spread_mean)
        return metric
    end



    ## Selecion of hyper-prior parameters
    GGidx = 4
    # power parameter
    r_range = [1.0, 0.5, -0.5, -1.0]
    r = r_range[GGidx] # select parameter 
    # shape parameter
    β_range = [1.501, 3.0918, 2.0165, 1.0017]
    β = β_range[GGidx] # shape parameter
    # rate parameters 
    ϑ_range = [5 * 10^(-2), 5.9323 * 10^(-3), 1.2583 * 10^(-3), 1.2308 * 10^(-4)]
    ϑ = 5 * 10^(-3)#ϑ_range[GGidx]
    dist = GeneralizedGamma(r, β, ϑ)


    Cθ = LinearMap(Diagonal(deepcopy(θinit)))
    Cϵ = LinearMap(ϵy.Σ)
    CX = LinearMap(Diagonal(1.0 .+ rand(Nx)))
    sys_ys = ObsConstraintSystem(H, S, Cθ, Cϵ, CX)

    sys_y = ObsSystem(H, Cϵ, CX)

    yidx = 1:Δ:Nx
    idx = vcat(collect(1:length(yidx))', collect(yidx)')

    # @assert length(yidx) == Ny

    # # Create Localization structure
    Gxx(i, j) = periodicmetric!(i, j, Nx)
    Gxy(i, j) = periodicmetric!(i, yidx[j], Nx)
    Gyy(i, j) = periodicmetric!(yidx[i], yidx[j], Nx)

    Ne_array = [30, 40, 50, 60, 80, 100]
    β_array = collect(1.0:0.01:1.05)
    L_array = collect(1:1:20)

end

for Ne in Ne_array
    X = zeros(model.Ny + model.Nx, Ne)

    # Generate the initial conditions for the state.
    for i = 1:Ne
        regenerate!(f0)
        X[Ny+1:Ny+Nx, i] = f0.(xgrid)
    end

    save(path_to_save * "initial_ensemble_inviscid_burgers_enkf_Ne_" *
         string(ceil(Int64, Ne)) * "_1.jld", "X", X)
end

@everywhere begin
    # Create iterator over the three collections: β ⊗ Ne ⊗ L
    # We put the ensemble size last as we don't want all the job with the largest ensemble size Ne to run at the same time
    βNeL_tensor = Iterators.product(β_array, Ne_array, L_array)
    βNeL_array = reshape(collect(βNeL_tensor), length(βNeL_tensor))

    @show size(βNeL_array)

    # Track progress
    prog = Progress(length(βNeL_array))
end
# -

@sync @distributed for idx in βNeL_array

    β, Ne, Lrad = idx

    @show Ne, β, Lrad

    ϵxβ = MultiAddInflation(Nx, β, zeros(Nx), σx)
    Loc = Localization(Lrad, Gxx, Gxy, Gxx)

    #. Walk-around to avoid X to be a local variable within the lock/unlock pattern
    X = load(path_to_save * "initial_ensemble_inviscid_burgers_enkf_Ne_" * string(ceil(Int64, Ne)) * "_1.jld", "X")

    enkf = LocEnKF(Ne, ϵy, sys_y, Loc, Δtdyn, Δtobs)

    henkf = HLocEnKF(Ne, ϵy, sys_ys, Loc, dist, deepcopy(θinit), Δtdyn, Δtobs)

    X_enkf = seqassim_trixi(data, Tf, ϵxβ, enkf, deepcopy(X), model.Ny, model.Nx, t0, sys_burgers)

    metric_enkf = custom_output_metrics(data, model, Tmetric, Tf, X_enkf)

    save(path_to_save * "benchmark_inviscid_burgers_enkf" *
         "_Ne_" * string(ceil(Int64, Ne)) *
         "_beta_" * string(ceil(Int64, 100 * β)) *
         "_L_" * string(ceil(Int64, Lrad)) * "_1.jld",
        "metric", metric_enkf)

    X_henkf, θ_henkf = seqassim_trixi(data, Tf, ϵxβ, henkf, deepcopy(X), model.Ny, model.Nx, t0, sys_burgers)

    metric_henkf = custom_output_metrics(data, model, Tmetric, Tf, X_henkf)

    save(path_to_save * "benchmark_inviscid_burgers_henkf" *
         "_Ne_" * string(ceil(Int64, Ne)) *
         "_beta_" * string(ceil(Int64, 100 * β)) *
         "_L_" * string(ceil(Int64, Lrad)) * "_1.jld",
        "metric", metric_henkf)
end




