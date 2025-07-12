# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: jl:percent
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Julia 1.11.5
#     language: julia
#     name: julia-1.11
# ---

# %%
using Pkg
proj_path = joinpath(@__DIR__, "..", "..")
Pkg.activate(proj_path)

# %%
using TransportBasedInference2
using HierarchicalDA
using LinearAlgebra
using OrdinaryDiffEq
using Trixi
using FFTW
using Distributions
using Statistics
using SparseArrays
using LinearMaps
using CairoMakie
using JLD2
using Dates
using Random

# %%
# Logistics
make_figs = false
my_theme = theme_latexfonts()
update_theme!(my_theme, linewidth=3.)
random_seed = rand(UInt)
data_path = joinpath(@__DIR__, "data")

# %%
# Problem setup params
polydeg = 7 # Order in space
Ncells = 100 # Number of DG cells
delta_y = 80 # Spatial frequency of observation. Not regularly spaced
delta_t_dyn = 0.005 # Timestep for PDE dynamics
delta_t_obs = 0.025 # Amount of time between each observation

sigma_x_data = 1e-3 # Noise in the state dynamics (i.e., the PDE solution itself)
sigma_y = 0.15 # Noise in the state observation (i.e., what the "sensors" record)

t0, tf = 0.0, 1.0 # Start and end time

# %%
# Important parameters for data assimilation
Ne = 40 # Ensemble size
Lrad = 10 # Localization radius
sigma_x_filter = 0.05 # State noise
beta_infl = 1.02 # Inflation param
alpha_k_f0, L_f0 = 0.7, 1.0 # Parameters for initial condition

# %%
# GSBL Hyperparams
order_PA = 3 # Poly annihilator order
Niter = 5
theta_init = 1.

hyperprior_idx = 3

# %%
# Assign any given arguments
isdefined(Main, :IJulia) || length(ARGS) > 0 && @info "Given arguments: " ARGS
isdefined(Main, :IJulia) || for arg in ARGS
    key, val = split(arg, "=")
    sym_key = Symbol(key)
    val_T = @eval($sym_key)
    pre_val_type = typeof(val_T)
    is_err = false
    try
        val_T = pre_val_type == String ? string(val) : parse(pre_val_type, val)
    catch e
        is_err = true
    end
    if is_err
        @error "Could not parse value in $key=$val, type $(typeof(val)), to type $pre_val_type"
        val_T = parse(pre_val_type, val)
    end
    @eval($sym_key = $val_T)
end

# %%
Random.seed!(random_seed);

# %%
Nx = (polydeg + 1) * Ncells
Ny = ceil(Int64, Nx / delta_y)
Tf = round(Int, tf / delta_t_obs)

# Define Trixi system for inviscid Burgers equation
sys_burgers = setup_burgers(polydeg, Ncells);

xgrid = vec(sys_burgers.mesh.md.xq);

# %%
π0 = MvNormal(zeros(Nx), Matrix(1.0 * I, Nx, Nx))

# %%
ϵx_data = AdditiveInflation(Nx, zeros(Nx), sigma_x_data)
ϵy = AdditiveInflation(Ny, zeros(Ny), sigma_y);

# %%
h(x, t) = x[1:delta_y:end]
H = LinearMap(sparse(Matrix(1.0 * I, Nx, Nx)[1:delta_y:end, :]))
F = StateSpace(x -> x, h)

# %%
model = Model(Nx, Ny, delta_t_dyn, delta_t_obs, ϵx_data, ϵy, π0, 0, 0, 0, F);

# %%
x0 = vec(1 / 2 .+ 0.5 * sin.(3 * π * sys_burgers.mesh.md.xq));

# %%
@info "Generating data..."
data = generate_data_trixi(model, x0, Tf, sys_burgers)

# %%
# CALCULATE ENTROPY
mesh_wts = vec(sys_burgers.mesh.md.wJq)
ents = [mesh_wts'Trixi.entropy.(x, (sys_burgers.equations,)) for x in eachcol(data.xt)]

# %%
lines(ents)

# %%
make_figs && heatmap(delta_t_obs * (1:Tf), xgrid, data.xt', axis=(; xlabel=L"t", ylabel=L"x", title=L"Solution of inviscid burgers, $u(x,t)$"))

# %%
make_figs && with_theme(my_theme) do
    fig = Figure()
    ax = Axis(fig[1, 1])

    lines!(ax, xgrid, data.xt[:, 1])
    lines!(ax, xgrid, data.xt[:, end])
    scatter!(ax, xgrid[1:delta_y:end], data.yt[:, end])

    fig
end

## Selecion of hyper-prior parameters
# power parameter
r_range = [1.0, 0.5, -0.5, -1.0];
r_GSBL = r_range[hyperprior_idx] # select parameter 
# shape parameter
beta_range = [1.501, 3.0918, 2.0165, 1.0017];
beta_GSBL = beta_range[hyperprior_idx] # shape parameter
# rate parameters 
ϑ_range = [5 * 10^(-2), 5.9323 * 10^(-3), 1.2583 * 10^(-3), 1.2308 * 10^(-4)];
ϑ_GSBL = ϑ_range[hyperprior_idx]

dist = GeneralizedGamma(r_GSBL, beta_GSBL, ϑ_GSBL);

# %%
# Define function class for the initial condition
f0 = SmoothPeriodic(xgrid, alpha_k_f0; L=L_f0);
X = zeros(model.Ny + model.Nx, Ne)

for i = 1:Ne
    regenerate!(f0)
    X[Ny+1:Ny+Nx, i] = f0.(xgrid) / 3 .+ 0.5#initial_condition(alpha_k, Δx, Nx)
end

# %%
Cϵ = LinearMap(ϵy.σ)
CX = LinearMap(Diagonal(1.0 .+ rand(Nx)))
sys_y = ObsSystem(H, Cϵ, CX);

# %%
yidx = 1:delta_y:Nx
idx = vcat(collect(1:length(yidx))', collect(yidx)')

# Create Localization structure
Gxx(i, j) = periodicmetric!(i, j, Nx)
Gxy(i, j) = periodicmetric!(i, yidx[j], Nx)
Gyy(i, j) = periodicmetric!(yidx[i], yidx[j], Nx)

Loc = Localization(Nx, Lrad, Gxx, is_sparse=true)
ϵxbeta_filter = MultiAddInflation(Nx, beta_infl, zeros(Nx), sigma_x_filter)

# %%
make_figs && with_theme(my_theme) do
    fig = Figure()

    ax = Axis(fig[1, 1])

    for i = 1:10
        lines!(ax, xgrid, X[Ny+1:Ny+Nx, i])
    end
    # lines!(xgrid, mean(X[Ny+1:Ny+Nx, :]; dims=2)[:, 1], linewidth=5, linestyle=:dash)

    lines!(ax, xgrid, x0, linewidth=10)

    fig
end

# %%
locenkf = LocEnKF(Ne, ϵy, sys_y, Loc, delta_t_dyn, delta_t_obs)

# %%
@info "Performing EnKF..."
X_locenkf = seqassim_trixi(data, Tf, ϵxbeta_filter, locenkf, deepcopy(X), model.Ny, model.Nx, t0, sys_burgers);

# %%
PA_offset = ceil(Int64, order_PA / 2)
Ns = Nx - 2 * PA_offset

PA = PolyAnnil(xgrid, order_PA; istruncated=true)
@assert size(PA.P) == (Ns, Nx)

S = LinearMaps.FunctionMap{Float64,true}((s, x) -> mul!(s, PA.P, x), (x, s) -> mul!(x, PA.P', s),
    Ns, Nx; issymmetric=false, isposdef=false)

theta_init_vec = fill(theta_init, Ns)
Cθ = LinearMap(Diagonal(theta_init_vec))
sys_ys = ObsConstraintSystem(H, S, Cθ, Cϵ, CX);

# %%
hlocenkf = HLocEnKF(Ne, ϵy, sys_ys, Loc, dist, theta_init_vec, delta_t_dyn, delta_t_obs; Niter, θinit=theta_init)

# %%
@info "Performing GSBL EnKF..."
X_hlocenkf, θ_hlocenkf = seqassim_trixi(data, Tf, ϵxbeta_filter, hlocenkf, deepcopy(X), model.Ny, model.Nx, t0, sys_burgers);

# %%
mesh_weights = vec(sys_burgers.mesh.md.wJq)
weighted_norm1 = (x, w) -> sum(dim_idx -> w[dim_idx] * abs(x[dim_idx]), eachindex(x, w))
weighted_norm2 = (x, w) -> sqrt(sum(dim_idx -> w[dim_idx] * abs2(x[dim_idx]), eachindex(x, w)))
rel_norms1 = map(Base.Fix2(weighted_norm1, mesh_weights), eachcol(data.xt))
rel_norms2 = map(Base.Fix2(weighted_norm2, mesh_weights), eachcol(data.xt))
get_errs = (X, metric) -> map(j -> CRPS(X[j+1], @view(data.xt[:, j]), metric, mesh_weights), axes(data.xt, 2))
get_Lp = (err, rel_norms, prop::Symbol) -> mean(er -> get_property(er[1], prop) / er[2], zip(err, rel_norms))
metrics_locenkf, metrics_hlocenkf = Dict{Symbol,Any}(), Dict{Symbol,Any}()
calc_moments = (ensemble, moment) -> weight_sum_reduction.(eachcol(ensemble), (moment,), (mesh_weights,))
burgers_entropy = u -> entropy(u, sys_burgers.equations)
mass_true, entropy_true = map(f -> calc_moments(data.xt, f), [abs, burgers_entropy])

# %%
for alg in ["locenkf", "hlocenkf"]
    # Error metrics
    metric_dict = @eval($Symbol("metrics_$alg"))
    X = @eval($Symbol("X_$alg"))
    for which_norm in [1, 2]
        norm = Symbol("norm$which_norm")
        errs = get_errs(X, norm)
        rel_norms = @eval($Symbol("rel_norms$which_norm"))
        for metric in [:rmse, :crps]
            metric_sym = Symbol("$metric$norm_$alg")
            metric_dict[metric_sym] = get_Lp(errs, rel_norms, metric)
        end
    end

    # Mass and Entropy
    mass_alg, entropy_alg = [reduce(hcat, calc_moments(x, f) for x in X) for f in [abs, burgers_entropy]]
    metric_dict[:mass] = mass_alg
    metric_dict[:entropy] = entropy_alg
end

# %%
jldopen(joinpath(data_path, "burgers_" * string(now()) * ".jld2"), "w") do file
    data_group = JLD2.Group(file, "data")
    for property in propertynames(data)
        data_group[string(property)] = getproperty(data, property)
    end

    data_param_group = JLD2.Group(file, "data_parameters")
    for data_param in [:random_seed, :polydeg, :Ncells, :delta_t_dyn, :delta_t_obs, :sigma_x_data, :sigma_y, :t0, :tf]
        data_param_group[string(data_param)] = @eval($data_param)
    end

    filter_param_group = JLD2.Group(file, "filter_parameters")
    for filter_param in [:Ne, :Lrad, :sigma_x_filter, :beta_infl, :alpha_k_f0, :L_f0]
        filter_param_group[string(filter_param)] = @eval($filter_param)
    end

    GSBL_param_group = JLD2.Group(file, "GSBL_parameters")
    for GSBL_param in [:order_PA, :Niter, :theta_init, :dist]
        GSBL_param_group[string(GSBL_param)] = @eval($GSBL_param)
    end

    filter_group = JLD2.Group(file, "filters")
    metric_group = JLD2.Group(file, "metrics")
    metric_group[:true_mass] = mass_true
    metric_group[:true_entropy] = entropy_true
    for alg in ["locenkf", "hlocenkf"]
        metric_subgroup = JLD2.Group(metric_group, alg)
        metric_dict = @eval($Symbol("metrics_$alg"))
        for key in keys(metric_dict)
            metric_subgroup[string(key)] = metric_dict[key]
        end
        X_alg = Symbol("X_" * alg)
        filter_group[string(X_alg)] = @eval($X_alg)
    end
end;

# %%
make_figs && with_theme(my_theme) do
    t_start = 1
    tsnap = Observable(t_start)
    x_tsnap = @lift(data.xt[:, $tsnap])
    y_tsnap = @lift(data.yt[:, $tsnap])
    ut = t -> map(x -> x[], vec(data.xt[:, round(Int, t / delta_t_obs)]))
    ys = @lift(ut(($tsnap) * delta_t_obs))
    X_hlocenkf_tsnap = @lift(vec(mean(X_hlocenkf[$tsnap+1]; dims=2)))
    X_ens_tsnap = [@lift(X_hlocenkf[$tsnap+1][:, j]) for j in 1:Ne]
    theta_tsnap = @lift(θ_hlocenkf[$tsnap+1])
    cols = Makie.wong_colors()

    fig = Figure()

    ax1 = Axis(fig[1, 1], title="Hierarchical Localized EnKF")

    # scatter!(ax1, xgrid, x_tsnap, label = "Truth")
    lines!(ax1, xgrid, X_hlocenkf_tsnap, linewidth=3, label="HLocEnKF")
    lines!(ax1, xgrid, ys, linewidth=3, label="State")
    lines!(ax1, xgrid[PA_offset+1:end-PA_offset], theta_tsnap, linewidth=3, label="θ")
    for j in 1:Ne
        lines!(ax1, xgrid, X_ens_tsnap[j], linewidth=0.9, color=(cols[1+(j%length(cols))], 0.2))
    end
    scatter!(ax1, xgrid[1:delta_y:end], y_tsnap)

    axislegend(ax1)


    framerate = 10
    timestamps = range(t_start, Tf, step=1)

    anim = Makie.Record(fig, timestamps; framerate=framerate) do t
        tsnap[] = t
    end
    save(joinpath(@__DIR__, "figs", "assim_hlenkf.mp4"), anim)
    anim
end

# %%
make_figs && with_theme(my_theme) do
    fig = Figure(fontsize=20, size=(1200, 400))

    ax1 = Axis(fig[1, 1],
        title=L"\text{Truth}",
        xlabel=L"t",
        ylabel=L"x",)

    h1 = heatmap!(ax1, data.tt, xgrid, data.xt')

    Colorbar(fig[1, 4], h1, label=L"u(x, t)")


    ax2 = Axis(fig[1, 2],
        title=L"\text{EnKF}",
        xlabel=L"t",
        ylabel=L"x",)
    h2 = heatmap!(ax2, data.tt, xgrid, mean_hist(X_locenkf)[:, 2:end]')


    ax3 = Axis(fig[1, 3],
        title=L"\text{GSBL EnKF}",
        xlabel=L"t",
        ylabel=L"x",)
    h3 = heatmap!(ax3, data.tt, xgrid, mean_hist(X_hlocenkf)[:, 2:end]')

    save(joinpath(@__DIR__, "figs", "heatmap_inviscid_burgers.png"), fig)

    fig
end;

# %%
make_figs && with_theme(my_theme) do
    t_start = 1
    tsnap = Observable(t_start)
    x_tsnap = @lift(data.xt[:, $tsnap])
    y_tsnap = @lift(data.yt[:, $tsnap])
    X_locenkf_tsnap = @lift(vec(mean(X_locenkf[$tsnap+1]; dims=2)))
    X_ens_tsnap = [@lift(X_locenkf[$tsnap+1][:, j]) for j in 1:Ne]
    cols = Makie.wong_colors()

    fig = Figure()

    ax1 = Axis(fig[1, 1], title="Localized EnKF")

    # scatter!(ax1, xgrid, x_tsnap, label = "Truth")
    lines!(ax1, xgrid, X_locenkf_tsnap, linewidth=3, label="LocEnKF")
    lines!(ax1, xgrid, ys, linewidth=3, label="State")
    for j in 1:Ne
        lines!(ax1, xgrid, X_ens_tsnap[j], linewidth=0.9, color=(cols[1+(j%length(cols))], 0.2))
    end
    scatter!(ax1, xgrid[1:delta_y:end], y_tsnap)

    axislegend(ax1)


    framerate = 10
    timestamps = range(t_start, Tf, step=1)

    anim = Makie.Record(fig, timestamps; framerate=framerate) do t
        tsnap[] = t
    end
    save(joinpath(@__DIR__, "figs", "assim_lenkf.mp4"), anim)
    anim
end

