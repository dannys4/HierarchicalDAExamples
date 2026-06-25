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
make_figs = false
data_path = joinpath(@__DIR__, "data")
proj_path = joinpath(@__DIR__, "..")
random_seed = rand(UInt)

proj_path = joinpath(@__DIR__, "../..")
make_figs = true

# %%
# Problem setup params
polydeg = 2 # Order in space
Ncells = 100 # Number of DG cells
delta_y = 15 # Spatial frequency of observation. Not regularly spaced
delta_t_dyn = 0.005 # Timestep for PDE dynamics
delta_t_obs = 0.025 # Amount of time between each observation

sigma_x_data = 0. # Noise in the state dynamics (i.e., the PDE solution itself)
sigma_y = 0.05 # Noise in the state observation (i.e., what the "sensors" record)

t0, tf = 0.0, 2.0 # Start and end time

# %%
# Important parameters for data assimilation
Ne = 40 # Ensemble size
Lrad = 0.6 * delta_t_obs # Localization radius
sigma_x_filter = 0.05 # State noise
beta_infl = 1.02 # Inflation param
alpha_k_f0, L_f0 = 0.7, 1.0 # Parameters for initial condition
initial_noise_perturb = 0.8

# %%
# GSBL Hyperparams
order_PA = 5 # Poly annihilator order
Niter = 2
theta_init = 1.
hyperprior_idx = 2
forecast_scale_gsbl = 3.0
is_theta_shared = false

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

# Lrad = 0.05 #delta_y / ((polydeg + 1) * Ncells) # Localization radius

# %%
using Pkg
Pkg.activate(proj_path)

# %%
using TransportBasedInference2
using HierarchicalDA
using LinearAlgebra
using OrdinaryDiffEq
using Trixi
using Distributions
using Statistics
using SparseArrays
using LinearMaps
using JLD2
using Dates
using Random
make_figs && using CairoMakie

# %%
make_figs && (my_theme = theme_minimal())
make_figs || macro L_str(args...) end; # Define L_str in case we aren't loading CairoMakie
make_figs || macro lift(args...) end; # Define L_str in case we aren't loading CairoMakie
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
make_figs && with_theme(my_theme) do
    lines(ents; axis=(; limits=(0, nothing, 0, nothing)))
end

# %%
x_plot, data_plot = get_plot_ensemble(data.xt, sys_burgers)
data_plot = data_plot[:, 1, :]

# %%
make_figs && with_theme(my_theme) do
    heatmap_data = interp_columns(data_plot, 10)
    fig, _ = heatmap(range(t0, tf, length=size(heatmap_data, 1)), x_plot, heatmap_data, axis=(; xlabel=L"t", ylabel=L"x", title=L"Solution of inviscid burgers, $u(x,t)$"))
    display(fig)
    save(joinpath(@__DIR__, "figs", "burgers", "heatmap_data.pdf"), fig)
end

# %%
make_figs && with_theme(my_theme) do
    cols = Makie.wong_colors()
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, x_plot, data_plot[:, 1], color=cols[2])
    lines!(ax, x_plot, data_plot[:, end], color=cols[1])
    scatter!(ax, xgrid[1:delta_y:end], data.yt[:, end], color=cols[1])
    fig
end

# %%
# Define function class for the initial condition
f0 = SmoothPeriodic(xgrid, alpha_k_f0; L=L_f0)
X = zeros(model.Nx, Ne)

for i = 1:Ne
    regenerate!(f0)
    X[:, i] = (1 - initial_noise_perturb) * x0 + initial_noise_perturb * (f0.(xgrid) / 3 .+ 0.5)
end

# %%
make_figs && with_theme(my_theme) do
    fig = Figure()
    ax = Axis(fig[1, 1])
    foreach(i -> lines!(ax, xgrid, X[:, i]), 1:10)
    lines!(ax, xgrid, x0, linewidth=10)
    fig
end

# %%
Cϵ = LinearMap(get_cov(ϵy, 0.))
# This CX is replaced with the estimated state cov at each step
CX = LocalizedEmpiricalCov(X, Loc)
sys_y = ObsSystem(H, Cϵ);

# %%
yidx = 1:delta_y:Nx
idx = vcat(collect(1:length(yidx))', collect(yidx)')

# Create Localization structure
metric = PeriodicMetric(-1, 1)
Loc = Localization(xgrid, Lrad, metric, is_sparse=true)

# beta_infl, sigma_x_filter = 1.04, 0.1
ϵxbeta_filter = MultiAddInflation(Nx, beta_infl, zeros(Nx), sigma_x_filter)

# %%
locenkf = LocEnKF(ϵy, sys_y, Loc, delta_t_dyn, delta_t_obs)

# %%
@info "Performing EnKF..."
X_locenkf = seqassim_trixi(data, Tf, ϵxbeta_filter, locenkf, deepcopy(X), model.Ny, model.Nx, t0, sys_burgers);

# %%
hyperprior_idx = 4
# Selection of hyper-prior parameters power parameter
r_range = [1.0, 0.5, -0.5, -1.0];
r_GSBL = r_range[hyperprior_idx] # select parameter
# shape parameter
β_shift = is_theta_shared ? Ne / 2 : 1 / 2
β_range = [1.001 + β_shift, 2.5918 + β_shift, 2.0165, 1.0017];
β_GSBL = β_range[hyperprior_idx] # shape parameter
# rate parameters
ϑ_range = [5 * 10^(-2), 5.9323 * 10^(-3), 1.2583 * 10^(-3), 1.2308 * 10^(-4)];
ϑ_GSBL = ϑ_range[hyperprior_idx]


r_GSBL = 0.5
β_GSBL = 0.05
ϑ_GSBL = 1e-3
dist = GeneralizedGamma(r_GSBL, β_GSBL, ϑ_GSBL);

# %%
# order_PA = 2
# PA = PolyAnnil(xgrid, order_PA; istruncated=true, isperiodic=true, periodic_limits=(-1., 1.))
# S = LinearMaps.WrappedMap(PA.P * PA.P)
diff_map = DGMultiDiff1D(sys_burgers, false)
diff_mat = sparse(diff_map)
S = LinearMap(diff_mat * diff_mat)

theta_init_vec = fill(theta_init, size(S, 1))
theta_init_space = is_theta_shared ? theta_init_vec : repeat(theta_init_vec, 1, Ne)
Cθ = LinearMap(Diagonal(theta_init_vec))
sys_ys = ObsConstraintSystem(H, S, Cθ, Cϵ)

# %%
forecast_scale_gsbl = 5
Lrad_gsbl = Lrad
Loc_gsbl = Localization(xgrid, Lrad_gsbl, metric, forecast_scale_gsbl, is_sparse=true)

# %%
Niter = 2
hlocenkf = HLocEnKF(Ne, ϵy, sys_ys, Loc_gsbl, dist, theta_init_space, delta_t_dyn, delta_t_obs; Niter, θinit=theta_init)

# %%
@info "Performing GSBL EnKF..."
T_hlocenkf = Tf # or Tf
X_hlocenkf, θ_hlocenkf = seqassim_trixi(data, T_hlocenkf, ϵxbeta_filter, hlocenkf, deepcopy(X), model.Ny, model.Nx, t0, sys_burgers);

# %%
make_figs && with_theme(my_theme) do
    tsnap = 40
    t_val = data.tt[tsnap]
    x_tsnap = data_plot[:, tsnap]
    y_tsnap = data.yt[:, tsnap]

    _, X_enkf_tsnap = get_plot_ensemble(X_locenkf[tsnap+1], sys_burgers)
    X_enkf_tsnap = X_enkf_tsnap[:, 1, :]
    X_locenkf_tsnap = vec(mean(X_enkf_tsnap; dims=2))

    _, X_gsbl_tsnap = get_plot_ensemble(X_hlocenkf[tsnap+1], sys_burgers)
    X_gsbl_tsnap = X_gsbl_tsnap[:, 1, :]
    X_hlocenkf_tsnap = vec(mean(X_gsbl_tsnap; dims=2))
    theta_tsnap = θ_hlocenkf[tsnap+1]
    cols = Makie.wong_colors()

    fig = Figure(size=(750, 550))
    ax_enkf = Axis(fig[1, 1], title="EnKF", xlabel=L"x", ylabel=L"u(x,%$(t_val))")
    ax_gsbl = Axis(fig[2, 1], title="GSBL-EnKF", xlabel=L"x", ylabel=L"u(x,%$(t_val))")
    ax_theta = Axis(fig[3, 1], ylabel=L"\theta", aspect=10, xlabel=L"x")
    linkxaxes!(ax_enkf, ax_gsbl, ax_theta)

    lines!(ax_enkf, x_plot, x_tsnap, linewidth=3, label="Data")
    lines!(ax_gsbl, x_plot, x_tsnap, linewidth=3, label="Data")
    is_theta_shared && scatter!(ax_theta, xgrid, theta_tsnap, label="θ", markersize=5)
    for j in 1:Ne
        color = (cols[1+(j%length(cols))], 0.4)
        lines!(ax_enkf, x_plot, X_enkf_tsnap[:, j], linewidth=0.8; color)
        lines!(ax_gsbl, x_plot, X_gsbl_tsnap[:, j], linewidth=0.8; color)
        is_theta_shared || scatter!(ax_theta, xgrid, theta_tsnap[:, j], label="θ", markersize=5; color)
    end
    lines!(ax_gsbl, x_plot, X_hlocenkf_tsnap, linewidth=3, label="Filter", color=cols[7], linestyle=:dot)
    scatter!(ax_gsbl, xgrid[1:delta_y:end], y_tsnap, label="Observation", color=:black)
    lines!(ax_enkf, x_plot, X_locenkf_tsnap, linewidth=3, label="Filter", color=cols[7], linestyle=:dot)
    scatter!(ax_enkf, xgrid[1:delta_y:end], y_tsnap, label="Observation", color=:black)
    axislegend(ax_enkf, orientation=:horizontal, position=(1.0, 1.2))
    Label(fig[0,1], "Time $t_val", tellwidth=false)
    display(fig)
    # save(joinpath(@__DIR__, "figs", "burgers", "profile_comparison.png"), fig)
    # save(joinpath(@__DIR__, "figs", "burgers", "profile_comparison.pdf"), fig)
end

# %%
begin
    equations = sys_burgers.equations
    Nvar = nvariables(equations)
    get_traj_quad_pts = traj -> map(x -> reshape(get_filter_quad_pts(x, sys_burgers), Nvar, :, Ne), traj)
    data_quad = reshape(get_filter_quad_pts(data.xt, sys_burgers), Nvar, :, Tf)
    mesh_wts = vec(sys_burgers.mesh.md.wJq)
    rel_norms1 = sum(j -> mesh_wts[j] * abs.(data_quad[:, j, :]), eachindex(mesh_wts))
    rel_norms2 = sqrt.(sum(j -> mesh_wts[j] * abs2.(data_quad[:, j, :]), eachindex(mesh_wts)))
    get_errs = (X, metric, which_var) -> map(axes(data.xt, 2)) do t_idx
        CRPS(X[t_idx+1][which_var, :, :], @view(data_quad[which_var, :, t_idx]), metric, mesh_wts)
    end
    get_Lp = (err, rel_norms, prop::Symbol) -> map(inp -> getproperty(inp[1], prop) / inp[2], zip(err, rel_norms))

    entropy_state = u -> Trixi.entropy(prim2cons(u, equations), equations)
    entropy_ensemble = u_ens -> map(entropy_state, eachslice(u_ens, dims=(2, 3)))' * mesh_wts

    TV_norm_state = u -> sum(abs, diff(u))
    TV_norm_ensemble = u_ens -> map(TV_norm_state, eachslice(u_ens, dims=(1, 3)))

    entropy_data = entropy_ensemble(data_quad)
    TV_data = TV_norm_ensemble(data_quad)
    metrics_locenkf, metrics_hlocenkf = Dict{Symbol,Any}(), Dict{Symbol,Any}()
end

# %%
for alg_name in ["locenkf", "hlocenkf"]
    start_idx = 1
    # Error metrics
    metric_sym = Symbol("metrics_$alg_name")
    metric_dict = @eval($metric_sym)
    X_sym = Symbol("X_$alg_name")
    X_traj = get_traj_quad_pts(@eval($X_sym))
    isnothing(X_traj) && continue
    for which_norm in [1, 2]
        norm = Symbol("norm$which_norm")
        errs = [get_errs(X_traj, norm, j) for j in 1:Nvar]
        rel_norms_sym = Symbol("rel_norms$which_norm")
        rel_norms = @eval($rel_norms_sym)
        for metric in [:rmse, :crps]
            metric_sym = Symbol(string(metric) * string(which_norm) * "_" * alg_name)
            metric_dict[metric_sym] = [get_Lp(errs[j][start_idx:end], rel_norms[j, start_idx:end], metric) for j in 1:Nvar]
        end
    end

    # TV, Mass, Entropy
    entropy_alg = map(entropy_ensemble, X_traj)
    tv_alg = map(TV_norm_ensemble, X_traj)
    metric_dict[:entropy] = entropy_alg
    metric_dict[:tv_norm] = tv_alg
end
@info "" mean(metrics_hlocenkf[:crps2_hlocenkf][]) mean(metrics_hlocenkf[:rmse2_hlocenkf][]) mean(metrics_locenkf[:crps2_locenkf][]) mean(metrics_locenkf[:rmse2_locenkf][])

# %%
make_figs && with_theme(my_theme) do
    fig = Figure(size=(500, 300))
    lims = (nothing, nothing, 1e-2, 1e0)
    ax = Axis(fig[1, 1], yscale=log10, limits=lims)
    gsbl_crps = metrics_hlocenkf[:crps2_hlocenkf][]
    gsbl_rmse = metrics_hlocenkf[:rmse2_hlocenkf][]
    enkf_crps = metrics_locenkf[:crps2_locenkf][]
    enkf_rmse = metrics_locenkf[:rmse2_locenkf][]
    cols = Makie.wong_colors()
    # vlines!([3, 5, 7], color=:black, label="Burn-in")
    lines!(gsbl_crps, color=cols[1], linewidth=3, label="GSBL CRPS")
    lines!(enkf_crps, color=cols[2], linewidth=3, label="EnKF CRPS")
    lines!(gsbl_rmse, color=cols[1], linestyle=:dash, linewidth=3, label="GSBL RMSE")
    lines!(enkf_rmse, color=cols[2], linestyle=:dash, linewidth=3, label="EnKF RMSE")
    axislegend()
    fig
end

# %%
jldopen(joinpath(data_path, "burgers_" * string(now()) * ".jld2"), "w") do file
    data_group = JLD2.Group(file, "data")
    for property in propertynames(data)
        data_group[string(property)] = getproperty(data, property)
    end

    data_param_group = JLD2.Group(file, "data_parameters")
    for data_param in [:random_seed, :polydeg, :Ncells, :delta_t_dyn, :delta_t_obs, :sigma_x_data, :sigma_y, :t0, :tf, :delta_y]
        data_param_group[string(data_param)] = @eval($data_param)
    end

    filter_param_group = JLD2.Group(file, "filter_parameters")
    for filter_param in [:Ne, :Lrad, :sigma_x_filter, :beta_infl, :alpha_k_f0]
        filter_param_group[string(filter_param)] = @eval($filter_param)
    end

    GSBL_param_group = JLD2.Group(file, "GSBL_parameters")
    for GSBL_param in [:order_PA, :Niter, :theta_init, :dist]
        GSBL_param_group[string(GSBL_param)] = @eval($GSBL_param)
    end

    filter_group = JLD2.Group(file, "filters")
    metric_group = JLD2.Group(file, "metrics")
    metric_group["true_entropy"] = entropy_data
    metric_group["true_tv_norm"] = TV_data
    for alg in ["locenkf", "hlocenkf"]
        X_alg = Symbol("X_$alg")
        X_traj = @eval($X_alg)
        filter_group[string(X_alg)] = X_traj
        if alg == "hlocenkf"
            filter_group["θ_hlocenkf"] = θ_hlocenkf
        end

        metric_sym = Symbol("metrics_$alg")
        metric_dict = @eval($metric_sym)
        metric_subgroup = JLD2.Group(metric_group, alg)
        for metric in keys(metric_dict)
            metric_subgroup[string(metric)] = metric_dict[metric]
        end
    end
end;

# %%
make_figs && with_theme(my_theme) do
    tsnap = 5#length(X_locenkf) - 1
    x_tsnap = data.xt[:, tsnap]
    X_locenkf_tsnap = vec(mean(X_locenkf[tsnap+1]; dims=2))
    X_ens_tsnap = [X_locenkf[tsnap+1][:, j] for j in 1:Ne]
    theta_tsnap = θ_hlocenkf[tsnap+1]
    theta_tsnap *= 0.1 / maximum(theta_tsnap)
    y_tsnap = data.yt[:, tsnap]
    cols = Makie.wong_colors()

    fig = Figure()

    ax1 = Axis(fig[1, 1], title="Localized EnKF")

    lines!(ax1, xgrid, X_locenkf_tsnap, linewidth=3, label="LocEnKF")
    lines!(ax1, xgrid, x_tsnap, linewidth=3, label="Truth")
    for j in 1:Ne
        lines!(ax1, xgrid, X_ens_tsnap[j], linewidth=0.9, color=(cols[1+(j%length(cols))], 0.4))
    end
    scatter!(ax1, xgrid[1:delta_y:end], y_tsnap)

    axislegend(ax1)

    display(fig)
end

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
    theta_tsnap = [@lift(θ_hlocenkf[$tsnap+1][:, j]) for j in 1:Ne]
    cols = Makie.wong_colors()

    fig = Figure()

    ax1 = Axis(fig[1, 1], title="Hierarchical Localized EnKF")

    lines!(ax1, xgrid, X_hlocenkf_tsnap, linewidth=3, label="HLocEnKF")
    lines!(ax1, xgrid, ys, linewidth=3, label="State")
    for j in 1:Ne
        lines!(ax1, xgrid, X_ens_tsnap[j], linewidth=0.9, color=(cols[1+(j%length(cols))], 0.2))
        lines!(ax1, xgrid, theta_tsnap[j], linewidth=3)
    end
    scatter!(ax1, xgrid[1:delta_y:end], y_tsnap)

    axislegend(ax1)
    framerate = 10
    timestamps = range(t_start, Tf, step=1)

    anim = Makie.Record(fig, timestamps; framerate=framerate) do t
        tsnap[] = t
    end
    save(joinpath(@__DIR__, "figs", "burgers", "assim_hlenkf.mp4"), anim)
    display(anim)
end

# %%
make_figs && with_theme(my_theme) do
    t_start = 1
    tsnap = Observable(t_start)
    x_tsnap = @lift(data.xt[:, $tsnap])
    y_tsnap = @lift(data.yt[:, $tsnap])
    ut = t -> map(x -> x[], vec(data.xt[:, round(Int, t / delta_t_obs)]))
    ys = @lift(ut(($tsnap) * delta_t_obs))
    X_locenkf_tsnap = @lift(vec(mean(X_locenkf[$tsnap+1]; dims=2)))
    X_ens_tsnap = [@lift(X_locenkf[$tsnap+1][:, j]) for j in 1:Ne]
    cols = Makie.wong_colors()

    fig = Figure()

    ax1 = Axis(fig[1, 1], title="Localized EnKF")
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
    save(joinpath(@__DIR__, "figs", "burgers", "assim_lenkf.mp4"), anim)
    display(anim)
end

# # %%
# make_figs && with_theme(my_theme) do
#     mean_locenkf = mean_hist(X_locenkf)[:, 1:end-1]
#     mean_hlocenkf = mean_hist(X_hlocenkf)[:, 1:end-1]
#     _, mean_locenkf = get_plot_ensemble(mean_locenkf, sys_burgers)
#     _, mean_hlocenkf = get_plot_ensemble(mean_hlocenkf, sys_burgers)
#     mean_locenkf = mean_locenkf[:, 1, :]
#     mean_hlocenkf = mean_hlocenkf[:, 1, :]
#     heatmap_locenkf = interp_columns(mean_locenkf, N_interp)
#     heatmap_hlocenkf = interp_columns(mean_hlocenkf, N_interp)
#     colorrange = extrema(reduce(vcat, collect(extrema(x)) for x in [heatmap_data, heatmap_locenkf, heatmap_hlocenkf]))
#     fig = Figure(fontsize=20, size=(1200, 400))
#     ax1 = Axis(fig[1, 1],
#         title=L"\text{Truth}",
#         xlabel=L"t",
#         ylabel=L"x",)
#     ax2 = Axis(fig[1, 2],
#         title=L"\text{EnKF}",
#         xlabel=L"t",
#         ylabel=L"x",)
#     ax3 = Axis(fig[1, 3],
#         title=L"\text{GSBL EnKF}",
#         xlabel=L"t",
#         ylabel=L"x",)
#     tgrid_plot = range(t0, tf, length=size(heatmap_data, 1))
#     h1 = heatmap!(ax1, tgrid_plot, x_plot, heatmap_data; colorrange)
#     heatmap!(ax2, tgrid_plot, x_plot, heatmap_locenkf; colorrange)
#     heatmap!(ax3, tgrid_plot, x_plot, heatmap_hlocenkf; colorrange)
#     Colorbar(fig[1, 4], label=L"u(x, t)"; colorrange)
#     display(fig)
#     save(joinpath(@__DIR__, "figs", "burgers", "heatmap_data.pdf"), fig)
#     save(joinpath(@__DIR__, "figs", "burgers", "heatmap_data.png"), fig)
# end

# %%
make_figs && with_theme(my_theme) do
    fig = Figure(size=(1050, 500))
    entropy_locenkf = reduce(hcat, metrics_locenkf[:entropy])'
    entropy_hlocenkf = reduce(hcat, metrics_hlocenkf[:entropy])'
    ylims = extrema(reduce(hcat, collect(extrema(x)) for x in [entropy_locenkf, entropy_hlocenkf]))
    ax1 = Axis(fig[1, 1],
        title="Burgers Entropy, EnKF",
        aspect=1.,
        xlabel=L"t",
        ylabel="Entropy",
        limits=(t0, tf, ylims...)
    )
    ax2 = Axis(fig[1, 2],
        title="Burgers Entropy, GSBL-EnKF",
        aspect=1.,
        xlabel=L"t",
        limits=(t0, tf, ylims...)
    )
    lines!(ax1, data.tt, entropy_data, linewidth=3, label="Entropy of solution")
    lines!(ax2, data.tt, entropy_data, linewidth=3, label="Entropy of solution")
    for ens_idx in 1:Ne
        lines!(ax1, data.tt, entropy_locenkf[2:end, ens_idx], linewidth=0.5)
        lines!(ax2, data.tt, entropy_hlocenkf[2:end, ens_idx], linewidth=0.5)
    end
    axislegend(ax1)
    axislegend(ax2)
    display(fig)
    save(joinpath(@__DIR__, "figs", "burgers", "entropy.pdf"), fig)
    save(joinpath(@__DIR__, "figs", "burgers", "entropy.png"), fig)
end

# %%
make_figs && with_theme(my_theme) do
    fig = Figure(size=(1050, 500))
    tv_locenkf = reduce(hcat, metrics_locenkf[:tv_norm]')'
    tv_hlocenkf = reduce(hcat, metrics_hlocenkf[:tv_norm]')'
    ylims = extrema(reduce(hcat, collect(extrema(x)) for x in [tv_locenkf, tv_hlocenkf]))
    ax1 = Axis(fig[1, 1],
        title="Burgers TV, EnKF",
        aspect=1.,
        xlabel=L"t",
        ylabel="TV",
        limits=(t0, tf, ylims...)
    )
    ax2 = Axis(fig[1, 2],
        title="Burgers TV, GSBL-EnKF",
        aspect=1.,
        xlabel=L"t",
        limits=(t0, tf, ylims...)
    )
    # lines!(ax1, data.tt, entropy_data, linewidth=3, label="TV of solution")
    # lines!(ax2, data.tt, entropy_data, linewidth=3, label="TV of solution")
    for ens_idx in 1:Ne
        lines!(ax1, data.tt, tv_locenkf[2:end, ens_idx], linewidth=0.5)
        lines!(ax2, data.tt, tv_hlocenkf[2:end, ens_idx], linewidth=0.5)
    end
    # axislegend(ax1)
    # axislegend(ax2)
    display(fig)
    save(joinpath(@__DIR__, "figs", "burgers", "tv.pdf"), fig)
    save(joinpath(@__DIR__, "figs", "burgers", "tv.png"), fig)
end
