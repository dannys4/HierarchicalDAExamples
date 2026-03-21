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
# PDE solution parameters
polydeg = 2
advection_velocity = 0.1
Ncells = 100
coordinates_min, coordinates_max = -1., 1.

# %%
# Data generation setup
delta_y = 10
delta_t_dyn = 0.05
delta_t_obs = 0.5
t0, tf = 0.0, 20.0
sigma_x_data = 0.
sigma_y = 0.05

# %% [markdown]
# ### Parameters for Filtering

# %%
# Important parameters for data assimilation
Ne = 75 # Ensemble size
Lrad = delta_t_obs * advection_velocity # Localization radius
sigma_x_filter = 0.05 # State noise
beta_infl = 1.02 # Inflation param
alpha_k_f0 = 0.8 # Parameter for initial condition

# %% [markdown]
# ### Parameters for GSBL

# %%
order_PA = 2
hyperprior_idx = 2
theta_init = 1.
Niter = 2
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

# %%
using Pkg
Pkg.activate(proj_path)
@info "Activated project"

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
using JLD2
using Trixi: entropy2cons
make_figs && using CairoMakie
@info "Loaded packages"

# %%
make_figs && (my_theme = theme_minimal())
make_figs || macro L_str(args...) end; # Define L_str in case we aren't loading CairoMakie
make_figs || macro lift(args...) end; # Define L_str in case we aren't loading CairoMakie
Random.seed!(random_seed);

# %%
# Set up functions for the PDE, initial condition and entropy setup
function sawtooth_fcn(
    x, t, α;
    a=-1, b=1, N_saw=4, u_a=0.0, u_b=1.0
)
    time_norm_x = (((x[] - a) - α[] * t[]) / (b - a)) % 1
    unit_x = time_norm_x < 0 ? 1 + time_norm_x : time_norm_x
    which_seg = unit_x * N_saw
    per_x = (which_seg - floor(which_seg))
    u_a + per_x * (u_b - u_a)
end

function true_solution_advection_sawtooth!(α, u, x, t)
    u .= sawtooth_fcn.(x, t, α)
end
function initial_condition_sawtooth(x, t, E::LinearScalarAdvectionEquation1D)
    SVector(sawtooth_fcn(x, t, E.advection_velocity))
end
function Trixi.entropy2cons(t, ::LinearScalarAdvectionEquation1D)
    t
end

# %%
# Set up the PDE
equations = LinearScalarAdvectionEquation1D(advection_velocity)
soln! = (args...) -> true_solution_advection_sawtooth!(advection_velocity, args...)
volume_flux = flux_central
surface_flux = flux_lax_friedrichs
basis = DGMultiBasis(Trixi.Line(), polydeg, approximation_type=GaussSBP())

indicator_sc = IndicatorHennemannGassner(equations, basis,
    alpha_max=0.5,
    alpha_min=0.001,
    alpha_smooth=true,
    variable=first)

surface_integral = SurfaceIntegralWeakForm(surface_flux)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc)
solver = DGMulti(basis; surface_integral, volume_integral)

mesh = DGMultiMesh(solver, (Ncells,); periodicity=true, coordinates_min, coordinates_max)
initial_condition = initial_condition_sawtooth

semi = SemidiscretizationHyperbolic(mesh,
    equations,
    initial_condition,
    solver, boundary_conditions=boundary_condition_periodic)
xgrid = vec(mesh.md.xq);

# %%
Nx = length(xgrid)
Tf = Int((tf - t0) / delta_t_obs)
π0 = MvNormal(zeros(Nx), Matrix(1.0 * I, Nx, Nx))

# %%
sparse_obs = 1:delta_y:Nx
# OPTIONAL: Put cluster of points near zero
# dens_obs_frac = 1 / 20
# center_region = round(Int, Nx*(0.5 - dens_obs_frac)) : (delta_y ÷ 3) : round(Int, Nx*(0.5 + dens_obs_frac))
# obs_indices = sort(unique(vcat(sparse_obs, center_region)))
obs_indices = sparse_obs
Ny = length(obs_indices)
h(x, t) = x[obs_indices]
H = LinearMap(sparse(Matrix(1.0 * I, Nx, Nx)[obs_indices, :]))
F = StateSpace(x -> x, h)
sys_advection = TrixiSystem(equations, solver, mesh, semi)
ϵx_data = AdditiveInflation(Nx, zeros(Nx), sigma_x_data)
ϵy = AdditiveInflation(Ny, zeros(Ny), sigma_y)
model = Model(Nx, Ny, delta_t_dyn, delta_t_obs, ϵx_data, ϵy, π0, 0, 0, 0, F)

# %%
u0 = sawtooth_fcn.(xgrid, (0,), (advection_velocity,))
@info "Generating data..."
data = generate_data_trixi(model, u0, Tf, sys_advection; (true_soln!)=soln!)

# %%
x_plot, data_plot = get_plot_ensemble(data.xt, sys_advection)
data_plot = data_plot[:, 1, :]

# %%
make_figs && with_theme(my_theme) do
    fig = Figure()
    ax = Axis(fig[1, 1])

    lines!(ax, xgrid, data.xt[:, 1], label=L"u(t_0, x)")
    lines!(ax, xgrid, data.xt[:, end], label=L"u(t_T,x)")
    scatter!(ax, xgrid[obs_indices], data.yt[:, end], label=L"y_T")
    # vlines!(ax, xgrid[center_region[[1,end]]], color=:black, label="Dens obs region")
    axislegend()
    fig
end

# %%
f0 = SmoothPeriodic(xgrid, alpha_k_f0)
X0 = zeros(model.Nx, Ne)
for i = 1:Ne
    regenerate!(f0)
    X0[:, i] = 0.5 * (f0.(xgrid) .+ 1.)
end

# %%
# Create Localization structure
# Gxx(i, j) = periodicmetric!(i, j, Nx)
metric = PeriodicMetric(round.(extrema(xgrid))...)
Loc = Localization(xgrid, Lrad, metric, symm_kernel=true, is_sparse=true)
ϵxβ_enkf = MultiAddInflation(Nx, beta_infl, zeros(Nx), sigma_x_filter)

# %%
Cϵ = LinearMap(ϵy.Σ, Ny)
# This CX is replaced with the estimated state cov at each step
sys_y = ObsSystem(H, Cϵ)
locenkf = LocEnKF(ϵy, sys_y, Loc, delta_t_dyn, delta_t_obs)

# %%
@info "Performing EnKF..."
X_locenkf = seqassim_trixi(data, Tf, ϵxβ_enkf, locenkf, deepcopy(X0), model.Ny, model.Nx, t0, sys_advection);

# %%
is_theta_shared = false
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

# r_GSBL, β_GSBL, ϑ_GSBL = 1., 30., 1e-3
ϑ_GSBL = 1e-7
dist = GeneralizedGamma(r_GSBL, β_GSBL, ϑ_GSBL);

# %%
# order_PA = 1
# PA = PolyAnnil(xgrid, order_PA; istruncated=true, isperiodic=true, periodic_limits=(coordinates_min, coordinates_max))
# S = LinearMap(PA.P * PA.P)
diff_map = DGMultiDiff1D(sys_advection, false)
diff_mat = sparse(diff_map)
sqrt_quad_wts = sqrt.(vec(sys_advection.mesh.md.wJq))
S = LinearMap(Matrix(Diagonal(sqrt_quad_wts) * diff_mat * diff_mat))
xgrid_S = xgrid

# %%
theta_init_vec = fill(theta_init, length(xgrid_S))
theta_init_space = is_theta_shared ? theta_init_vec : repeat(theta_init_vec, 1, Ne)
Cθ = LinearMap(Diagonal(theta_init_vec))
sys_ys = ObsConstraintSystem(H, S, Cθ, Cϵ)

# %%
forecast_scale_gsbl = 20
Lrad_gsbl = 0.5Lrad
# Loc_gsbl = ShockLocalization(PA.P, xgrid, Lrad_gsbl, metric, forecast_scale_gsbl, symm_kernel=true, is_sparse=true, is_periodic=true, thresh=0.75)
Loc_gsbl = Localization(xgrid, Lrad_gsbl, metric, forecast_scale_gsbl, symm_kernel=true, is_sparse=true)

# %%
Niter = 20
hlocenkf = HLocEnKF(Ne, ϵy, sys_ys, Loc_gsbl, dist, theta_init_space, delta_t_dyn, delta_t_obs; Niter=Niter, θinit=theta_init)

# %%
beta_infl_hlocenkf = beta_infl
ϵxβ_hlocenkf = MultiAddInflation(Nx, beta_infl_hlocenkf, zeros(Nx), sigma_x_filter)

@info "Performing GSBL EnKF..."
X_hlocenkf, θ_hlocenkf = seqassim_trixi(data, 10, ϵxβ_hlocenkf, hlocenkf, deepcopy(X0), model.Ny, model.Nx, t0, sys_advection);

# %%
make_figs && with_theme(my_theme) do
    tsnap = length(X_hlocenkf) - 1
    t_val = data.tt[tsnap]
    x_tsnap = data_plot[:, tsnap]
    y_tsnap = data.yt[:, tsnap]

    _, X_enkf_tsnap = get_plot_ensemble(X_locenkf[tsnap+1], sys_advection)
    X_enkf_tsnap = X_enkf_tsnap[:, 1, :]
    X_locenkf_tsnap = vec(mean(X_enkf_tsnap; dims=2))

    _, X_gsbl_tsnap = get_plot_ensemble(X_hlocenkf[tsnap+1], sys_advection)
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
    if is_theta_shared
        scatter!(ax_theta, xgrid, theta_tsnap, label="θ", markersize=5)
    end
    for j in 1:Ne
        col = (cols[1+(j%length(cols))], 0.4)
        lines!(ax_enkf, x_plot, X_enkf_tsnap[:, j], linewidth=2.0, color=col)
        lines!(ax_gsbl, x_plot, X_gsbl_tsnap[:, j], linewidth=2.0, color=col)
        if !is_theta_shared
            scatter!(ax_theta, xgrid, theta_tsnap[:, j], label="θ", markersize=5, color=col)
        end
    end
    lines!(ax_gsbl, x_plot, X_hlocenkf_tsnap, linewidth=3, label="Filter", color=cols[7], linestyle=:dot)
    scatter!(ax_gsbl, xgrid[obs_indices], y_tsnap, label="Observation", color=:black)
    lines!(ax_enkf, x_plot, X_locenkf_tsnap, linewidth=3, label="Filter", color=cols[7], linestyle=:dot)
    scatter!(ax_enkf, xgrid[obs_indices], y_tsnap, label="Observation", color=:black)
    axislegend(ax_enkf, orientation=:horizontal, position=(1.0, 1.2))
    display(fig)
    save(joinpath(@__DIR__, "figs", "linear_advection", "profile_comparison.png"), fig)
    save(joinpath(@__DIR__, "figs", "linear_advection", "profile_comparison.pdf"), fig)
end

# %%
begin
    equations = sys_advection.equations
    Nvar = nvariables(equations)
    get_traj_quad_pts = traj -> map(x -> reshape(get_filter_quad_pts(x, sys_advection), Nvar, :, Ne), traj)
    data_quad = reshape(get_filter_quad_pts(data.xt, sys_advection), Nvar, :, Tf)
    mesh_wts = vec(sys_advection.mesh.md.wJq)
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
            metric_dict[metric_sym] = [get_Lp(errs[j], rel_norms[j, :], metric) for j in 1:Nvar]
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
    lines!(gsbl_crps, color=cols[1], linewidth=3, label="GSBL CRPS")
    lines!(enkf_crps, color=cols[2], linewidth=3, label="EnKF CRPS")
    lines!(gsbl_rmse, color=cols[1], linestyle=:dash, linewidth=3, label="GSBL RMSE")
    lines!(enkf_rmse, color=cols[2], linestyle=:dash, linewidth=3, label="EnKF RMSE")
    axislegend()
    fig
end

# %%
jldopen(joinpath(data_path, "advection_" * string(now()) * ".jld2"), "w") do file
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

        metric_sym = Symbol("metrics_$alg")
        metric_dict = @eval($metric_sym)
        metric_subgroup = JLD2.Group(metric_group, alg)
        for metric in keys(metric_dict)
            metric_subgroup[string(metric)] = metric_dict[metric]
        end
    end
end;

# %%
# make_figs && with_theme(my_theme) do
#     xt = data.xt[:, 9]
#     fig = Figure()
#     ax = Axis(fig[1, 1])
#     lines!(xgrid, xt, linewidth=3, label="True state")
#     sp = 10 * PA.P * xt
#     lines!(xgrid_S, sp, linewidth=3, label="PA application")
#     axislegend()
#     display(fig)
# end

# %%
make_figs && with_theme(my_theme) do
    tsnap = length(X_hlocenkf) - 1
    x_tsnap = data.xt[:, tsnap]
    X_hlocenkf_tsnap = vec(mean(X_hlocenkf[tsnap+1]; dims=2))
    X_ens_tsnap = [X_hlocenkf[tsnap+1][:, j] for j in 1:Ne]
    theta_tsnap = θhist[tsnap+1]
    theta_tsnap *= 0.1 / maximum(theta_tsnap)
    y_tsnap = data.yt[:, tsnap]
    cols = Makie.wong_colors()

    fig = Figure()

    ax1 = Axis(fig[1, 1], title="Hierarchical Localized EnKF")

    # scatter!(ax1, xgrid, x_tsnap, label = "Truth")
    lines!(ax1, xgrid, X_hlocenkf_tsnap, linewidth=3, label="HLocEnKF")
    lines!(ax1, xgrid, x_tsnap, linewidth=3, label="Truth")
    # lines!(ax1, xgrid, ys, linewidth=3, label="State")
    scatter!(ax1, xgrid_S, theta_tsnap, label="θ", markersize=5)
    for j in 1:Ne
        lines!(ax1, xgrid, X_ens_tsnap[j], linewidth=0.9, color=(cols[1+(j%length(cols))], 0.4))
    end
    scatter!(ax1, xgrid[obs_indices], y_tsnap)

    axislegend(ax1)

    display(fig)
end


# %%
make_figs && with_theme(my_theme) do
    cols = Makie.wong_colors()
    t_start = 4
    tsnap = Observable(t_start)
    x_tsnap = @lift(data.xt[:, $tsnap])
    y_tsnap = @lift(data.yt[:, $tsnap])
    X_hlocenkf_tsnap = @lift(vec(mean(X_hlocenkf[$tsnap+1]; dims=2)))
    X_ens_tsnap = [@lift(X_hlocenkf[$tsnap+1][:, j]) for j in 1:Ne]
    theta_tsnap = @lift(θhist[$tsnap+1])
    fig = Figure()
    ax1 = Axis(fig[1, 1], title="Hierarchical Localized EnKF")

    lines!(ax1, xgrid, X_hlocenkf_tsnap, linewidth=3, label="HLocEnKF")
    lines!(ax1, xgrid, x_tsnap, linewidth=3, label="Truth")
    scatter!(ax1, xgrid_S, theta_tsnap, label="θ")
    for j in 1:Ne
        lines!(ax1, xgrid, X_ens_tsnap[j], linewidth=0.9, color=(cols[1+(j%length(cols))], 0.2))
    end
    scatter!(ax1, xgrid[obs_indices], y_tsnap)
    axislegend(ax1)
    framerate = 10
    timestamps = range(t_start, Tf, step=1)

    anim = CairoMakie.Makie.Record(fig, timestamps; framerate=framerate) do t
        tsnap[] = t
    end
    save(joinpath(@__DIR__, "figs", "linear_advection", "assim_hlenkf.mp4"), anim)
    anim
end

# %%
make_figs && with_theme(my_theme) do
    cols = Makie.wong_colors()
    t_start = 4
    tsnap = Observable(t_start)
    x_tsnap = @lift(data.xt[:, $tsnap])
    y_tsnap = @lift(data.yt[:, $tsnap])
    X_locenkf_tsnap = @lift(vec(mean(X_locenkf[$tsnap+1]; dims=2)))
    X_ens_tsnap = [@lift(X_locenkf[$tsnap+1][:, j]) for j in 1:Ne]
    fig = Figure()
    ax1 = Axis(fig[1, 1], title="Localized EnKF")

    lines!(ax1, xgrid, X_locenkf_tsnap, linewidth=3, label="LocEnKF")
    lines!(ax1, xgrid, x_tsnap, linewidth=3, label="Truth")
    for j in 1:Ne
        lines!(ax1, xgrid, X_ens_tsnap[j], linewidth=0.9, color=(cols[1+(j%length(cols))], 0.2))
    end
    scatter!(ax1, xgrid[obs_indices], y_tsnap)
    axislegend(ax1)
    framerate = 10
    timestamps = range(t_start, Tf, step=1)

    anim = CairoMakie.Makie.Record(fig, timestamps; framerate=framerate) do t
        tsnap[] = t
    end
    save(joinpath(@__DIR__, "figs", "linear_advection", "assim_lenkf.mp4"), anim)
    anim
end

