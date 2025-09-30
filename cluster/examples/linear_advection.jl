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
polydeg = 3
advection_velocity = 0.1

Ncells = 100
coordinates_min, coordinates_max = -1., 1.

# %%
# Data generation setup
delta_y = 25
delta_t_dyn = 0.05
delta_t_obs = 0.25
t0, tf = 0.0, 10.0
sigma_x_data = 0.
sigma_y = 5e-2

# %% [markdown]
# ### Parameters for Filtering

# %%
# Important parameters for data assimilation
Ne = 50 # Ensemble size
Lrad = 7 # Localization radius
sigma_x_filter = 0.15 # State noise
beta_infl = 1.02 # Inflation param
alpha_k_f0 = 0.8 # Parameter for initial condition

# %% [markdown]
# ### Parameters for GSBL

# %%
order_PA = 3
hyperprior_idx = 4
theta_init = 1.
Niter = 5

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
function initial_condition_sawtooth_fcn(x, t;
    a=-1, b=1, N_saw=4, u_a=0.0, u_b=0.95)
    normalized_x = (x[] - a) / (b - a)
    which_seg = normalized_x * N_saw
    per_x = (which_seg - floor(which_seg))
    u_a + per_x * (u_b - u_a)
end
function initial_condition_sawtooth(x, t, _::LinearScalarAdvectionEquation1D)
    SVector(initial_condition_sawtooth_fcn(x, t))
end
function Trixi.entropy2cons(t, ::LinearScalarAdvectionEquation1D)
    t
end

# %%
# Set up the PDE
equations = LinearScalarAdvectionEquation1D(advection_velocity)
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
Ny = ceil(Int64, Nx / delta_y)
Tf = Int((tf - t0) / delta_t_obs)
π0 = MvNormal(zeros(Nx), Matrix(1.0 * I, Nx, Nx))

# %%
h(x, t) = x[1:delta_y:end]
H = LinearMap(sparse(Matrix(1.0 * I, Nx, Nx)[1:delta_y:end, :]))
F = StateSpace(x -> x, h)
sys_advection = TrixiSystem(equations, solver, mesh, semi)
ϵx_data = AdditiveInflation(Nx, zeros(Nx), sigma_x_data)
ϵy = AdditiveInflation(Ny, zeros(Ny), sigma_y)
model = Model(Nx, Ny, delta_t_dyn, delta_t_obs, ϵx_data, ϵy, π0, 0, 0, 0, F)

# %%
u0 = initial_condition_sawtooth_fcn.(xgrid, (0,))
@info "Generating data..."
data = generate_data_trixi(model, u0, Tf, sys_advection)

# %%
make_figs && with_theme(my_theme) do
    fig = Figure()
    ax = Axis(fig[1, 1])

    lines!(ax, xgrid, data.xt[:, 1])
    lines!(ax, xgrid, data.xt[:, end])
    scatter!(ax, xgrid[1:delta_y:end], data.yt[:, end])

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
Nx = length(xgrid)
yidx = 1:delta_y:Nx

# Create Localization structure
# Gxx(i, j) = periodicmetric!(i, j, Nx)
metric = PeriodicMetric(Nx)
Loc = Localization(Nx, Lrad, metric, is_sparse=true)
ϵxβ_enkf = MultiAddInflation(Nx, beta_infl, zeros(Nx), sigma_x_filter)

# %%
Cϵ = LinearMap(ϵy.Σ, Ny)
# This CX is replaced with the estimated state cov at each step
sys_y = ObsSystem(H, Cϵ)
locenkf = LocEnKF(Ne, ϵy, sys_y, Loc, delta_t_dyn, delta_t_obs)

# %%
@info "Performing EnKF..."
X_locenkf = seqassim_trixi(data, Tf, ϵxβ_enkf, locenkf, deepcopy(X0), model.Ny, model.Nx, t0, sys_advection);

# %%
# Selection of hyper-prior parameters power parameter
hyperprior_idx = 4
r_range = [1.0, 0.5, -0.5, -1.0];
r_GSBL = r_range[hyperprior_idx] # select parameter
# shape parameter
β_range = [1.501, 3.0918, 2.0165, 1.0017];
β_GSBL = β_range[hyperprior_idx] # shape parameter
# rate parameters
ϑ_range = [5 * 10^(-2), 5.9323 * 10^(-3), 1.2583 * 10^(-3), 1.2308 * 10^(-4)];
ϑ_GSBL = ϑ_range[hyperprior_idx]

# r_GSBL, β_GSBL, ϑ_GSBL = 1., 30., 1e-3
# ϑ_GSBL = 1e-3
dist = GeneralizedGamma(r_GSBL, β_GSBL, ϑ_GSBL);

# %%
order_PA = 3
# PA_offset = ceil(Int, order_PA / 2)
# Ns = Nx - 2PA_offset
PA = PolyAnnil(xgrid, order_PA; istruncated=true, isperiodic=true, periodic_limits=(coordinates_min, coordinates_max))
S = LinearMaps.LinearMap(PA.P)
xgrid_S = xgrid #xgrid[PA_offset+1:end-PA_offset];

# %%
theta_init_vec = fill(theta_init, length(xgrid_S))
Cθ = LinearMap(Diagonal(theta_init_vec))
sys_ys = ObsConstraintSystem(H, S, Cθ, Cϵ)

# %%
hlocenkf = HLocEnKF(Ne, ϵy, sys_ys, Loc, dist, theta_init_vec, delta_t_dyn, delta_t_obs; Niter=5, θinit=1.)

# %%
@info "Performing GSBL EnKF..."
X_hlocenkf, θhist = seqassim_trixi(data, Tf, ϵxβ_enkf, hlocenkf, deepcopy(X0), model.Ny, model.Nx, t0, sys_advection);

# %%
begin
    mesh_weights_state = vec(sys_advection.mesh.md.wJq)
    mesh_weights = repeat(mesh_weights_state, nvariables(equations))
    calc_moments = (ensemble, moment) -> weight_sum_reduction.(eachcol(ensemble), (moment,), (mesh_weights,))
    weighted_norm1 = (x, w) -> sum(dim_idx -> w[dim_idx] * abs(x[dim_idx]), eachindex(x, w))
    weighted_norm2 = (x, w) -> sqrt(sum(dim_idx -> w[dim_idx] * abs2(x[dim_idx]), eachindex(x, w)))
    rel_norms1 = calc_moments(data.xt, abs)# map(Base.Fix2(weighted_norm1, mesh_weights), eachcol(data.xt))
    rel_norms2 = map(Base.Fix2(weighted_norm2, mesh_weights), eachcol(data.xt))
    get_errs = (X, metric) -> map(j -> CRPS(X[j+1], @view(data.xt[:, j]), metric, mesh_weights), axes(data.xt, 2))
    get_Lp = (err, rel_norms, prop::Symbol) -> mean(er -> getproperty(er[1], prop) / er[2], zip(err, rel_norms))
    advection_entropy = u -> Trixi.entropy(u, sys_advection.equations)
    TV_norm_state = u -> sum(abs, diff(reshape(u, :, nvariables(equations)), dims=1))
    TV_norm_ensemble = u_ens -> TV_norm_state.(eachcol(u_ens))
    TVN_true = TV_norm_ensemble(data.xt)
    mass_true, entropy_true = map(f -> calc_moments(data.xt, f), [abs, advection_entropy])

    metrics_locenkf, metrics_hlocenkf = Dict{Symbol,Any}(), Dict{Symbol,Any}()
end

# %%
for alg_name in ["locenkf", "hlocenkf"]
    # Error metrics
    metric_sym = Symbol("metrics_$alg_name")
    metric_dict = @eval($metric_sym)
    X_sym = Symbol("X_$alg_name")
    X = @eval($X_sym)
    for which_norm in [1, 2]
        norm = Symbol("norm$which_norm")
        errs = get_errs(X, norm)
        rel_norms_sym = Symbol("rel_norms$which_norm")
        rel_norms = @eval($rel_norms_sym)
        for metric in [:rmse, :crps]
            metric_sym = Symbol(string(metric) * string(which_norm) * "_" * alg_name)
            metric_dict[metric_sym] = get_Lp(errs, rel_norms, metric)
        end
    end

    # Mass, Entropy, and TV
    tv_alg = reduce(hcat, TV_norm_ensemble(x) for x in X)
    mass_alg, entropy_alg = map(f -> reduce(hcat, calc_moments(x, f) for x in X), [abs, advection_entropy])
    metric_dict[:mass] = mass_alg
    metric_dict[:entropy] = entropy_alg
    metric_dict[:tv_norm] = tv_alg
end

# %%
jldopen(joinpath(data_path, "advection_" * string(now()) * ".jld2"), "w") do file
    data_group = JLD2.Group(file, "data")
    for property in propertynames(data)
        data_group[string(property)] = getproperty(data, property)
    end

    data_param_group = JLD2.Group(file, "data_parameters")
    for data_param in [:random_seed, :polydeg, :Ncells, :delta_t_dyn, :delta_t_obs, :sigma_x_data, :sigma_y, :t0, :tf]
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
    metric_group["true_mass"] = mass_true
    metric_group["true_entropy"] = entropy_true
    metric_group["true_tv_norm"] = TVN_true
    for alg in ["locenkf", "hlocenkf"]
        metric_subgroup = JLD2.Group(metric_group, alg)
        metric_alg = Symbol("metrics_$alg")
        metric_dict = @eval($metric_alg)
        for key in keys(metric_dict)
            metric_subgroup[string(key)] = metric_dict[key]
        end
        X_alg = Symbol("X_$alg")
        filter_group[string(X_alg)] = @eval($X_alg)
    end
end;

# %%
make_figs && with_theme(my_theme) do
    xt = data.xt[:, 9]
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(xgrid, xt, linewidth=3, label="True state")
    sp = 10 * PA.P * xt
    lines!(xgrid_S, sp, linewidth=3, label="PA application")
    axislegend()
    display(fig)
end

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
    scatter!(ax1, xgrid[1:delta_y:end], y_tsnap)

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
    scatter!(ax1, xgrid[1:delta_y:end], y_tsnap)
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
    scatter!(ax1, xgrid[1:delta_y:end], y_tsnap)
    axislegend(ax1)
    framerate = 10
    timestamps = range(t_start, Tf, step=1)

    anim = CairoMakie.Makie.Record(fig, timestamps; framerate=framerate) do t
        tsnap[] = t
    end
    save(joinpath(@__DIR__, "figs", "linear_advection", "assim_lenkf.mp4"), anim)
    anim
end

