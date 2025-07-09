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
#my_theme = Theme()

# %%
random_seed = rand(UInt)

# %%
# PDE solution parameters
polydeg = 6
advection_velocity = 0.1

Ncells = 100
coordinates_min, coordinates_max = -1., 1.

# %%
# Data generation setup
delta_y = 50
delta_t_dyn = 0.05
delta_t_obs = 0.25
t0, tf = 0.0, 1.0
sigma_x_data = 1e-5
sigma_y = 0.2

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
# using CairoMakie
using JLD2
using Dates
using Random
using JLD2
using Trixi: entropy2cons
@info "Loaded packages"

# %%
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
X0 = zeros(model.Ny + model.Nx, Ne)
for i = 1:Ne
    regenerate!(f0)
    X0[Ny+1:Ny+Nx, i] = 0.5 * (f0.(xgrid) .+ 1.)
end

# %%
Nx = length(xgrid)
yidx = 1:delta_y:Nx

# # Create Localization structure
Gxx(i, j) = periodicmetric!(i, j, Nx)
Gxy(i, j) = periodicmetric!(i, yidx[j], Nx)
Gyy(i, j) = periodicmetric!(yidx[i], yidx[j], Nx)

Loc = Localization(Lrad, Gxx, Gxy, Gxx)
ϵxβ_enkf = MultiAddInflation(Nx, beta_infl, zeros(Nx), sigma_x_filter)

# %%
Cϵ = LinearMap(ϵy.Σ)
# This CX is replaced with the estimated state cov at each step
CX = LinearMap(I(Nx))
sys_y = ObsSystem(H, Cϵ, CX)
locenkf = LocEnKF(Ne, ϵy, sys_y, Loc, delta_t_dyn, delta_t_obs)

# %%
@info "Performing EnKF..."
X_locenkf = seqassim_trixi(data, Tf, ϵxβ_enkf, locenkf, deepcopy(X0), model.Ny, model.Nx, t0, sys_advection);

## Selecion of hyper-prior parameters
# power parameter
r_range = [1.0, 0.5, -0.5, -1.0];
r_GSBL = r_range[hyperprior_idx] # select parameter 
# shape parameter
β_range = [1.501, 3.0918, 2.0165, 1.0017];
β_GSBL = β_range[hyperprior_idx] # shape parameter
# rate parameters 
ϑ_range = [5 * 10^(-2), 5.9323 * 10^(-3), 1.2583 * 10^(-3), 1.2308 * 10^(-4)];
ϑ_GSBL = ϑ_range[hyperprior_idx]

dist = GeneralizedGamma(r_GSBL, β_GSBL, ϑ_GSBL);

# %%
PA_offset = ceil(Int, order_PA / 2)
Ns = Nx - 2PA_offset
PA = PolyAnnil(xgrid, order_PA; istruncated=true)
S = LinearMaps.FunctionMap{Float64,true}((s, x) -> mul!(s, PA.P, x), (x, s) -> mul!(x, PA.P', s), Ns, Nx; issymmetric=false, isposdef=false)
xgrid_S = xgrid[PA_offset+1:end-PA_offset];

# %%
theta_init_vec = fill(theta_init, Ns)
Cθ = LinearMap(Diagonal(theta_init_vec))
sys_ys = ObsConstraintSystem(H, S, Cθ, Cϵ, CX)

# %%
hlocenkf = HLocEnKF(Ne, ϵy, sys_ys, Loc, dist, theta_init_vec, delta_t_dyn, delta_t_obs; Niter, θinit=theta_init)

# %%
@info "Performing GSBL EnKF..."
X_hlocenkf, θhist = seqassim_trixi(data, Tf, ϵxβ_enkf, hlocenkf, deepcopy(X0), model.Ny, model.Nx, t0, sys_advection);

# %%
mesh_weights = vec(sys_advection.mesh.md.wJq);

# %%
weighted_norm2 = (x, w) -> sqrt(sum(dim_idx -> w[dim_idx] * abs2(x[dim_idx]), eachindex(x, w)))
rel_norms = map(Base.Fix2(weighted_norm2, mesh_weights), eachcol(data.xt))

errs_locenkf2 = map(j -> CRPS(X_locenkf[j+1], @view(data.xt[:, j]), :norm2, mesh_weights), axes(data.xt, 2))
errs_hlocenkf2 = map(j -> CRPS(X_hlocenkf[j+1], @view(data.xt[:, j]), :norm2, mesh_weights), axes(data.xt, 2))

rmse2_locenkf, rmse2_hlocenkf = [mean(i -> err[i].rmse / rel_norms[i], eachindex(err, rel_norms)) for err in [errs_locenkf2, errs_hlocenkf2]]
crps2_locenkf, crps2_hlocenkf = [mean(i -> err[i].crps / rel_norms[i], eachindex(err, rel_norms)) for err in [errs_locenkf2, errs_hlocenkf2]]
make_figs && @info "2-Norm results" rmse2_locenkf crps2_locenkf "======================" rmse2_hlocenkf crps2_hlocenkf;

# %%
weighted_norm1 = (x, w) -> sum(dim_idx -> w[dim_idx] * abs(x[dim_idx]), eachindex(x, w))
rel_norms = map(Base.Fix2(weighted_norm1, mesh_weights), eachcol(data.xt))

errs_locenkf1 = map(j -> CRPS(X_locenkf[j+1], @view(data.xt[:, j]), :norm1, mesh_weights), axes(data.xt, 2))
errs_hlocenkf1 = map(j -> CRPS(X_hlocenkf[j+1], @view(data.xt[:, j]), :norm1, mesh_weights), axes(data.xt, 2))

rmse1_locenkf, rmse1_hlocenkf = [mean(i -> err[i].rmse / rel_norms[i], eachindex(err, rel_norms)) for err in [errs_locenkf1, errs_hlocenkf1]]
crps1_locenkf, crps1_hlocenkf = [mean(i -> err[i].crps / rel_norms[i], eachindex(err, rel_norms)) for err in [errs_locenkf1, errs_hlocenkf1]]
make_figs && @info "1-Norm results" rmse1_locenkf crps1_locenkf "======================" rmse1_hlocenkf crps1_hlocenkf;

# %%
mass_true, energy_true = [weight_sum_reduction.(eachcol(data.xt), fcn, (mesh_weights,)) for fcn in (abs, abs2)]
mass_locenkf, energy_locenkf = [reduce(hcat, weight_sum_reduction.(eachcol(x), fcn, (mesh_weights,)) for x in X_locenkf) for fcn in (abs, abs2)]
mass_hlocenkf, energy_hlocenkf = [reduce(hcat, weight_sum_reduction.(eachcol(x), fcn, (mesh_weights,)) for x in X_hlocenkf) for fcn in (abs, abs2)]
mass_err_locenkf, energy_err_locenkf = [mean(t_idx -> abs(mean(enkf[:, t_idx+1]) - truth[t_idx]) / truth[t_idx], eachindex(truth)) for (truth, enkf) in [(mass_true, mass_locenkf), (energy_true, energy_locenkf)]]
mass_err_hlocenkf, energy_err_hlocenkf = [mean(t_idx -> abs(mean(enkf[:, t_idx+1]) - truth[t_idx]) / truth[t_idx], eachindex(truth)) for (truth, enkf) in [(mass_true, mass_hlocenkf), (energy_true, energy_hlocenkf)]]
make_figs && @info "Summary stat results" mass_err_locenkf energy_err_locenkf "======================" mass_err_hlocenkf energy_err_hlocenkf;

# %%
jldopen(joinpath(data_path, "linear_advection_" * string(now()) * ".jld2"), "w") do file
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

    metric_group = JLD2.Group(file, "metrics")

    for alg in ["locenkf", "hlocenkf"]
        metric_subgroup = JLD2.Group(metric_group, alg)
        for metric in ["rmse1", "crps1", "rmse2", "crps2", "mass_err", "energy_err"]
            metric_symbol = Symbol(metric * "_" * alg)
            metric_subgroup[metric] = @eval($metric_symbol)
        end
    end


    filter_group = JLD2.Group(file, "filters")
    filter_group["X_locenkf"] = ("Localized EnKF", X_locenkf)
    filter_group["X_hlocenkf"] = ("Hierarchical Localized EnKF", X_hlocenkf)
end;

# %%
make_figs && with_theme(my_theme) do
    t_start = 4
    tsnap = Observable(t_start)
    #x_tsnap = @lift(data.xt[:, $tsnap])
    #y_tsnap = @lift(data.yt[:, $tsnap])
    #X_hlocenkf_tsnap = @lift(vec(mean(X_hlocenkf[$tsnap+1]; dims=2)))
    #X_ens_tsnap = [@lift(X_hlocenkf[$tsnap+1][:, j]) for j in 1:Ne]
    #theta_tsnap = @lift(θhist[$tsnap+1])
    cols = Makie.wong_colors()

    fig = Figure()

    ax1 = Axis(fig[1, 1], title="Hierarchical Localized EnKF")

    # scatter!(ax1, xgrid, x_tsnap, label = "Truth")
    lines!(ax1, xgrid, X_hlocenkf_tsnap, linewidth=3, label="HLocEnKF")
    lines!(ax1, xgrid, x_tsnap, linewidth=3, label="Truth")
    # lines!(ax1, xgrid, ys, linewidth=3, label="State")
    lines!(ax1, xgrid[PA_offset+1:end-PA_offset], theta_tsnap, linewidth=3, label="θ")
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
    save(joinpath(@__DIR__, "figs", "assim_hlenkf.mp4"), anim)
    anim
end

# %%
make_figs && with_theme(my_theme) do
    t_start = 3
    tsnap = Observable(t_start)
    cols = Makie.wong_colors()

    #x_tsnap = @lift(data.xt[:, $tsnap])
    #x_tsnap_plus = @lift(data.xt[:, $tsnap] .+ sigma_x_filter)
    #y_tsnap = @lift(data.yt[:, $tsnap])
    #X_locenkf_tsnap = @lift(vec(mean(X_locenkf[$tsnap+1]; dims=2)))
    #X_locenkf_ens_tsnap = [@lift(X_locenkf[$tsnap+1][:, j]) for j in 1:Ne]

    fig = Figure()

    ax1 = Axis(fig[1, 1], title="Unadjusted EnKF")

    lines!(ax1, xgrid, x_tsnap, linewidth=3, label="Truth")
    lines!(ax1, xgrid, x_tsnap_plus, linewidth=3, label="Truth+σ")
    lines!(ax1, xgrid, X_locenkf_tsnap, linewidth=3, label="EnKF")
    for j in 1:Ne
        lines!(ax1, xgrid, X_locenkf_ens_tsnap[j], linewidth=0.9, color=(cols[1+(j%length(cols))], 0.2))
    end
    scatter!(ax1, xgrid[1:delta_y:end], y_tsnap)

    axislegend(ax1)


    framerate = 10
    timestamps = range(t_start, Tf, step=1)

    anim = CairoMakie.Makie.Record(fig, timestamps; framerate=framerate) do t
        tsnap[] = t
    end
    save(joinpath(@__DIR__, "figs", "assim_enkf.mp4"), anim)
    anim
end

