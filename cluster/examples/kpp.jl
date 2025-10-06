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
verbose = false
data_path = joinpath(@__DIR__, "data")
proj_path = joinpath(@__DIR__, "..")
random_seed = rand(UInt)

proj_path = joinpath(@__DIR__, "../..")
# make_figs = true

# %%
# Problem setup params
polydeg = 2 # Order in space
Ncells_dim = 32 # Number of DG cells
delta_y = 25 # Spatial frequency of observation. Not regularly spaced
delta_t_dyn = 0.005 # Timestep for PDE dynamics
delta_t_obs = 0.025 # Amount of time between each observation

sigma_x_data = 0. # Noise in the state dynamics (i.e., the PDE solution itself)
sigma_y = 0.01 # Noise in the state observation (i.e., what the "sensors" record)

t0, tf = 0.0, 0.5 # Start and end time

# %%
# Important parameters for data assimilation
Ne = 50 # Ensemble size
Lrad = 10 # Localization radius
sigma_x_filter = 0.02 # State noise
beta_infl = 1.02 # Inflation param
alpha_k_f0, L_f0 = 0.7, 1.0 # Parameters for initial condition

# %%
# GSBL Hyperparams
order_PA = 3 # Poly annihilator order
Niter = 5
theta_init = 1.
hyperprior_idx = 4

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
sys_kpp = setup_kpp(polydeg, Ncells_dim)
Tf = ceil(Int, (tf - t0) / delta_t_obs)

# %%
Nvar = nvariables(sys_kpp.equations)
H = create_observation_operator2d(
    sys_kpp, polydeg + 1; offset=(polydeg + 1) ÷ 2, Nvar
)
Ny, Nx = size(H)
Nx_var = Nx ÷ Nvar

# %%
π0 = MvNormal(I(Nx))
@show sigma_x_data sigma_x_filter

ϵx_data = AdditiveInflation(Nx, sigma_x_data)
ϵx_filter = AdditiveInflation(Nx, sigma_x_filter)

ϵy = AdditiveInflation(Ny, sigma_y);

# %%
h(x, t) = H * x
F = StateSpace(identity, h)
model = Model(Nx, Ny, delta_t_dyn, delta_t_obs, ϵx_data, ϵy, π0, 0, 0, 0, F)
ode_solver = SSPRK43()
x0_sol = map(xy -> initial_condition_kpp(xy, 0., sys_kpp.equations), zip(sys_kpp.mesh.md.xyzq...))
x0 = sol2vec(x0_sol, sys_kpp.equations)

# %%
data = generate_data_trixi(model, x0, Tf, sys_kpp; ode_solver, cfl=0.9)

# %%
unique_digits = 5
grid1d = unique(x -> round(x, digits=unique_digits), sys_kpp.mesh.md.xq)
f0_row = SmoothPeriodic(grid1d, alpha_k_f0; L=L_f0, Nvar)
f0_col = SmoothPeriodic(grid1d, alpha_k_f0; L=L_f0, Nvar)
# Need prim vars ρ and p to be positive
x0_ens = reduce(hcat, sample_initial_state2d(sys_kpp, f0_row, f0_col; Nvar, unique_digits) for _ in 1:Ne)
noise_level_t0 = 0.05
for c_idx in CartesianIndices(x0_ens)
    (state_idx, ens_idx) = Tuple(c_idx)
    x0_ens[c_idx] = muladd(noise_level_t0, x0_ens[c_idx], (1 - noise_level_t0) * x0[state_idx])
end

# %%
Loc = Localization(sys_kpp, Lrad; Nvar, isperiodic=true)

# %%
Cϵ = LinearMap(ϵy.Σ, Ny)
ĈX = LocalizedEmpiricalCov(x0_ens, Loc; with_matrix=false)

# %%
CX_init = ĈX #LinearMaps.FunctionMap{Float64,true}(, Nx, issymmetric=true)
sparse_pattern = Int.(filter(!iszero, H * (1:size(H, 2))))
sys_y = ObsSystem(H, Cϵ, CX_init; use_workspace=true, sparse_pattern);

# %%
filter_inflation = MultiAddInflation(Nx, beta_infl, zeros(Nx), sigma_x_filter)
locenkf = LocEnKF(identity, Ne, ϵy, sys_y, Loc, delta_t_dyn, delta_t_obs, isfiltered=true, isiterative=true);

# %%
store_state_path = joinpath(@__DIR__, "data")
X_locenkf = seqassim_trixi(data, Tf, filter_inflation, locenkf, copy(x0_ens), model.Ny, model.Nx, t0, sys_kpp; ode_solver, cfl=0.8, store_state_path, verbose);

# %%
# Selection of hyper-prior parameters
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
# PA_offset = ceil(Int64, order_PA / 2)
PA, PA_nz_idx = VerticalPolyAnnil2D(sys_kpp, order_PA; Nvar, istruncated=true, isperiodic=false, periodic_limits=(-2, 2))
S = LinearMap(PA.P)
Ns = size(PA.P, 1)
theta_init_vec = fill(theta_init, Ns)
Cθ = LinearMap(Diagonal(theta_init_vec))
isiterative = true
sys_ys = ObsConstraintSystem(LinearMap(Matrix(H)), S, Cθ, Cϵ, CX_init; isiterative)

# %%
ϵy = AdditiveInflation(Ny, zeros(Ny), sigma_y);
hlocenkf = HLocEnKF(identity, Ne, ϵy, sys_ys, Loc, dist, theta_init_vec, delta_t_dyn, delta_t_obs; Niter=2, θinit=theta_init, isiterative, isfiltered=false)

# %%
X_hlocenkf, θ_hlocenkf = seqassim_trixi(data, Tf, filter_inflation, hlocenkf, copy(x0_ens), model.Ny, model.Nx, t0, sys_kpp; store_state_path, verbose)

# %%
# tspan = (0.0, 1.0)
# ode = semidiscretize(sys_kpp.semi, tspan)

# summary_callback = SummaryCallback()
# analysis_callback = AnalysisCallback(sys_kpp.semi, interval=100, uEltype=Float64)
# stepsize_callback = StepsizeCallback(; cfl=0.8)
# alive_callback = AliveCallback(analysis_interval=200)
# callbacks = CallbackSet(summary_callback,
#     analysis_callback, alive_callback,
#     stepsize_callback
# )
# sol_dgmulti = solve(ode, SSPRK43(); ode_default_options()..., callback=callbacks)

# ##
# s_state = sol_dgmulti(1.0)
# # s_sp = PA.P * vec(s_state)
# # pd_sol = PlotData2D(reshape(s_sp, size(s_state)), sys_kpp.semi)
# pd_sol = PlotData2D(s_state, sys_kpp.semi)
# plot(pd_sol)

##
# plot_data = SVector{1}.(reshape(@view(data.xt[:, 2]), (polydeg + 1)^2, Ncells_dim^2))
# pd_sol = PlotData2D(plot_data, sys_kpp.semi)
# plot(pd_sol)

##
# plot_ens = SVector{1}.(reshape(@view(X_hlocenkf[1][:, 1]), (polydeg + 1)^2, Ncells_dim^2))
# pd_sol = PlotData2D(plot_ens, sys_kpp.semi)
# fig = plot(pd_sol)
# x_coord = H * vec(sys_kpp.mesh.md.xq)
# y_coord = H * vec(sys_kpp.mesh.md.yq)
# scatter(x_coord, y_coord)
# fig

##
# PA_app = PA.P * data.xt[:, 2]
# out = zeros(size(PA.P, 2))
# mul!(@view(out[PA_nz_idx]), PA.P, @view(data.xt[:, 2]))
# heatmap(reshape(out, Ncells_dim * (polydeg + 1), :), axis=(aspect=1.,))