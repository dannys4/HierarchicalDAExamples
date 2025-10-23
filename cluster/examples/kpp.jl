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
verbose = true
data_root = joinpath(@__DIR__, "data")
proj_path = joinpath(@__DIR__, "..")
random_seed = rand(UInt)

proj_path = joinpath(@__DIR__, "../..")
make_figs = true

# %%
# Problem setup params
polydeg = 3 # Order in space
Ncells_dim = 48 # Number of DG cells
delta_y = 8 # Spatial frequency of observation. Not regularly spaced
delta_t_dyn = 0.005 # Timestep for PDE dynamics
delta_t_obs = 0.025 # Amount of time between each observation

sigma_x_data = 0. # Noise in the state dynamics (i.e., the PDE solution itself)
sigma_y = 1.0 # Noise in the state observation (i.e., what the "sensors" record)

t0, tf = 0.0, 0.75 # Start and end time

# %%
# Important parameters for data assimilation
Ne = 50 # Ensemble size
Nx_dim = Ncells_dim * (polydeg + 1)
Lrad = polydeg + 1 # Localization radius
sigma_x_filter = 0.25 # State noise
beta_infl = 1.02 # Inflation param
alpha_k_f0, L_f0 = 0.7, 1.0 # Parameters for initial condition
cg_tol = 1e-2

# %%
# GSBL Hyperparams
order_PA = 2 # Poly annihilator order
Niter = 2
theta_init = 1.
hyperprior_idx = 2

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
H, H_points = create_observation_operator2d(
    sys_kpp, delta_y; offset=(polydeg + 1) ÷ 2, Nvar
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
sim_id = Int(rand(UInt32))
data_path = mkdir(joinpath(data_root, "sim$(string(sim_id))"))
@save joinpath(data_path, "data.jld2") data polydeg Ncells_dim
store_state_path = data_path

# %%
unique_digits = 5
grid1d = unique(x -> round(x, digits=unique_digits), sys_kpp.mesh.md.xq)
f0_row = SmoothPeriodic(grid1d, alpha_k_f0; L=L_f0, Nvar)
f0_col = SmoothPeriodic(grid1d, alpha_k_f0; L=L_f0, Nvar)
# Need prim vars ρ and p to be positive
x0_ens = reduce(hcat, sample_initial_state2d(sys_kpp, f0_row, f0_col; Nvar, unique_digits) for _ in 1:Ne)
noise_rank = 10
x0_ens_aux = map(_ -> reduce(hcat, sample_initial_state2d(sys_kpp, f0_row, f0_col; Nvar, unique_digits) for _ in 1:Ne), 1:noise_rank)
noise_level_t0 = 0.5
for c_idx in CartesianIndices(x0_ens)
    (state_idx, ens_idx) = Tuple(c_idx)
    aux_noise = sum(x0a[c_idx] for x0a in x0_ens_aux)
    x0_ens[c_idx] = muladd(noise_level_t0, (x0_ens[c_idx] + aux_noise), (1 - noise_level_t0) * x0[state_idx])
end

# %%
make_figs && with_theme(my_theme) do
    initial_condition = ensemble_to_itp(x0_ens, sys_kpp)
    pd_sol = PlotData2D(initial_condition[:, :, 1], sys_kpp.semi)
    plot(pd_sol)
end

# %%
Loc = Localization(sys_kpp, Lrad; Nvar, isperiodic=true)

# %%
Cϵ = LinearMap(ϵy.Σ, Ny)
ĈX = LocalizedEmpiricalCov(x0_ens, Loc; with_matrix=false)

# %%
CX_init = ĈX #LinearMaps.FunctionMap{Float64,true}(, Nx, issymmetric=true)
sparse_pattern = Int.(H * (1:size(H, 2)))
sys_y = ObsSystem(H, Cϵ, CX_init; use_workspace=true, sparse_pattern);

# %%
filter_inflation = MultiAddInflation(Nx, beta_infl, zeros(Nx), sigma_x_filter)
isiterative = true
locenkf = LocEnKF(ϵy, sys_y, Loc, delta_t_dyn, delta_t_obs; isiterative, cg_tol)

# %%
X_locenkf = seqassim_trixi(data, Tf, filter_inflation, locenkf, copy(x0_ens), model.Ny, model.Nx, t0, sys_kpp; ode_solver, cfl=0.8, verbose=false, store_state_path);

# %%
# Selection of hyper-prior parameters
# power parameter

r_range = [1.0, 0.5, -0.5, -1.0];
r_GSBL = r_range[hyperprior_idx] # select parameter
# shape parameter
β_range = [1.001 + Ne / 2, 2.5918 + Ne / 2, 2.0165, 1.0017];
β_GSBL = β_range[hyperprior_idx] # shape parameter

unadj_means = [β_range[1], β_range[2] * (β_range[2] + 1), 1 / ((β_range[3] - 2) * (β_range[3] - 1)), 1 / (β_range[4] - 1)]
target_mean = 0.0724

# rate parameters
ϑ_range = target_mean ./ unadj_means;
ϑ_GSBL = ϑ_range[hyperprior_idx]
# ϑ_GSBL = 1e-1

dist = GeneralizedGamma(r_GSBL, β_GSBL, ϑ_GSBL);

# %%
# PA_offset = ceil(Int64, order_PA / 2)
PA, PA_nz_idx = PolyAnnil2D(sys_kpp, order_PA; Nvar, istruncated=true, isperiodic=true, periodic_limits=(-2, 2))
S = LinearMap(PA.P)
Ns = size(PA.P, 1)
theta_init_vec = fill(theta_init, Ns)
Cθ = LinearMap(Diagonal(theta_init_vec))
sys_ys = ObsConstraintSystem(H, S, Cθ, Cϵ, CX_init; cache_matrix=false, isiterative)

# %%
ϵy = AdditiveInflation(Ny, zeros(Ny), sigma_y);
hlocenkf = HLocEnKF(identity, Ne, ϵy, sys_ys, Loc, dist, theta_init_vec, delta_t_dyn, delta_t_obs; Niter, θinit=theta_init, isiterative, isfiltered=false, cg_tol=1e-2)

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

# %%
quad_wts = sys_kpp.mesh.md.wJq
quad_truth = vec2sol(@view(data.xt[:, 1]), sys_kpp)
norm2_truth = sum(eachindex(quad_wts)) do j
    quad_wts[j] * abs2.(quad_truth[j])
end
function rmse_one!(tmp, x)
    vec2sol!(tmp, x, sys_kpp.equations)
    mse_x = sum(eachindex(quad_wts)) do j
        quad_wts[j] * abs2.(quad_truth[j] - tmp[j])
    end
    sqrt(mse_x / norm2_truth)
end

# %%
tmp = similar(quad_truth)
rmse_hlocenkf = mean(1:Ne) do j
    rmse_one!(tmp, @view(X_hlocenkf[][:, j]))
end
rmse_locenkf = mean(1:Ne) do j
    rmse_one!(tmp, @view(X_locenkf[][:, j]))
end
@info "" rmse_hlocenkf[] rmse_locenkf[]

# %% Map KPP data to interpolation points
itp_data = ensemble_to_itp(data.xt, sys_kpp)
itp_locenkf = ensemble_to_itp(X_locenkf[end], sys_kpp)
itp_hlocenkf = ensemble_to_itp(X_hlocenkf[end], sys_kpp)
θ_plot = ensemble_to_itp(reshape(θ_hlocenkf[end], :, 2), sys_kpp)

# %% Animate data.xt
# make_figs && with_theme(my_theme) do
#     pd_sols = map(slice -> PlotData2D(slice, sys_kpp.semi)["u"], eachslice(itp_data, dims=3))
#     fig = Figure(size=(600, 500))
#     ax = Axis(fig[1, 1], aspect=1., limits=(-2, 2, -2, 2))
#     colorrange = map(getindex, extrema(itp_data))
#     cb = Colorbar(fig[1, 2]; colorrange)

#     for timestamp in 1:Tf
#         plt_t = TrixiMakie.trixiheatmap!(ax, pd_sols[timestamp], plot_mesh=false; colorrange)
#         pic_fname = joinpath(@__DIR__, "figs", "kpp_sol", "t$(timestamp).png")
#         save(pic_fname, fig)
#         delete!(ax, plt_t)
#     end
# end

# %%
# obs_end = reshape(data.yt[:, end], :, Int(sqrt(size(H_points, 2))))
# hx, hy = reshape.(eachrow(H_points), (:,), Int(sqrt(size(H_points, 2))))
# heatmap(hx[:, 1], hy[1, :], obs_end; axis=(; aspect=1.))

# %%
make_figs && with_theme(my_theme) do
    pd_sols = map([itp_data, itp_hlocenkf, itp_locenkf]) do itp
        PlotData2D(itp[:, :, 1], sys_kpp.semi)["u"]
    end
    fig = Figure(size=(1900, 500))
    titles = ["Truth", "GSBL-DA", "LEnKF"]
    axs = map(1:3) do j
        Axis(fig[1, j], aspect=1., limits=(-2, 2, -2, 2), title=titles[j])
    end
    colorranges = map(pd_sols) do pds
        map(getindex, extrema(pds.plot_data.data))
    end
    colorrange = (minimum(first.(colorranges)), maximum(last.(colorranges)))
    for j in 1:3
        TrixiMakie.trixiheatmap!(axs[j], pd_sols[j], plot_mesh=false; colorrange)
    end
    cb = Colorbar(fig[1, 4]; colorrange)
    fig
end

# %%
make_figs && with_theme(my_theme) do
    pd_sol = PlotData2D(itp_hlocenkf[:, :, 1], sys_kpp.semi)
    plot(pd_sol)
end

# %%
make_figs && with_theme(my_theme) do
    pd_sol = PlotData2D(itp_locenkf[:, :, 2], sys_kpp.semi)
    plot(pd_sol)
end

# %%
make_figs && with_theme(my_theme) do
    mean_locenkf = mean(itp_locenkf, dims=3)[:, :]
    pd_sol = PlotData2D(mean_locenkf, sys_kpp.semi)
    plot(pd_sol)
end

# %%
make_figs && with_theme(my_theme) do
    mean_hlocenkf = mean(itp_hlocenkf, dims=3)[:, :]
    pd_sol = PlotData2D(mean_hlocenkf, sys_kpp.semi)
    plot(pd_sol)
end

# %%
make_figs && with_theme(my_theme) do
    pd_θ_x = PlotData2D(θ_plot[:, :, 1], sys_kpp.semi)
    plot(pd_θ_x)
end

# %%
make_figs && with_theme(my_theme) do
    pd_θ_y = PlotData2D(θ_plot[:, :, 2], sys_kpp.semi)
    plot(pd_θ_y)
end

##
# PA_app = PA.P * data.xt[:, 2]
# out = zeros(size(PA.P, 2))
# mul!(@view(out[PA_nz_idx]), PA.P, @view(data.xt[:, 2]))
# heatmap(reshape(out, Ncells_dim * (polydeg + 1), :), axis=(aspect=1.,))