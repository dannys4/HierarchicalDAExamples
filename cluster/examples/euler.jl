# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:percent
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

# %% [markdown]
# ### Overall Parameters

# %%
proj_path = joinpath(@__DIR__, "..")
data_path = joinpath(@__DIR__, "data")
make_figs = false
random_seed = rand(UInt)

# %% [markdown]
# ### Data generating parameters

# %%
polydeg = 2
Ncells = 200
Nvar = 3

delta_t_dyn = 0.02
delta_t_obs = 0.04

t0 = 0.0
tf = 2.0

delta_y = 100
density_thresh, pressure_thresh = 5e-6, 5e-6
sigma_y = 0.1
sigma_x_data = 0.0

# %% [markdown]
# ### EnKF Parameters

# %%
alpha_k_f0, L_f0 = 1.0, 10.0
sigma_x_filter = 0.02
beta_infl = 1.00
Lrad = 7
Ne = 40
cfl = 0.9

# %% [markdown]
# ### GSBL parameters

# %%
order_PA = 3
hyperprior_idx = 4
theta_init = 1.0
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

# %%
using Revise
using Trixi
using LinearAlgebra
using OrdinaryDiffEq
using HierarchicalDA
using LinearMaps
using TransportBasedInference2
using Distributions
using SparseArrays
using JLD2
using Dates
using Random

# %%
Random.seed!(random_seed);

# %%
equations = CompressibleEulerEquations1D(1.4)

Nxvar = (polydeg + 1) * Ncells
Nx = Nvar * Nxvar

Nyvar = ceil(Int64, Nxvar / delta_y)
Ny = Nvar * Nyvar

# Define Trixi system for inviscid Burgers equation
sys_euler = setup_euler(polydeg, Ncells);

xgrid = GridFromMesh(sys_euler);

# %%
PA_skip = ceil(Int64, order_PA / 2)
Nsvar = Nxvar - 2 * PA_skip
Ns = Nvar * Nsvar

PA = PolyAnnil(xgrid, order_PA; Nvar=Nvar, istruncated=true)

@assert size(PA.P) == (Ns, Nx)

S = LinearMaps.LinearMap(PA.P)
PA_offset = ceil(Int, order_PA / 2)
xs = xgrid[PA_offset+1:end-PA_offset];

# %%
idxρ = 1:length(xgrid)
idxv = length(xgrid) .+ collect(1:length(xgrid))
idxp = 2 * length(xgrid) .+ collect(1:length(xgrid));

idxρy = 1:ceil(Int64, Nx / (delta_y * Nvar))
idxvy = ceil(Int64, Nx / (delta_y * Nvar)) .+ collect(1:ceil(Int64, Nx / (delta_y * Nvar)))
idxpy = 2 * ceil(Int64, Nx / (delta_y * Nvar)) .+ collect(1:ceil(Int64, Nx / (delta_y * Nvar)))

idxρs = idxρ[PA_skip+1:end-PA_skip]
idxvs = idxv[PA_skip+1:end-PA_skip]
idxps = idxp[PA_skip+1:end-PA_skip];

# %%
Tf = round(Int, (tf - t0) / delta_t_obs)

π0 = MvNormal(zeros(Nx), Matrix(1.0 * I, Nx, Nx))

# %%
ϵx_true = AdditiveInflation(Nx, zeros(Nx), sigma_x_data)
ϵx_filter = AdditiveInflation(Nx, zeros(Nx), sigma_x_filter)

ϵy = AdditiveInflation(Ny, zeros(Ny), sigma_y);

# %%
h(x, t) = x[unroll(1:delta_y:length(xgrid), length(xgrid), Nvar)]
H = LinearMap(sparse(Matrix(1.0 * I, Nx, Nx)[unroll(1:delta_y:length(xgrid), length(xgrid), Nvar), :]))
F = StateSpace(x -> x, h)

model = Model(Nx, Ny, delta_t_dyn, delta_t_obs, ϵx_true, ϵy, π0, 0, 0, 0, F);

# Define function class for the initial condition
f0 = SmoothPeriodic(xgrid, alpha_k_f0; L=L_f0, Nvar=Nvar)

# %%
xshuosher = zeros(Nx)
x0 = zeros(Nx)

for (i, xi) in enumerate(xgrid)
    x̃i = cons2prim(initial_condition_shu_osher(xi, 0.0, sys_euler.equations), sys_euler.equations)
    for k = 1:Nvar
        xshuosher[Nxvar*(k-1)+i] = x̃i[k]
    end
end

x0 = xshuosher;# + 0.01*f0(xgrid);

# %%
thresholds = (density_thresh, pressure_thresh)
variables = (Trixi.density, Trixi.pressure)
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=thresholds,
    variables=variables)
ode_solver = SSPRK43(stage_limiter!)

# %%
data = generate_data_trixi(deepcopy(model), deepcopy(x0), Tf, deepcopy(sys_euler); ode_solver, cfl=0.9)

# %%
quad_wts = vec(sys_euler.mesh.md.wJq)
ents = zeros(size(data.xt, 2))
for (t, x) in enumerate(eachcol(data.xt))
    u_state = reshape(x, :, 3)
    ents[t] = quad_wts'map(u -> Trixi.entropy(vec(u), sys_euler.equations), eachrow(u_state))
end

# %%
eval(Expr(:isdefined, :Makie)) || macro L_str(args...) end; # Define L_str in case we aren't loading CairoMakie
eval(Expr(:isdefined, :Makie)) || macro lift(args...) end; # Define L_str in case we aren't loading CairoMakie

# %%
make_figs && display(lines(data.tt, ents, axis=(; title="Entropy of Shu-Osher shock", xlabel=L"t", ylabel=L"e")));

# %%
make_figs && with_theme(my_theme) do
    fig = Figure()
    ax = Axis(fig[1, 1], title="Initial Condition", ylabel=L"u(0,x)", xlabel=L"x")
    xgrid = -5:0.01:5
    u0 = reduce(hcat, initial_condition_shu_osher.(xgrid, (0.,), equations))
    lines!(xgrid, u0[1, :], label=L"\rho")
    lines!(xgrid, u0[2, :], label=L"\rho v_1")
    lines!(xgrid, u0[3, :], label=L"\rho e")
    axislegend()
    save(joinpath(@__DIR__, "figs", "initial_condition.pdf"))
    display(fig)
end;

# %%
make_figs && with_theme(my_theme) do
    N_T = length(data.tt)
    p_t = t -> data.xt[idxps, t]
    ρ_t = t -> data.xt[idxρs, t]
    v_t = t -> data.xt[idxvs, t]

    time = Observable(1)
    ps = @lift(p_t($time))
    ρs = @lift(ρ_t($time))
    vs = @lift(v_t($time))

    fig = Figure()
    title_times = round.(data.tt, digits=2)
    ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"u(t,x)", title=@lift("Shu-Osher, t = $(title_times[$time])"))
    lines!(xs, ρs, label=L"\rho", linewidth=3)
    lines!(xs, ps, label=L"p", linewidth=3)
    lines!(xs, vs, label=L"v", linewidth=3)
    axislegend()
    # hlines!(vec(mesh_x), fill(0.05, length(mesh_x)))
    timestamps = 1:N_T
    anim = CairoMakie.Makie.Record(fig, timestamps; framerate=N_T ÷ 4) do t
        time[] = t
    end
    save(joinpath(@__DIR__, "figs", "solution.mp4"), anim)
    display(anim)
end;

# %%
make_figs && with_theme(my_theme) do
    fig = Figure(size=(2100, 700))
    for (i, idx_i) in enumerate([(idxρ, idxρy), (idxv, idxvy), (idxp, idxpy)])
        idx_x, idx_y = idx_i
        axi = Axis(fig[1, i])
        lines!(axi, xgrid, data.xt[idx_x, 1], linewidth=3)
        scatter!(axi, xgrid[1:delta_y:end], data.yt[idx_y, 1], markersize=18)
        errorbars!(axi, xgrid[1:delta_y:end], data.yt[idx_y, 1], fill(2sigma_y, length(idx_y)))
        lines!(axi, xgrid, data.xt[idx_x, div(end, 3)], linewidth=3)
        scatter!(axi, xgrid[1:delta_y:end], data.yt[idx_y, div(end, 3)], markersize=18)
        errorbars!(axi, xgrid[1:delta_y:end], data.yt[idx_y, div(end, 3)], fill(2sigma_y, length(idx_y)))
    end
    save(joinpath(@__DIR__, "figs", "time_slices.pdf"), fig)
    display(fig)
end;

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
X = zeros(model.Ny + model.Nx, Ne)

for i = 1:Ne
    regenerate!(f0)
    X[Ny+1:Ny+Nx, i] = max.(1e-5, xshuosher + 0.1 * f0(xgrid))#initial_condition(αk, Δx, Nx)
end

# %%
CX = LinearMap(collect(1. * I(Nx)))
Cϵ = LinearMap(ϵy.Σ)
sys_y = ObsSystem(H, Cϵ, CX)

theta_init_vec = fill(theta_init, Ns)
Cθ = LinearMap(Diagonal(theta_init_vec))
sys_ys = ObsConstraintSystem(H, S, Cθ, Cϵ)

# %%
yidx = 1:delta_y:Nx
idx = vcat(collect(1:length(yidx))', collect(yidx)')

# # Create Localization structure
@inline Gxx(i, j) = cartesianmetric(mod(i, Nxvar), mod(j, Nxvar))
# @inline Gxy(i, j) = cartesianmetric(mod(i, Nxvar), mod(yidx[j], Nxvar))
# @inline Gyy(i, j) = cartesianmetric(mod(yidx[i], Nxvar), mod(yidx[j], Nxvar))
Loc = Localization(Nx, Lrad, Gxx, is_sparse=true)

filter_inflation = MultiAddInflation(Nx, beta_infl, zeros(Nx), sigma_x_filter)

# %%
success_locenkf = success_hlocenkf = true;

# %%
# locenkf = LocEnKF(x -> max.(x, 1e-6), Ne, ϵy, sys_y, Loc, delta_t_dyn, delta_t_obs, isfiltered=true)

# %%
local X_locenkf
try
    global X_locenkf
    X_locenkf = seqassim_trixi(data, Tf, filter_inflation, locenkf, deepcopy(X), model.Ny, model.Nx, t0, sys_euler; ode_solver, cfl)
catch e
    global X_locenkf
    @warn "Localized EnKF failed for Shu-Osher"
    success_locenkf = false
    X_locenkf = nothing
end

# %%
hlocenkf = HLocEnKF(x -> max.(x, 1e-6), Ne, ϵy, sys_ys, Loc, dist, theta_init_vec, delta_t_dyn, delta_t_obs; Niter, θinit=theta_init, isfiltered=true)

# %%
local X_hlocenkf, θ_hlocenkf
try
    global X_hlocenkf, θ_hlocenkf
    X_hlocenkf, θ_hlocenkf = seqassim_trixi(data, Tf, filter_inflation, hlocenkf, deepcopy(X), model.Ny, model.Nx, t0, sys_euler; ode_solver, cfl)
catch e
    rethrow(e)
    global X_hlocenkf, θ_hlocenkf
    @warn "GSBL localized EnKF failed for Shu-Osher"
    success_hlocenkf = false
    X_hlocenkf = θ_hlocenkf = nothing
end

# %%
mesh_weights = repeat(vec(sys_euler.mesh.md.wJq), nvariables(sys_euler.equations))
wt_state = @view(mesh_weights[idxρ]) # weights for a single state variable
mass_true = data.xt[idxρ, :]' * wt_state

weighted_norm1 = (x, w) -> sum(dim_idx -> w[dim_idx] * abs(x[dim_idx]), eachindex(x, w))
rel_norms1 = map(Base.Fix2(weighted_norm1, mesh_weights), eachcol(data.xt))

weighted_norm2 = (x, w) -> sqrt(sum(dim_idx -> w[dim_idx] * abs2(x[dim_idx]), eachindex(x, w)))
rel_norms2 = map(Base.Fix2(weighted_norm2, mesh_weights), eachcol(data.xt))

# Functor to calculate the energy over for one full state at one single time
entropy_functor = u -> Trixi.entropy.(eachrow(reshape(u, :, 3)), sys_euler.equations)' * wt_state
entropy_true = [entropy_functor(x) for x in eachcol(data.xt)];

# %% [raw]
# entropy_true

# %%
hlocenkf_metrics = Dict{Symbol,Any}()
if @isdefined(X_hlocenkf)
    errs_hlocenkf1, errs_hlocenkf2 = [
        map(j -> CRPS(X_hlocenkf[j+1], @view(data.xt[:, j]), which_norm, mesh_weights), axes(data.xt, 2))
        for which_norm in [:norm1, :norm2]
    ]

    rmse1_hlocenkf, crps1_hlocenkf = [
        mean(i -> getproperty(errs_hlocenkf1[i], prop) / rel_norms1[i], eachindex(errs_hlocenkf1, rel_norms1))
        for prop in [:rmse, :crps]
    ]

    rmse2_hlocenkf, crps2_hlocenkf = [
        mean(i -> getproperty(errs_hlocenkf2[i], prop) / rel_norms2[i], eachindex(errs_hlocenkf2, rel_norms2))
        for prop in [:rmse, :crps]
    ]

    mass_hlocenkf = reduce(hcat, @view(x[idxρ, :])' * wt_state for x in X_hlocenkf)
    entropy_hlocenkf = reduce(hcat, entropy_functor.(eachcol(x)) for x in X_hlocenkf)

    for res in [:rmse1_hlocenkf, :rmse2_hlocenkf, :crps1_hlocenkf, :crps2_hlocenkf, :mass_hlocenkf, :entropy_hlocenkf]
        hlocenkf_metrics[res] = @eval($res)
    end
    @info "GSBL results" hlocenkf_metrics
end

# %%
locenkf_metrics = Dict{Symbol,Float64}()
if @isdefined(X_locenkf)
    errs_locenkf1, errs_locenkf2 = [
        map(j -> CRPS(X_locenkf[j+1], @view(data.xt[:, j]), which_norm, mesh_weights), axes(data.xt, 2))
        for which_norm in [:norm1, :norm2]
    ]

    rmse1_locenkf, crps1_locenkf = [
        mean(i -> getproperty(errs_locenkf1[i], prop) / rel_norms2[i], eachindex(errs_locenkf1, rel_norms1))
        for prop in [:rmse, :crps]
    ]

    rmse2_locenkf, crps2_locenkf = [
        mean(i -> getproperty(errs_locenkf2[i], prop) / rel_norms2[i], eachindex(errs_locenkf2, rel_norms2))
        for prop in [:rmse, :crps]
    ]

    mass_locenkf = reduce(hcat, @view(x[idxρ, :])' * wt_state for x in X_locenkf)
    entropy_locenkf = reduce(hcat, entropy_functor.(eachcol(x)) for x in X_locenkf)

    for res in [:rmse1_locenkf, :rmse2_locenkf, :crps1_locenkf, :crps2_locenkf, :mass_locenkf, :entropy_locenkf]
        locenkf_metrics[res] = @eval($res)
    end
    @info "EnKF results" locenkf_metrics
end

# %%
jldopen(joinpath(data_path, "shu_osher_" * string(now()) * ".jld2"), "w") do file
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
        eval(Expr(:isdefined, Symbol("X_" * alg))) || continue
        metric_dict = @eval($Symbol(alg * "_metrics"))
        metric_subgroup = JLD2.Group(metric_group, alg)
        for metric in keys(metric_dict)
            metric_subgroup[metric] = metric_dict[metric]
        end
    end

    filter_group = JLD2.Group(file, "filters")
    @isdefined(X_locenkf) && (filter_group["X_locenkf"] = ("Localized EnKF", @eval($(:X_locenkf))))
    @isdefined(X_hlocenkf) && (filter_group["X_hlocenkf"] = ("Hierarchical Localized EnKF", X_hlocenkf))
end;

# %%
make_figs && with_theme(my_theme()) do
    fig = Figure(size=(1010, 500))
    tsnap = 50
    idx = 10
    ax1 = Axis(fig[1, 1])
    ax2 = Axis(fig[1, 2])
    lines!(ax1, xgrid, data.xt[idxρ, tsnap], linewidth=3, label="Truth")
    lines!(ax2, xgrid, data.xt[idxρ, tsnap], linewidth=3, label="Truth")

    # lines!(ax, xgrid, X[Ny+1:Ny+Nx,2])
    # lines!(ax, xgrid, X_enkf[tsnap+1][:,idx], linewidth = 3, label = "EnKF")
    # lines!(ax, xgrid, mean(X_enkf[tsnap+1]; dims = 2)[:,1], linewidth = 3, label = "EnKF")

    # lines!(ax, xgrid, X_locenkf[tsnap+1][:,1][idxρ,1], linewidth = 3, label = "LocEnKF")
    # lines!(ax, xgrid, mean(X_locenkf[tsnap+1]; dims = 2)[idxρ,1], linewidth = 3, label = "LocEnKF")


    # lines!(ax, xgrid, X_henkf[tsnap+1][:,2], linewidth = 3, label = "HEnKF")
    # lines!(ax, xgrid, X_henkf[tsnap+1][:,2])
    # lines!(ax, xgrid, mean(X_henkf[tsnap+1]; dims = 2)[:,1], linewidth = 3, label = "HEnKF")

    cols = Makie.wong_colors()
    for j in 1:Ne
        col = cols[mod1(j, length(cols))]
        lines!(ax1, xgrid, X_locenkf[tsnap+1][:, j][idxρ, 1], linewidth=0.8, label=ifelse(j == 1, "Loc-EnKF", nothing), color=(col, 0.2))
        lines!(ax2, xgrid, X_hlocenkf[tsnap+1][:, j][idxρ, 1], linewidth=0.8, label=ifelse(j == 1, "GSBL-EnKF", nothing), color=(col, 0.2))
    end
    # lines!(ax, xgrid, mean(X_hlocenkf[tsnap+1]; dims = 2)[idxρ,1], linewidth = 3, label = "HLocEnKF")

    # ax2 = Axis(fig[1,2])

    # fig[1, 2] = Legend(fig, ax, "Filters", framevisible = false)

    # lines!(ax, xgrid[1:2:end], data.yt[idxpy,tsnap], linewidth = 3)


    # scatter!(ax, xgrid[1:delta_y:end], data.yt[:,tsnap])
    # lines!(ax, xs, PA.P*X_enkf[tsnap+1][:,2])

    axislegend(ax1)
    axislegend(ax2)
    fig
end
# %%
make_figs && with_theme(my_theme) do
    cols = Makie.wong_colors()
    N_T = length(data.tt)
    p_t = t -> data.xt[idxps, t]
    ρ_t = t -> data.xt[idxρs, t]
    v_t = t -> data.xt[idxvs, t]
    J_Ens = 5
    locenkf_p_t = t -> X_locenkf[t+1][:, J_Ens][idxps]
    locenkf_ρ_t = t -> X_locenkf[t+1][:, J_Ens][idxρs]
    locenkf_v_t = t -> X_locenkf[t+1][:, J_Ens][idxvs]
    # Offset each to account for previous offsets in data

    time = Observable(1)
    ps = @lift(p_t($time))
    ρs = @lift(ρ_t($time))
    vs = @lift(v_t($time))
    locenkf_ps = @lift(locenkf_p_t($time))
    locenkf_ρs = @lift(locenkf_ρ_t($time))
    locenkf_vs = @lift(locenkf_v_t($time))
    xs_theta = xs[1+PA_offset:end-PA_offset]

    px_size = 600
    fig = Figure(size=(3px_size, px_size))
    ax_p = Axis(fig[1, 1], xlabel=L"x", title=L"p", aspect=1.)
    ax_ρ = Axis(fig[1, 2], xlabel=L"x", title=L"\rho", aspect=1.)
    ax_v = Axis(fig[1, 3], xlabel=L"x", title=L"v", aspect=1.)
    lines!(ax_p, xs, ps, linewidth=3, label=L"p", color=cols[1], linestyle=:dash)
    lines!(ax_p, xs, locenkf_ps, linewidth=3, label="LEnKF", color=cols[1])
    axislegend(ax_p, position=:rt)
    lines!(ax_ρ, xs, ρs, linewidth=3, label=L"\rho", color=cols[2], linestyle=:dash)
    lines!(ax_ρ, xs, locenkf_ρs, linewidth=3, label="LEnKF", color=cols[2])
    axislegend(ax_ρ, position=:rt)
    lines!(ax_v, xs, vs, linewidth=3, label=L"v", color=cols[3], linestyle=:dash)
    lines!(ax_v, xs, locenkf_vs, linewidth=3, label="LEnKF", color=cols[3])
    axislegend(ax_v, position=:rt)

    timestamps = 1:N_T
    anim = CairoMakie.Makie.Record(fig, timestamps; framerate=N_T ÷ 4) do t
        time[] = t
    end
    save("figs/loc_enkf_result.mp4", anim)
    anim
end

# %%
with_theme(my_theme) do
    cols = Makie.wong_colors()
    N_T = length(data.tt)
    p_t = t -> data.xt[idxps, t]
    ρ_t = t -> data.xt[idxρs, t]
    v_t = t -> data.xt[idxvs, t]
    J_Ens = 5
    hlocenkf_p_t = t -> X_hlocenkf[t][:, J_Ens][idxps]
    hlocenkf_ρ_t = t -> X_hlocenkf[t][:, J_Ens][idxρs]
    hlocenkf_v_t = t -> X_hlocenkf[t][:, J_Ens][idxvs]
    # Offset each to account for previous offsets in data
    θ_ρ_t = t -> (θ_hlocenkf[t][idxρs[PA_offset+1:end-PA_offset].-0PA_offset])
    θ_v_t = t -> (θ_hlocenkf[t][idxvs[PA_offset+1:end-PA_offset].-2PA_offset])
    θ_p_t = t -> (θ_hlocenkf[t][idxps[PA_offset+1:end-PA_offset].-4PA_offset])

    time = Observable(2)
    ps = @lift(p_t($time - 1))
    ρs = @lift(ρ_t($time - 1))
    vs = @lift(v_t($time - 1))
    θ_ps = @lift(θ_p_t($time) / norm(θ_p_t($time)))
    θ_ρs = @lift(θ_ρ_t($time) / norm(θ_ρ_t($time)))
    θ_vs = @lift(θ_v_t($time) / norm(θ_v_t($time)))
    hlocenkf_ps = @lift(hlocenkf_p_t($time))
    hlocenkf_ρs = @lift(hlocenkf_ρ_t($time))
    hlocenkf_vs = @lift(hlocenkf_v_t($time))
    xs_theta = xs[1+PA_offset:end-PA_offset]

    px_size = 600
    fig = Figure(size=(3px_size, px_size))
    ax_p = Axis(fig[1, 1], xlabel=L"x", title=L"p", aspect=1.)
    ax_ρ = Axis(fig[1, 2], xlabel=L"x", title=L"\rho", aspect=1.)
    ax_v = Axis(fig[1, 3], xlabel=L"x", title=L"v", aspect=1.)
    scatter!(ax_p, xs_theta, θ_ps, label=L"\theta_p", color=cols[1])
    lines!(ax_p, xs, ps, linewidth=3, label=L"p", color=cols[1], linestyle=:dash)
    lines!(ax_p, xs, hlocenkf_ps, linewidth=3, label="GSBL", color=cols[1])
    axislegend(ax_p, position=:rt)
    scatter!(ax_ρ, xs_theta, θ_ρs, label=L"\theta_\rho", color=cols[2])
    lines!(ax_ρ, xs, ρs, linewidth=3, label=L"\rho", color=cols[2], linestyle=:dash)
    lines!(ax_ρ, xs, hlocenkf_ρs, linewidth=3, label="GSBL", color=cols[2])
    axislegend(ax_ρ, position=:rt)
    scatter!(ax_v, xs_theta, θ_vs, label=L"\theta_v", color=cols[3])
    lines!(ax_v, xs, vs, linewidth=3, label=L"v", color=cols[3], linestyle=:dash)
    lines!(ax_v, xs, hlocenkf_vs, linewidth=3, label="GSBL", color=cols[3])
    axislegend(ax_v, position=:rt)

    timestamps = 2:N_T
    anim = CairoMakie.Makie.Record(fig, timestamps; framerate=N_T ÷ 4) do t
        time[] = t
    end
    save("figs/hloc_enkf_result.mp4", anim)
    anim
end

# %%
