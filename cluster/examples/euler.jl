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

proj_path = joinpath(@__DIR__, "../..")
make_figs = true

# %% [markdown]
# ### Data generating parameters

# %%
polydeg = 3
Ncells = 100
Nvar = 3

delta_t_dyn = 0.02
delta_t_obs = 0.04

t0 = 0.0
tf = 2.0

delta_y = 25
density_thresh, pressure_thresh = 5e-6, 5e-6
sigma_y = 0.1
sigma_x_data = 0.0

# %% [markdown]
# ### EnKF Parameters

# %%
alpha_k_f0, L_f0 = 1.0, 10.0
sigma_x_filter = 0.05
beta_infl = 1.02
Lrad = 7
Ne = 50
cfl = 0.9

# %% [markdown]
# ### GSBL parameters

# %%
order_PA = 6
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

# %%
begin # Execute all loads as part of one expr
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
    make_figs && using CairoMakie
end

# %%
make_figs && (my_theme = theme_minimal())
make_figs || macro L_str(args...) end; # Define L_str in case we aren't loading CairoMakie
make_figs || macro lift(args...) end; # Define L_str in case we aren't loading CairoMakie
Random.seed!(random_seed);

# %%
equations = CompressibleEulerEquations1D(1.4)

Nxvar = (polydeg + 1) * Ncells
Nx = Nvar * Nxvar

# Define Trixi system for inviscid Burgers equation
sys_euler = setup_euler(polydeg, Ncells);

xgrid = GridFromMesh(sys_euler);

# %%
idxρ = 3 * (0:(length(xgrid)-1)) .+ 1
idxv = 3 * (0:(length(xgrid)-1)) .+ 2
idxp = 3 * (0:(length(xgrid)-1)) .+ 3

idxρy = 3 * (0:ceil(Int, Nx / (delta_y * Nvar))-1) .+ 1
idxvy = 3 * (0:ceil(Int, Nx / (delta_y * Nvar))-1) .+ 2
idxpy = 3 * (0:ceil(Int, Nx / (delta_y * Nvar))-1) .+ 3

# %%
Tf = round(Int, (tf - t0) / delta_t_obs)
π0 = MvNormal(zeros(Nx), Matrix(1.0 * I, Nx, Nx))

# %%
ϵx_true = AdditiveInflation(Nx, zeros(Nx), sigma_x_data)
ϵx_filter = AdditiveInflation(Nx, zeros(Nx), sigma_x_filter)

# %%
all_idxy = sort(reduce(vcat, Nvar * (0:delta_y:(length(xgrid)-1)) .+ i for i in 2:3))#1:3))

Ny = length(all_idxy)

ϵy = AdditiveInflation(Ny, zeros(Ny), sigma_y);

h(x, t) = x[all_idxy]
H = LinearMap(sparse(Matrix(1.0 * I, Nx, Nx)[all_idxy, :]))
F = StateSpace(x -> x, h)

model = Model(Nx, Ny, delta_t_dyn, delta_t_obs, ϵx_true, ϵy, π0, 0, 0, 0, F);

# Define function class for the initial condition
f0 = SmoothPeriodic(xgrid, alpha_k_f0; L=L_f0, Nvar=Nvar)

# %%
# Gives me initial condition in cons
x0_quad = map(x -> initial_condition_shu_osher(x, 0., sys_euler.equations), sys_euler.mesh.md.xq)
# x0 is in prims
x0 = sol2vec(x0_quad, sys_euler.equations)# + 0.01*f0(xgrid);

# %%
thresholds = (density_thresh, pressure_thresh)
variables = (Trixi.density, Trixi.pressure)
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=thresholds,
    variables=variables)
ode_solver = SSPRK43(stage_limiter!)
# ode_solver = SSPRK43()

# %%
@info "Generating data..."
data = generate_data_trixi(deepcopy(model), deepcopy(x0), Tf, deepcopy(sys_euler); ode_solver, cfl=0.2)

# %%
false && with_theme(my_theme) do
    quad_wts = vec(sys_euler.mesh.md.wJq)
    ents = zeros(size(data.xt, 2))
    for (t, x) in enumerate(eachcol(data.xt))
        u_state = reshape(x, :, 3)
        ents[t] = quad_wts'map(u -> Trixi.entropy(vec(u), sys_euler.equations), eachrow(u_state))
    end
    display(lines(data.tt, ents, axis=(; title="Entropy of Shu-Osher shock", xlabel=L"t", ylabel=L"e")))
end

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
false && with_theme(my_theme) do
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
        lines!(axi, xgrid, data.xt[idx_x, div(end, 3)], linewidth=3)
        if i in [2, 3]
            i -= 1
            scatter!(axi, xgrid[1:delta_y:end], data.yt[i:2:end, 1], markersize=18)
            errorbars!(axi, xgrid[1:delta_y:end], data.yt[i:2:end, 1], fill(2sigma_y, length(idx_y)))
            scatter!(axi, xgrid[1:delta_y:end], data.yt[i:2:end, div(end, 3)], markersize=18)
            errorbars!(axi, xgrid[1:delta_y:end], data.yt[i:2:end, div(end, 3)], fill(2sigma_y, length(idx_y)))
        end
    end
    save(joinpath(@__DIR__, "figs", "time_slices.pdf"), fig)
    display(fig)
end;

# %%
pos_vars = ["rho", "p"]
pos_var_flags = in.(Trixi.varnames(cons2cons, equations), (pos_vars,))

##
noise_proportion = 0.25
X0 = positivity_preserving_noise1d(f0, initial_condition_shu_osher, Ne, sys_euler, pos_vars, noise_proportion)

# %%
make_figs && with_theme(my_theme) do
    fig = Figure(size=(2100, 700))
    axs = [Axis(fig[1, i], aspect=1., title=Trixi.varnames(cons2prim, equations)[i]) for i in 1:Nvar]
    num_ens_viz = 10
    for ens_idx in 1:num_ens_viz
        X_sample = @view X0[:, ens_idx]
        for (i, idx_i) in enumerate([(idxρ, idxρy), (idxv, idxvy), (idxp, idxpy)])
            idx_x, idx_y = idx_i
            lines!(axs[i], xgrid, X_sample[idx_x], linewidth=3)
            # lines!(axi, xgrid, X_sample[idx_x, div(end, 3)], linewidth=3)
        end
    end
    save(joinpath(@__DIR__, "figs", "initial_ensemble.pdf"), fig)
    display(fig)
end;

# %%
CX = LinearMap(collect(1. * I(Nx)))
Cϵ = LinearMap(ϵy.Σ, size(H, 1))
sys_y = ObsSystem(H, Cϵ)

yidx = 1:delta_y:Nx
idx = vcat(collect(1:length(yidx))', collect(yidx)')

# Create Localization structure
metric = (i, j) -> cartesianmetric((i - 1) ÷ Nvar + 1, (j - 1) ÷ Nvar + 1)
Loc = Localization(Nx, Lrad, metric, is_sparse=true, is_herm=false)

filter_inflation = MultiAddInflation(Nx, beta_infl, zeros(Nx), sigma_x_filter)

# %%
function filtering_fcn!(x)
    pressure = @view x[idxp]
    density = @view x[idxρ]
    pressure .= max.(pressure, (1e-6,))
    density .= max.(density, (1e-6,))
    nothing
end

# %%
locenkf = LocEnKF(filtering_fcn!, Ne, ϵy, sys_y, Loc, delta_t_dyn, delta_t_obs, isfiltered=true)

# %%
# X_locenkf = seqassim_trixi(data, 2, filter_inflation, locenkf, copy(X0), model.Ny, model.Nx, t0, sys_euler; ode_solver, cfl=0.2)

# %%
local X_locenkf
@info "Performing EnKF..."
try
    global X_locenkf
    X_locenkf = seqassim_trixi(data, Tf, filter_inflation, locenkf, copy(X0), model.Ny, model.Nx, t0, sys_euler; ode_solver, cfl)
catch e
    global X_locenkf
    @warn "Localized EnKF failed for Shu-Osher, $(typeof(e))"
end

# %%
order_PA = 8
PA_skip = ceil(Int64, order_PA / 2)
Nsvar = Nxvar - 2 * PA_skip
Ns = Nvar * Nsvar

PA = PolyAnnil(xgrid, order_PA; Nvar=Nvar, istruncated=true)

@assert size(PA.P) == (Ns, Nx)

S = LinearMaps.LinearMap(PA.P / Nvar)
xs = xgrid[PA_skip+1:end-PA_skip];

idxρs = idxρ[PA_skip+1:end-PA_skip]
idxvs = idxv[PA_skip+1:end-PA_skip]
idxps = idxp[PA_skip+1:end-PA_skip];

# %%
# Selection of hyper-prior parameters
# power parameter
hyperprior_idx = 4
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
theta_init_vec = fill(theta_init, Ns)
Cθ = LinearMap(Diagonal(theta_init_vec))
sys_ys = ObsConstraintSystem(H, S, Cθ, Cϵ)

# %%
ϵy = AdditiveInflation(Ny, zeros(Ny), sigma_y);
hlocenkf = HLocEnKF(filtering_fcn!, Ne, ϵy, sys_ys, Loc, dist, theta_init_vec, delta_t_dyn, delta_t_obs; Niter, θinit=theta_init, isfiltered=false)

# %%
X_hlocenkf, θ_hlocenkf = seqassim_trixi(data, 10, filter_inflation, hlocenkf, copy(X0), model.Ny, model.Nx, t0, sys_euler; ode_solver, cfl)

# %%
local X_hlocenkf, θ_hlocenkf
@info "Performing GSBL EnKF..."
try
    global X_hlocenkf, θ_hlocenkf
    X_hlocenkf, θ_hlocenkf = seqassim_trixi(data, Tf, filter_inflation, hlocenkf, copy(X0), model.Ny, model.Nx, t0, sys_euler; ode_solver, cfl)
catch e
    global X_hlocenkf, θ_hlocenkf
    @warn "GSBL localized EnKF failed for Shu-Osher, $(typeof(e))"
end

# %%
begin
    mesh_weights_state = vec(sys_euler.mesh.md.wJq)
    mesh_weights = repeat(mesh_weights_state, nvariables(equations))
    calc_moments = (ensemble, moment) -> weight_sum_reduction.(eachcol(ensemble), (moment,), (mesh_weights,))
    weighted_norm1 = (x, w) -> sum(dim_idx -> w[dim_idx] * abs(x[dim_idx]), eachindex(x, w))
    weighted_norm2 = (x, w) -> sqrt(sum(dim_idx -> w[dim_idx] * abs2(x[dim_idx]), eachindex(x, w)))
    rel_norms1 = calc_moments(data.xt, abs)# map(Base.Fix2(weighted_norm1, mesh_weights), eachcol(data.xt))
    rel_norms2 = map(Base.Fix2(weighted_norm2, mesh_weights), eachcol(data.xt))
    get_errs = (X, metric) -> map(j -> CRPS(X[j+1], @view(data.xt[:, j]), metric, mesh_weights), axes(data.xt, 2))
    get_Lp = (err, rel_norms, prop::Symbol) -> mean(er -> getproperty(er[1], prop) / er[2], zip(err, rel_norms))
    euler_entropy = u -> Trixi.entropy(u, equations)
    euler_qoi_member = (u, fcn) -> mesh_weights_state' * fcn.(eachrow(reshape(u, :, nvariables(equations))), (equations,))
    euler_qoi_ens = (u_ens, fcn) -> euler_qoi_member.(eachcol(u_ens), fcn)
    mass_true, entropy_true = euler_qoi_ens(data.xt, Trixi.density), euler_qoi_ens(data.xt, Trixi.entropy)
    TV_norm_state = u -> sum(abs, diff(reshape(u, :, nvariables(equations)), dims=1))
    TV_norm_ensemble = u_ens -> TV_norm_state.(eachcol(u_ens))
    TVN_true = TV_norm_ensemble(data.xt)
    metrics_locenkf, metrics_hlocenkf = Dict{Symbol,Any}(), Dict{Symbol,Any}()
end

# %%
for alg_name in ["locenkf", "hlocenkf"]
    # Error metrics
    metric_sym = Symbol("metrics_$alg_name")
    metric_dict = @eval($metric_sym)
    X_sym = Symbol("X_$alg_name")
    eval(Expr(:isdefined, X_sym)) || continue
    X_traj = @eval($X_sym)
    isnothing(X_traj) && continue
    for which_norm in [1, 2]
        norm = Symbol("norm$which_norm")
        errs = get_errs(X_traj, norm)
        rel_norms_sym = Symbol("rel_norms$which_norm")
        rel_norms = @eval($rel_norms_sym)
        for metric in [:rmse, :crps]
            metric_sym = Symbol(string(metric) * string(which_norm) * "_" * alg_name)
            metric_dict[metric_sym] = get_Lp(errs, rel_norms, metric)
        end
    end

    # TV, Mass, Entropy
    tv_alg = reduce(hcat, TV_norm_ensemble(x) for x in X_traj)
    mass_alg, entropy_alg = map(f -> reduce(hcat, euler_qoi_ens(x, f) for x in X_traj), [Trixi.density, Trixi.entropy])
    metric_dict[:mass] = mass_alg
    metric_dict[:entropy] = entropy_alg
    metric_dict[:tv_norm] = tv_alg
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

    filter_group = JLD2.Group(file, "filters")
    metric_group = JLD2.Group(file, "metrics")
    metric_group["true_mass"] = mass_true
    metric_group["true_entropy"] = entropy_true
    metric_group["true_tv_norm"] = TVN_true
    for alg in ["locenkf", "hlocenkf"]
        X_alg = Symbol("X_$alg")
        eval(Expr(:isdefined, X_alg)) || continue
        X_traj = @eval($X_alg)
        isnothing(X_traj) && continue
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
make_figs && with_theme(my_theme) do
    fig = Figure(size=(1010, 500))
    tsnap = length(X_hlocenkf) - 1#minimum(length.([X_hlocenkf, X_locenkf]) .- 1)
    ax1 = Axis(fig[1, 1])
    ax2 = Axis(fig[1, 2])
    plot_idx, plot_idx_y = idxv, idxvy
    lines!(ax1, xgrid, data.xt[plot_idx, tsnap], linewidth=3, label="Truth")
    lines!(ax2, xgrid, data.xt[plot_idx, tsnap], linewidth=3, label="Truth")

    cols = Makie.wong_colors()
    for j in 1:Ne
        col = cols[mod1(j, length(cols))]
        # lines!(ax1, xgrid, X_locenkf[tsnap+1][plot_idx, j], linewidth=0.8, label=ifelse(j == 1, "Loc-EnKF", nothing), color=(col, 0.2))
        lines!(ax2, xgrid, X_hlocenkf[tsnap+1][plot_idx, j], linewidth=0.8, label=ifelse(j == 1, "GSBL-EnKF", nothing), color=(col, 0.2))
    end
    # lines!(ax, xgrid, mean(X_hlocenkf[tsnap+1]; dims = 2)[idxρ,1], linewidth = 3, label = "HLocEnKF")
    # ax2 = Axis(fig[1,2])
    # fig[1, 2] = Legend(fig, ax, "Filters", framevisible = false)
    # lines!(ax, xgrid[1:2:end], data.yt[idxpy,tsnap], linewidth = 3)
    # scatter!(ax1, xgrid[1:delta_y:end], data.yt[plot_idx_y, tsnap], markersize=18, label="Observations")
    # scatter!(ax2, xgrid[1:delta_y:end], data.yt[plot_idx_y, tsnap], markersize=18, label="Observations")
    lines!(ax2, xs, θ_hlocenkf[tsnap+1][3:3:end])
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
make_figs && with_theme(my_theme) do
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
