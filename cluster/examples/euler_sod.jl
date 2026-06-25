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
polydeg = 2
Ncells = 100
Nvar = 3

delta_t_dyn = 5e-4
delta_t_obs = 0.025

t0 = 0.0
tf = 0.2

delta_y = 10
density_thresh, pressure_thresh = 5e-5, 5e-5
sigma_y = 0.05
sigma_x_data = 0.0
use_positivity_transform = true

# %% [markdown]
# ### EnKF Parameters

# %%
alpha_k_f0, L_f0 = 1.0, 10.0
initial_noise_perturb = 0.075
sigma_x_filter = 0.0
beta_infl = 1.02
wave_speed = 13.912
Lrad = 0.1 # wave_speed * delta_t_obs
Ne = 40
cfl = 0.4

# %% [markdown]
# ### GSBL parameters

# %%
# order_PA = 3
# hyperprior_idx = 1
forecast_scale_gsbl = 3.0
theta_init = 1.0
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

# %%
begin # Execute all loads as part of one expr
    using Trixi
    using StaticArrays
    using LinearAlgebra
    using OrdinaryDiffEq
    using HierarchicalDA
    using HierarchicalDA: initial_condition_sod
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
sys_euler = setup_euler(polydeg, Ncells, initial_condition=:sod, bcs=nothing)

xgrid = GridFromMesh(sys_euler)
y_obs_idx = (delta_y ÷ 2):delta_y:length(xgrid)
ygrid = xgrid[y_obs_idx]

# %%
idxρ = 3 * ((1:length(xgrid)) .- 1) .+ 1
idxv = 3 * ((1:length(xgrid)) .- 1) .+ 2
idxp = 3 * ((1:length(xgrid)) .- 1) .+ 3

idxρy_xgrid = idxρ[y_obs_idx]
idxvy_xgrid = idxv[y_obs_idx]
idxpy_xgrid = idxp[y_obs_idx]

idxρy_ygrid = 3 * ((1:length(ygrid)) .- 1) .+ 1
idxvy_ygrid = 3 * ((1:length(ygrid)) .- 1) .+ 2
idxpy_ygrid = 3 * ((1:length(ygrid)) .- 1) .+ 3

# %%
pos_vars = ["rho", "p"]
pos_var_flags = collect(in.(Trixi.varnames(cons2prim, equations), (pos_vars,)))

to_solver_transforms = ntuple(Returns(identity), Nvar)
from_solver_transforms = ntuple(Returns(identity), Nvar)
ode_transforms = (;
        to_solver_transform = identity,
        from_solver_transform = identity
    )
if use_positivity_transform
    to_solver_transforms = tuple([identity, exp][pos_var_flags .+ 1]...)
    from_solver_transforms = tuple([identity, log][pos_var_flags .+ 1]...)
    ode_transforms = (;
        to_solver_transform = x->SVector(ntuple(i->to_solver_transforms[i](x[i]), 3)),
        from_solver_transform = x->SVector(ntuple(i->from_solver_transforms[i](x[i]), 3))
    )
end

# %%
Tf = round(Int, (tf - t0) / delta_t_obs)
π0 = MvNormal(zeros(Nx), Matrix(1.0 * I, Nx, Nx))

# %%
ϵx_true = AdditiveInflation(Nx, zeros(Nx), sigma_x_data)
ϵx_filter = AdditiveInflation(Nx, zeros(Nx), sigma_x_filter)

# %%
# Gives me initial condition in cons
problem_setup = SodShock()
x0_quad = map(x -> initial_condition_sod(x, 0., sys_euler.equations, problem_setup), sys_euler.mesh.md.xq)
true_soln_sod!(u, x, t) = sod_solution!(u, x, t, problem_setup)

# x0 is in prims
x0 = sol2vec(x0_quad, sys_euler.equations; g=(x,eqns)->ode_transforms.from_solver_transform(cons2prim(x, eqns)))

# %%
all_idxy = idxpy_xgrid #sort(vcat(idxρy_xgrid, idxvy_xgrid, idxpy_xgrid))
Ny = length(all_idxy)
# sigma_y_offset = 0.01
# sigma_y_offset_scale = 0.05
# ϵy = AdditiveInflation(Ny, zeros(Ny), sigma_y_state .+ sigma_y_offset)
value_transformation = use_positivity_transform ? :log_affine : :affine
# ϵy = RelativeAdditiveInflation(Ny, nothing, Tf, value_transformation; scale=sigma_y_offset_scale, shift=sigma_y_offset)
ϵy = AdditiveInflation(Ny, sigma_y)

h(x, t) = x[all_idxy]
H = SelectionMap(all_idxy, :out, in_size=Nx) # LinearMap(sparse(Matrix(1.0 * I, Nx, Nx)[all_idxy, :]))
# H = sparse(I(Nx)[all_idxy,:])
F = StateSpace(x -> x, h)
model = Model(Nx, Ny, delta_t_dyn, delta_t_obs, ϵx_true, ϵy, π0, 0, 0, 0, F);

# %%
thresholds = (density_thresh, pressure_thresh)
variables = (Trixi.density, Trixi.pressure)
stage_limiter! = PositivityPreservingLimiterZhangShu(
    thresholds=thresholds,
    variables=variables
)
ode_solver = SSPRK43(stage_limiter!)

# %%
@info "Generating data..."
data = generate_data_trixi(model, x0, Tf, sys_euler; ode_transforms, (true_soln!) = true_soln_sod!)

# %%
false && make_figs && with_theme(my_theme) do
    quad_wts = vec(sys_euler.mesh.md.wJq)
    ents = zeros(size(data.xt, 2))
    for (t, x) in enumerate(eachcol(data.xt))
        u_state = reshape(x, 3, :)
        ents[t] = quad_wts' * map(u -> Trixi.entropy(vec(u), sys_euler.equations), eachcol(u_state))
    end
    display(lines(data.tt, ents, axis=(; title="Entropy of Sod shock", xlabel=L"t", ylabel=L"e")))
end

false && make_figs && with_theme(my_theme) do
    fig = Figure()
    ax = Axis(fig[1, 1], title="Initial Condition", ylabel=L"u(0,x)", xlabel=L"x")
    xgrid = 0.:0.01:1.
    u0 = reduce(hcat, initial_condition_sod.(xgrid, (0.,), equations))
    lines!(xgrid, u0[1, :], label=L"\rho")
    lines!(xgrid, u0[2, :], label=L"\rho v_1")
    lines!(xgrid, u0[3, :], label=L"\rho e")
    axislegend()
    save(joinpath(@__DIR__, "figs", "euler", "initial_condition.pdf"))
    display(fig)
end;

x_plot, data_plot = get_plot_ensemble(data.xt, sys_euler; use_cons=false, ode_transforms)

(false && make_figs) && with_theme(my_theme) do
    N_T = length(data.tt)
    p_t = t -> data_plot[:, 1, t]
    ρ_t = t -> data_plot[:, 2, t]
    v_t = t -> data_plot[:, 3, t]

    time = Observable(1)
    ps = @lift(p_t($time))
    ρs = @lift(ρ_t($time))
    vs = @lift(v_t($time))

    fig = Figure()
    title_times = round.(data.tt, digits=2)
    ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"u(t,x)", title=@lift("Shu-Osher, t = $(title_times[$time])"))
    lines!(x_plot, ρs, label=L"\rho", linewidth=3)
    lines!(x_plot, ps, label=L"p", linewidth=3)
    lines!(x_plot, vs, label=L"v", linewidth=3)
    axislegend()
    timestamps = 1:N_T
    anim = CairoMakie.Makie.Record(fig, timestamps; framerate=N_T ÷ 4) do t
        time[] = t
    end
    save(joinpath(@__DIR__, "figs", "euler", "solution.mp4"), anim)
    display(anim)
end

make_figs && with_theme(my_theme) do
    # x_plot, data_plot = xgrid, permutedims(reshape(data.xt, 3, length(xgrid), :), (2,1,3))
    fig = Figure(size=(2100, 700))
    tsnap2 = size(data_plot, 3)
    tsnap2_val = data.tt[tsnap2]
    std_0 = sqrt.(diag(get_cov(ϵy, 0.)))
    std_tsnap2 = sqrt.(diag(get_cov(ϵy, tsnap2_val)))
    for (i, idx_y) in enumerate([idxρy_ygrid, idxvy_ygrid, idxpy_ygrid])
        axi = Axis(fig[1, i])
        lines!(axi, x_plot, data_plot[:, i, 1], linewidth=3)
        lines!(axi, x_plot, data_plot[:, i, tsnap2], linewidth=3)
        # if i in [1, 2, 3]
        #     scatter!(axi, ygrid, data.yt[idx_y, 1], markersize=18)
        #     errorbars!(axi, ygrid, data.yt[idx_y, 1], fill(2sigma_y, length(idx_y)))
        #     scatter!(axi, ygrid, data.yt[idx_y, div(end, 3)], markersize=18)
        #     errorbars!(axi, ygrid, data.yt[idx_y, div(end, 3)], fill(2sigma_y, length(idx_y)))
        # end
        if i == 3
            scatter!(axi, ygrid, data.yt[:, 1], markersize=18)
            errorbars!(axi, ygrid, data.yt[:, 1], 2 * std_0)
            scatter!(axi, ygrid, data.yt[:, tsnap2], markersize=18)
            errorbars!(axi, ygrid, data.yt[:, tsnap2], 2 * std_tsnap2)
        end
    end
    save(joinpath(@__DIR__, "figs", "euler", "time_slices.pdf"), fig)
    display(fig)
end;

# %%
which_initial = :random_shock
f0 = nothing
if which_initial == :linear
    initial_noise_perturb = 0.1
    alpha_k_f0, L_f0 = 1.0, 1.0
    # Define function class for the initial condition
    f0 = SmoothPeriodic(xgrid, alpha_k_f0; L=L_f0, Nvar=Nvar, is_dirichlet=true)
elseif which_initial == :sigmoid
    min_grid, max_grid = extrema(sys_euler.mesh.md.VX)
    from_solver = ode_transforms.from_solver_transform
    levels_L = from_solver([1.0, 0., 1.0])
    levels_R = from_solver([0.125, 0., 0.1])
    scale_dist = LogNormal(0, 1.0)
    f0 = SmoothSigmoid(
        min_grid, max_grid, levels_L, levels_R; shift_mean = 0.5, shift_scale = 0.02, scale_dist
    )
elseif which_initial == :random_shock
    min_grid, max_grid = extrema(sys_euler.mesh.md.VX)
    from_solver = ode_transforms.from_solver_transform
    # if from_solver != identity
    #     throw(ArgumentError("Expected no solver transformation"))
    # end
    levels_L_mean = from_solver([1.0, 0., 1.0])
    levels_L_std = [0.05, 0., 0.05]
    levels_R_mean = from_solver([0.125, 0., 0.1])
    levels_R_std = use_positivity_transform ? [0.05, 0., 0.05] : [0.006, 0., 0.005]
    dist_L = MvNormal(levels_L_mean, levels_L_std)
    dist_R = MvNormal(levels_R_mean, levels_R_std)
    shock_loc_dist = Truncated(Normal(0.5, 0.125), 1e-3, 1 - 1e-3)
    f0 = RandomShockInitialization(dist_L, dist_R, shock_loc_dist)
else
    throw(ArgumentError("Unexpected initial condition $which_initial"))
end

X0 = nothing
if which_initial == :linear
    if use_positivity_transform
        min_grid, max_grid = extrema(sys_euler.mesh.md.VX)
        from_solver = ode_transforms.from_solver_transform
        to_solver = ode_transforms.to_solver_transform
        levels_L = @SVector[1.0, 0., 1.0]
        levels_R = @SVector[0.125, 0., 0.1]
        normalized_coords = (xgrid .- min_grid) / (max_grid - min_grid)
        lin_X0 = reduce(hcat, collect(to_solver((1 - x) * from_solver(levels_L) + x * from_solver(levels_R))) for x in normalized_coords)
        X0 = Array{Float64}(undef, Nvar, Nxvar, Ne)
        for ens_idx in axes(X0, 3)
            X0_idx = @view X0[:,:,ens_idx]
            noise_idx = reshape(f0(xgrid), :, Nvar)'
            noise_idx .*= initial_noise_perturb
            for grid_idx in axes(X0_idx, 2)
                X0_idx[:, grid_idx] .= ode_transforms.from_solver_transform(lin_X0[:, grid_idx] + noise_idx[:, grid_idx])
            end
            regenerate!(f0)
        end
        X0 = reshape(X0, :, Ne)
    else
        xmin, xmax = extrema(sys_euler.mesh.md.VX)
        function initial_ensemble(x, t, eq::CompressibleEulerEquations1D; levels_L=nothing, levels_R=nothing, xmin=0., xmax=1.)
            isnothing(levels_L) && (levels_L = @SVector[1.0, 0., 1.0])
            isnothing(levels_R) && (levels_R = @SVector[0.125, 0., 0.1])
            prims = levels_L + ((x - xmin) / (xmax - xmin)) * (levels_R - levels_L)
            return Vector(prim2cons(prims, eq))
        end

        X0, _ = positivity_preserving_noise1d(f0, initial_ensemble, Ne, sys_euler, pos_vars, 0.1)
    end
elseif which_initial == :sigmoid
    X0 = Array{Float64}(undef, Nx, Ne)
    xmin, xmax = extrema(sys_euler.mesh.md.VX)
    perturb, N_terms = [1e-2, 1e-2, 1e-2], 5
    for ens_idx in axes(X0, 2)
        regenerate!(f0)
        freqs = 1:N_terms
        xgrid_coeff = ((xgrid .- xmin)/(xmax - xmin)) .* (freqs')
        X0_idx = @view X0[:, ens_idx]
        f0(X0_idx, xgrid)
        for var_idx in 1:Nvar
            s_x_coeff = randn(N_terms) ./ (freqs.^alpha_k_f0)
            s_x = sin.(2pi*xgrid_coeff) * s_x_coeff
            X0_var = @view X0_idx[var_idx:Nvar:end]
            X0_var .+= perturb[var_idx] * s_x
            if pos_var_flags[var_idx]
                X0_var .= max.(X0_var, 1e-4)
            end
        end
    end
elseif which_initial == :random_shock
    X0 = Array{Float64}(undef, Nx, Ne)
    xmin, xmax = extrema(sys_euler.mesh.md.VX)
    elems = sys_euler.mesh.md.VX
    for ens_idx in axes(X0, 2)
        X0_idx = @view X0[:, ens_idx]
        regenerate!(f0)
        f0(X0_idx, xgrid)
        # f0.shock_loc[] = elems[findmin(abs2, elems .- f0.shock_loc[])[2]]
        # f0(X0_idx, xgrid)
    end
end

make_figs && with_theme(my_theme) do
    _, X0_plot = get_plot_ensemble(X0, sys_euler)
    fig = Figure(size=(2100, 500))
    axs = map(1:Nvar) do i
        Axis(fig[1, i],
            title=Trixi.varnames(cons2prim, equations)[i],
            # limits=(nothing, nothing, -0.05, 1.25),
            # yticks=0.:0.4:1.2
        )
    end
    num_ens_viz = Ne
    X0_vars = reshape(X0, Nvar, :, size(X0, 2))
    cols = Makie.wong_colors()
    for ens_idx in 1:num_ens_viz
        X_sample = @view X0_vars[:, :, ens_idx]
        col = cols[mod1(ens_idx, length(cols))]
        for (var_idx, idx_y) in enumerate([idxρy_xgrid, idxvy_xgrid, idxpy_xgrid])
            lines!(axs[var_idx], xgrid, X_sample[var_idx, :], linewidth=5, color=(col, 0.2))
            # lines!(axi, xgrid, X_sample[idx_x, div(end, 3)], linewidth=3)
        end
    end
    save(joinpath(@__DIR__, "figs", "euler", "initial_ensemble.pdf"), fig)
    display(fig)
end;

# %%
false && make_figs && with_theme(my_theme) do
    indicator_sc = IndicatorHennemannGassner(
        equations,
        sys_euler.dg.basis,
        alpha_max=0.5,
        alpha_min=0.001,
        alpha_smooth=true,
        variable=density_pressure,
    )
    u_x = x0; #data.xt[:, 1]
    u = vec2sol(u_x, sys_euler)
    cache = Trixi.create_cache(sys_euler.mesh, equations, sys_euler.dg.volume_integral, sys_euler.dg, Float64, Float64)

    alpha = indicator_sc(u, sys_euler.mesh, equations, sys_euler.dg, cache)
    scatter(alpha)
end

# %%
metric = CartesianMetric(Float64)
Loc = Localization(xgrid, Lrad, metric; Nvar, symm_kernel=true, is_sparse=false, herm_matrix=true)
CX = LocalizedEmpiricalCov(X0, Loc)
Cϵ = LinearMap(get_cov(ϵy, 0.))
sys_y = ObsSystem(H, Cϵ, CX)

# Create Localization structure
filter_inflation = MultiAddInflation(Nx, beta_infl, zeros(Nx), sigma_x_filter)

# %%
locenkf = LocEnKF(ϵy, sys_y, Loc, delta_t_dyn, delta_t_obs, isfiltered=false)

# %%
Trixi.TrixiBase.disable_debug_timings()
X_locenkf = seqassim_trixi(data, Tf, filter_inflation, locenkf, copy(X0), model.Ny, model.Nx, t0, sys_euler; ode_solver, ode_transforms, cfl)

# %%
local X_locenkf
@info "Performing EnKF..."
false && try
    global X_locenkf
    X_locenkf = seqassim_trixi(data, Tf, filter_inflation, locenkf, copy(X0), model.Ny, model.Nx, t0, sys_euler; ode_solver, cfl)
catch e
    global X_locenkf
    @warn "Localized EnKF failed for Shu-Osher, $(typeof(e))"
end

# %%
# order_PA = 3
# PA_skip = ceil(Int64, order_PA / 2)
# Nsvar = Nxvar - 2 * PA_skip
# Ns = Nvar * Nsvar
# PA = PolyAnnil(xgrid, order_PA; Nvar=Nvar, istruncated=true)
# @assert size(PA.P) == (Ns, Nx)
# S = LinearMaps.LinearMap(PA.P)
# θgrid = xgrid[PA_skip+1:end-PA_skip];
sqrt_quad_wts = kron(sqrt.(vec(sys_euler.mesh.md.wJq)), ones(Nvar))
diff_map = DGMultiDiff1D(sys_euler, false)
grid_sz = sys_euler.mesh.md.J[1]
diff_mat = Diagonal(sqrt_quad_wts) * sparse(diff_map * (I + diff_map * (I + diff_map)))/3
# diff_mat = sparse(diff_map * diff_map)
edge_cutoff = 0
# S_out_idx = (Nvar * (edge_cutoff+1)):(Nvar * ((size(diff_mat, 1) ÷ Nvar) - edge_cutoff))
S = LinearMap(diff_mat)
θgrid = copy(xgrid)
Ns = size(S,1)

# %%
false && make_figs && with_theme(my_theme) do
    fig = Figure(size=(900, 300))
    f_x = S * vec(reduce(vcat, sin.(xgrid * pi / 5)' for _ in 1:3))
    for start_idx in 1:3
        ax = Axis(fig[1, start_idx])
        lines!(xgrid, f_x[start_idx:3:end], linewidth=3)
        lines!(xgrid, -sin.(xgrid * pi / 5) * ((pi / 5)^2), linestyle=:dash, linewidth=3)
    end
    display(fig)
end


# %%
idxρθ_xgrid = 3 * ((1:length(xgrid)) .- 1) .+ 1
idxvθ_xgrid = 3 * ((1:length(xgrid)) .- 1) .+ 2
idxpθ_xgrid = 3 * ((1:length(xgrid)) .- 1) .+ 3

idxρθ_θgrid = 3 * ((1:length(θgrid)) .- 1) .+ 1
idxvθ_θgrid = 3 * ((1:length(θgrid)) .- 1) .+ 2
idxpθ_θgrid = 3 * ((1:length(θgrid)) .- 1) .+ 3

# %%
# Selection of hyper-prior parameters
hyperprior_idx = 4
# power parameter
r_range = [1.0, 0.5, -0.5, -1.0];
r_GSBL = r_range[hyperprior_idx] # select parameter
beta_shift = is_theta_shared ? Ne : 1
# shape parameter
β_range = [1.001 + beta_shift / 2, 2.5918 + beta_shift / 2, 2.0165, 1.0017];
β_GSBL = β_range[hyperprior_idx] # shape parameter
# rate parameters
ϑ_range = [5 * 10^(-2), 5.9323 * 10^(-3), 1.2583 * 10^(-3), 1.2308 * 10^(-4)];
ϑ_GSBL = ϑ_range[hyperprior_idx]

r_GSBL = 0.5
β_GSBL = 0.05
ϑ_GSBL = 1e-3
dist = GeneralizedGamma(r_GSBL, β_GSBL, ϑ_GSBL);

make_figs && with_theme(my_theme) do
    R = r_GSBL
    VARTHETA = ϑ_GSBL
    fig = Figure()
    ax = Axis(fig[1,1], xscale=log10)
    theta_grid = 1e-5:1e-5:1e-3
    for beta in [0.05, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3, 3.5, 4.0]
        lines!(theta_grid, logpdf(GeneralizedGamma(R, beta, VARTHETA), theta_grid), label=string(beta))
    end
    # axislegend()
    fig
end

# %%
theta_init_vec = fill(theta_init, Ns)
Cθ = LinearMap(Diagonal(theta_init_vec))
theta_init_space = is_theta_shared ? theta_init_vec : repeat(theta_init_vec, 1, Ne)
isiterative = false
sys_ys = nothing
if isiterative
    CX_init = LocalizedEmpiricalCov(X0, Loc; with_matrix=false)
    sys_ys = ObsConstraintSystem(H, S, Cθ, Cϵ, CX_init; cache_matrix=false, isiterative=true)
else
    sys_ys = ObsConstraintSystem(LinearMap(H), S, Cθ, Cϵ)
end

# %%
forecast_scale_gsbl = 1 #0.25
Lrad_gsbl = Lrad #4maximum(diff(xgrid))
Niter = 2
Loc_gsbl = Localization(xgrid, Lrad_gsbl, metric, forecast_scale_gsbl; Nvar, symm_kernel=true, is_sparse=false)
hlocenkf = HLocEnKF(identity, Ne, ϵy, sys_ys, Loc_gsbl, dist, theta_init_space, delta_t_dyn, delta_t_obs; Niter, θinit=theta_init, isiterative, isfiltered=false, cg_tol=1e-3)

# %%
local X_hlocenkf, θ_hlocenkf
@info "Performing GSBL EnKF..."
T_hlocenkf = 3
start_hlocenkf = time()
X_hlocenkf, θ_hlocenkf = seqassim_trixi(data, T_hlocenkf, filter_inflation, hlocenkf, copy(X0), model.Ny, model.Nx, t0, sys_euler; ode_solver, cfl, ode_transforms)
hloc_elaps = time() - start_hlocenkf
@info "GSBL EnKF Took $(hloc_elaps)s"

# %%
false && try
    global X_hlocenkf, θ_hlocenkf
    X_hlocenkf, θ_hlocenkf = seqassim_trixi(data, Tf, filter_inflation, hlocenkf, copy(X0), model.Ny, model.Nx, t0, sys_euler; ode_solver, cfl)
catch e
    global X_hlocenkf, θ_hlocenkf
    @warn "GSBL localized EnKF failed for Shu-Osher, $(typeof(e))"
end

make_figs && with_theme(my_theme) do
    show_theta = true
    tsnap = length(X_hlocenkf) - 1#minimum(length.([X_hlocenkf, X_locenkf]) .- 1)
    postproc_locenkf = map(x -> get_plot_ensemble(x, sys_euler)[2], X_locenkf)
    postproc_hlocenkf = map(x -> get_plot_ensemble(x, sys_euler)[2], X_hlocenkf)
    all_idxs = Dict(
        :ρ => (1, idxρ, idxρy_ygrid, idxρθ_θgrid),
        :v => (2, idxv, idxvy_ygrid, idxvθ_θgrid),
        :p => (3, idxp, idxpy_ygrid, idxpθ_θgrid),
    )
    for which_var in ["ρ", "v", "p"]
        fig = Figure(size=(1010, 500))
        ax1 = Axis(fig[1, 1], title="EnKF $(which_var)")
        ax2 = Axis(fig[1, 2], title="GSBL-EnKF $(which_var)")
        var_idx, plot_idx, plot_idx_y, plot_idx_θ = all_idxs[Symbol(which_var)]
        lines!(ax1, xgrid, data.xt[plot_idx, tsnap], linewidth=3, label="Truth")
        lines!(ax2, xgrid, data.xt[plot_idx, tsnap], linewidth=3, label="Truth")
        hlines!(ax1, 0.)
        hlines!(ax2, 0.)

        cols = Makie.wong_colors()
        for j in 1:Ne
            col = (cols[mod1(j, length(cols))], 0.2)
            lines!(ax1, x_plot, postproc_locenkf[tsnap+1][:, var_idx, j], linewidth=0.8, label=ifelse(j == 1, "Loc-EnKF", nothing), color=col)
            if tsnap < length(postproc_hlocenkf)
                lines!(ax2, x_plot, postproc_hlocenkf[tsnap+1][:, var_idx, j], linewidth=0.8, label=ifelse(j == 1, "GSBL-EnKF", nothing), color=col)
            end
            show_theta && (is_theta_shared || scatter!(ax2, θgrid, -(3 + log(ϑ_GSBL)) .+ log.(θ_hlocenkf[tsnap+1][plot_idx_θ, j]), color=col, markersize=3))
        end
        if which_var == "p"
            scatter!(ax1, ygrid, data.yt[:, tsnap], markersize=18, label="Observations")
            scatter!(ax2, ygrid, data.yt[:, tsnap], markersize=18, label="Observations")
        end
        show_theta && is_theta_shared && scatter!(ax2, θgrid, θ_hlocenkf[tsnap+1][plot_idx_θ], label="θ")
        axislegend(ax1, position=:lc)
        axislegend(ax2, position=:lc)
        save(joinpath(@__DIR__, "figs", "euler", "compare_$(which_var)_t$(tsnap).pdf"))
        display(fig)
    end
    @info "Plotted!"
end

# %%
make_figs && with_theme(my_theme) do
    show_theta = false
    for idx_start in 1:3
        fig = Figure()
        ax = Axis(fig[1,1])
        t_snap = length(X_hlocenkf)
        snap_hlocenkf = @view X_hlocenkf[t_snap][idx_start:Nvar:end,:]
        snap_locenkf = @view X_locenkf[t_snap][idx_start:Nvar:end,:]
        for (j,(X_gsbl_j, X_enkf_j)) in enumerate(zip(eachcol.((snap_hlocenkf, snap_locenkf))...))
            lab1, lab2 = j == 1 ? ("GSBL", "EnKF") : (nothing, nothing)
            lines!(xgrid, X_gsbl_j, color=(:black, 0.5), label=lab1)
            lines!(xgrid, X_enkf_j, color=(:red, 0.5), label=lab2)
            thetas = θ_hlocenkf[t_snap][idx_start:3:end, j]
            thetas /= (2*maximum(thetas))
            show_theta && (is_theta_shared || scatter!(θgrid, thetas, markersize=5))
        end
        data_snap = data.xt[idx_start:Nvar:end, t_snap - 1]
        lines!(xgrid, data_snap, label="Truth", linewidth=3,)
        axislegend(ax)
        display(fig)
    end
    @info "Plotted on quad"
end

# %%
begin
    get_traj_quad_pts = traj -> map(x -> reshape(get_filter_quad_pts(x, sys_euler), Nvar, :, Ne), traj)
    data_quad = reshape(get_filter_quad_pts(data.xt, sys_euler), Nvar, :, Tf)
    mesh_wts = vec(sys_euler.mesh.md.wJq)
    rel_norms1 = sum(j -> mesh_wts[j] * abs.(data_quad[:, j, :]), eachindex(mesh_wts))
    rel_norms2 = sqrt.(sum(j -> mesh_wts[j] * abs2.(data_quad[:, j, :]), eachindex(mesh_wts)))
    get_errs = (X, metric, which_var, start_time) -> map(start_time:size(data.xt,2)) do t_idx
        CRPS(X[t_idx+1][which_var, :, :], @view(data_quad[which_var, :, t_idx]), metric, mesh_wts)
    end
    get_Lp = (err, rel_norms, prop::Symbol) -> mean(inp -> getproperty(inp[1], prop) / inp[2], zip(err, rel_norms))
    # g=(x,eqns)->ode_transforms.from_solver_transform(cons2prim(x, eqns))
    density_state = u -> Trixi.density(prim2cons(ode_transforms.to_solver_transform(u), equations), equations)
    mass_ensemble = u_ens -> map(density_state, eachslice(u_ens, dims=(2, 3)))' * mesh_wts

    entropy_state = u -> Trixi.entropy(prim2cons(ode_transforms.to_solver_transform(u), equations), equations)
    entropy_ensemble = u_ens -> map(entropy_state, eachslice(u_ens, dims=(2, 3)))' * mesh_wts

    TV_norm_state = u -> sum(abs, diff(u))
    TV_norm_ensemble = u_ens -> map(TV_norm_state, eachslice(u_ens, dims=(1, 3)))

    mass_data = mass_ensemble(data_quad)
    entropy_data = entropy_ensemble(data_quad)
    TV_data = TV_norm_ensemble(data_quad)
    metrics_locenkf, metrics_hlocenkf = Dict{Symbol,Any}(), Dict{Symbol,Any}()
end

# %%
function derivative_rmse(diff_op, Nvar, quad_wts, truth, ens)
    diff_truth = collect(diff_op * truth)
    diff_ens = collect(diff_op * ens)
    mse = zeros(Nvar)
    truth_norm = zeros(Nvar)
    for (idx, wt) in enumerate(quad_wts)
        for var_idx in 1:Nvar
            pt_idx = (idx - 1)*Nvar + var_idx
            discrep = abs2.(diff_ens[pt_idx,:] .- diff_truth[pt_idx])
            mse[var_idx] += mean(discrep)*wt
            truth_norm[var_idx] += abs2(diff_truth[pt_idx])*wt
        end
    end
    sqrt.(mse ./ truth_norm)
end

# %%
start_time = 1
for alg_name in ["locenkf", "hlocenkf"]
    # Error metrics
    metric_sym = Symbol("metrics_$alg_name")
    metric_dict = @eval($metric_sym)
    X_sym = Symbol("X_$alg_name")
    eval(Expr(:isdefined, X_sym)) || continue
    X_traj = get_traj_quad_pts(@eval($X_sym))
    isnothing(X_traj) && continue
    for which_norm in [1, 2]
        norm = Symbol("norm$which_norm")
        errs = [get_errs(X_traj, norm, j, start_time) for j in 1:Nvar]
        rel_norms_sym = Symbol("rel_norms$which_norm")
        rel_norms = @eval($rel_norms_sym)
        for metric in [:rmse, :crps]
            metric_sym = Symbol(string(metric) * string(which_norm) * "_" * alg_name)
            metric_dict[metric_sym] = [get_Lp(errs[j], rel_norms[j, :], metric) for j in 1:Nvar]
        end
    end

    # TV, Mass, Entropy
    mass_alg = map(mass_ensemble, X_traj)
    entropy_alg = map(entropy_ensemble, X_traj)
    tv_alg = map(TV_norm_ensemble, X_traj)
    metric_dict[:mass] = mass_alg
    metric_dict[:entropy] = entropy_alg
    metric_dict[:tv_norm] = tv_alg
end
@info "" tuple(metrics_hlocenkf[:crps2_hlocenkf]) tuple(metrics_hlocenkf[:rmse2_hlocenkf]) tuple(metrics_locenkf[:crps2_locenkf]) tuple(metrics_locenkf[:rmse2_locenkf])
@info "" tuple(metrics_hlocenkf[:crps2_hlocenkf] ./ metrics_locenkf[:crps2_locenkf])
# entropy_hlocenkf, entropy_locenkf = map(x->x[end],), map(x->x[end],metrics_locenkf[:entropy])
# @info "" sum(abs2, reduce(hcat, metrics_hlocenkf[:entropy][2:end])' .- entropy_data)
# @info "" sum(abs2, reduce(hcat,  metrics_locenkf[:entropy][2:end])' .- entropy_data)

# %%
jldopen(joinpath(data_path, "sod_" * string(now()) * ".jld2"), "w") do file
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
    for GSBL_param in [:Niter, :theta_init, :dist]
        GSBL_param_group[string(GSBL_param)] = @eval($GSBL_param)
    end

    filter_group = JLD2.Group(file, "filters")
    metric_group = JLD2.Group(file, "metrics")
    metric_group["true_mass"] = mass_data
    metric_group["true_entropy"] = entropy_data
    metric_group["true_tv_norm"] = TV_data
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
    postproc_locenkf = map(x -> get_plot_ensemble(x, sys_euler)[2], X_locenkf)
    cols = Makie.wong_colors()
    N_T = length(data.tt)
    ρ_t = t -> data_plot[:, 1, t]
    v_t = t -> data_plot[:, 2, t]
    p_t = t -> data_plot[:, 3, t]
    Nens_plot = Ne
    locenkf_ρ_t = (t, ens_member) -> @view postproc_locenkf[t+1][:, 1, ens_member]
    locenkf_v_t = (t, ens_member) -> @view postproc_locenkf[t+1][:, 2, ens_member]
    locenkf_p_t = (t, ens_member) -> @view postproc_locenkf[t+1][:, 3, ens_member]
    # Offset each to account for previous offsets in data

    time = Observable(1)
    ps = @lift(p_t($time))
    ρs = @lift(ρ_t($time))
    vs = @lift(v_t($time))
    locenkf_ps = [@lift(locenkf_p_t($time, j)) for j in 1:Nens_plot]
    locenkf_ρs = [@lift(locenkf_ρ_t($time, j)) for j in 1:Nens_plot]
    locenkf_vs = [@lift(locenkf_v_t($time, j)) for j in 1:Nens_plot]

    px_size = 600
    which_plot = ["ρ", "p"]
    which_axis = 0
    fig = Figure(size=(length(which_plot)*px_size, px_size))
    for (j, vv) in enumerate([
        ("ρ", ρs, locenkf_ρs),
        ("v", vs, locenkf_vs),
        ("p", ps, locenkf_ps),
    ])
        which_var, truth, locenkf = vv
        in(which_var, which_plot) || continue
        which_axis += 1
        ax = Axis(fig[1, which_axis], xlabel=L"x", title=which_var, aspect=1.)
        lines!(ax, x_plot, truth, linewidth=3, label="truth", color=cols[1])
        lines!(ax, x_plot, locenkf[1], linewidth=0.8, label="EnKF")
        for ens_idx in 2:Nens_plot
            lines!(ax, x_plot, locenkf[ens_idx], linewidth=0.4)
        end
        axislegend(ax, position=:rt)
    end

    timestamps = 1:N_T
    anim = CairoMakie.Makie.Record(fig, timestamps; framerate=N_T ÷ 4) do t
        time[] = t
    end
    save(joinpath(@__DIR__, "figs", "euler", "loc_enkf_result.mp4"), anim)
    anim
end

# %%
make_figs && with_theme(my_theme) do
    postproc_hlocenkf = map(x -> get_plot_ensemble(x, sys_euler)[2], X_hlocenkf)
    cols = Makie.wong_colors()
    N_T = length(data.tt)
    ρ_t = t -> data_plot[:, 1, t]
    v_t = t -> data_plot[:, 2, t]
    p_t = t -> data_plot[:, 3, t]
    Nens_plot = Ne
    hlocenkf_ρ_t = (t, ens_member) -> @view postproc_hlocenkf[t+1][:, 1, ens_member]
    hlocenkf_v_t = (t, ens_member) -> @view postproc_hlocenkf[t+1][:, 2, ens_member]
    hlocenkf_p_t = (t, ens_member) -> @view postproc_hlocenkf[t+1][:, 3, ens_member]
    hlocenkf_θ_ρ_t = t -> @view θ_hlocenkf[t+1][idxρθ_θgrid]
    hlocenkf_θ_v_t = t -> @view θ_hlocenkf[t+1][idxvθ_θgrid]
    hlocenkf_θ_p_t = t -> @view θ_hlocenkf[t+1][idxpθ_θgrid]
    # Offset each to account for previous offsets in data

    time = Observable(1)
    ps = @lift(p_t($time))
    ρs = @lift(ρ_t($time))
    vs = @lift(v_t($time))
    hlocenkf_ps = [@lift(hlocenkf_p_t($time, j)) for j in 1:Nens_plot]
    hlocenkf_ρs = [@lift(hlocenkf_ρ_t($time, j)) for j in 1:Nens_plot]
    hlocenkf_vs = [@lift(hlocenkf_v_t($time, j)) for j in 1:Nens_plot]
    hlocenkf_θ_ps = @lift(hlocenkf_θ_p_t($time))
    hlocenkf_θ_ρs = @lift(hlocenkf_θ_ρ_t($time))
    hlocenkf_θ_vs = @lift(hlocenkf_θ_v_t($time))

    px_size = 600
    which_plot = ["ρ", "p"]
    which_axis = 0
    fig = Figure(size=(length(which_plot)*px_size, px_size))
    for (j, vv) in enumerate([
        ("ρ", ρs, hlocenkf_ρs, hlocenkf_θ_ρs),
        ("v", vs, hlocenkf_vs, hlocenkf_θ_vs),
        ("p", ps, hlocenkf_ps, hlocenkf_θ_ps)
    ])
        which_var, truth, hlocenkf, hlocenkf_θ_var = vv
        in(which_var, which_plot) || continue
        which_axis += 1
        ax = Axis(fig[1, which_axis], xlabel=L"x", title=which_var, aspect=1.)
        lines!(ax, x_plot, truth, linewidth=3, label="truth", color=cols[1])
        lines!(ax, x_plot, hlocenkf[1], linewidth=0.8, label="GSBL-EnKF")
        for ens_idx in 2:Nens_plot
            lines!(ax, x_plot, hlocenkf[ens_idx], linewidth=0.4)
        end
        # scatter!(ax, θgrid, hlocenkf_θ_var, label="θ_$which_var")
        axislegend(ax, position=:rt)
    end

    timestamps = 1:N_T
    anim = CairoMakie.Makie.Record(fig, timestamps; framerate=N_T ÷ 4) do t
        time[] = t
    end
    save(joinpath(@__DIR__, "figs", "euler", "loc_henkf_result.mp4"), anim)
    anim
end

# %%
make_figs && with_theme(my_theme) do
    fig = Figure(size=(1050, 500))
    entropy_locenkf = reduce(hcat, metrics_locenkf[:entropy])'
    entropy_hlocenkf = reduce(hcat, metrics_hlocenkf[:entropy])'
    ylims = extrema(reduce(hcat, collect(extrema(x)) for x in [entropy_locenkf[2:end,:], entropy_hlocenkf[2:end,:]]))
    ax1 = Axis(fig[1, 1],
        title="Euler Entropy, EnKF",
        aspect=1.,
        xlabel=L"t",
        ylabel="Entropy",
        limits=(t0, tf, ylims...)
    )
    ax2 = Axis(fig[1, 2],
        title="Euler Entropy, GSBL-EnKF",
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
    save(joinpath(@__DIR__, "figs", "euler", "entropy.pdf"), fig)
end