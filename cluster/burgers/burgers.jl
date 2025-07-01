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
Pkg.activate("../..")

# %%
using Revise
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

# %%
make_figs = false
my_theme = Theme()

# %%
# Problem setup params
polydeg = 7 # Order in space
Ncells = 100 # Number of DG cells
Δy = 80 # Spatial frequency of observation. Not regularly spaced
Δtdyn = 0.005 # Timestep for PDE dynamics
Δtobs = 0.025 # Amount of time between each observation

σx_data = 1e-3 # Noise in the state dynamics (i.e., the PDE solution itself)
σy = 0.15 # Noise in the state observation (i.e., what the "sensors" record)

t0, tf = 0.0, 1.0 # Start and end time

# %%
Nx = (polydeg + 1) * Ncells
Ny = ceil(Int64, Nx / Δy)
Tf = round(Int, tf / Δtobs)

# Define Trixi system for inviscid Burgers equation
sys_burgers = setup_burgers(polydeg, Ncells);

xgrid = vec(sys_burgers.mesh.md.xq);

# %%
π0 = MvNormal(zeros(Nx), Matrix(1.0 * I, Nx, Nx))

# %%
ϵx_data = AdditiveInflation(Nx, zeros(Nx), σx_data)
ϵy = AdditiveInflation(Ny, zeros(Ny), σy);

# %%
h(x, t) = x[1:Δy:end]
H = LinearMap(sparse(Matrix(1.0 * I, Nx, Nx)[1:Δy:end, :]))
F = StateSpace(x -> x, h)

# %%
model = Model(Nx, Ny, Δtdyn, Δtobs, ϵx_data, ϵy, π0, 0, 0, 0, F);

# %%
x0 = vec(1 / 2 .+ 0.5 * sin.(3 * π * sys_burgers.mesh.md.xq));

# %%
@time data = generate_data_trixi(model, x0, Tf, sys_burgers)

# %%
make_figs && heatmap(Δtobs * (1:Tf), xgrid, data.xt', axis=(; xlabel=L"t", ylabel=L"x", title=L"Solution of inviscid burgers, $u(x,t)$"))

# %%
make_figs && with_theme(my_theme) do
    fig = Figure()
    ax = Axis(fig[1, 1])
    
    lines!(ax, xgrid, data.xt[:, 1])
    lines!(ax, xgrid, data.xt[:, end])
    scatter!(ax, xgrid[1:Δy:end], data.yt[:, end])
    
    fig
end

# %%
# Important parameters for data assimilation
Ne = 40 # Ensemble size
Lrad = 10 # Localization radius
σx_filter = 0.05 # State noise
β = 1.02 # Inflation param

# %%
# Define function class for the initial condition
αk = 0.7
f0 = SmoothPeriodic(xgrid, αk; L=1.0);
X = zeros(model.Ny + model.Nx, Ne)

for i = 1:Ne
    regenerate!(f0)
    X[Ny+1:Ny+Nx, i] = f0.(xgrid) / 3 .+ 0.5#initial_condition(αk, Δx, Nx)
end

# %%
Cϵ = LinearMap(ϵy.Σ)
CX = LinearMap(Diagonal(1.0 .+ rand(Nx)))
sys_y = ObsSystem(H, Cϵ, CX);

# %%
yidx = 1:Δy:Nx
idx = vcat(collect(1:length(yidx))', collect(yidx)')

# Create Localization structure
Gxx(i, j) = periodicmetric!(i, j, Nx)
Gxy(i, j) = periodicmetric!(i, yidx[j], Nx)
Gyy(i, j) = periodicmetric!(yidx[i], yidx[j], Nx)

Loc = Localization(Lrad, Gxx, Gxy, Gxx)
ϵxβ_filter = MultiAddInflation(Nx, β, zeros(Nx), σx_filter)

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
locenkf = LocEnKF(Ne, ϵy, sys_y, Loc, Δtdyn, Δtobs)

# %%
X_locenkf = seqassim_trixi(data, Tf, ϵxβ_filter, locenkf, deepcopy(X), model.Ny, model.Nx, t0, sys_burgers);

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
    scatter!(ax1, xgrid[1:Δy:end], y_tsnap)

    axislegend(ax1)


    framerate = 10
    timestamps = range(t_start, Tf, step=1)

    anim = Makie.Record(fig, timestamps; framerate=framerate) do t
        tsnap[] = t
    end
    save("figs/assim_lenkf.mp4", anim)
    anim
end

# %%
# GSBL Hyperparams
order_PA = 3 # Poly annihilator order
idx = 3
Niter = 5
θinit = 1.

## Selecion of hyper-prior parameters
# power parameter
r_range = [1.0, 0.5, -0.5, -1.0];
r = r_range[idx] # select parameter 
# shape parameter
β_range = [1.501, 3.0918, 2.0165, 1.0017];
β = β_range[idx] # shape parameter
# rate parameters 
ϑ_range = [5 * 10^(-2), 5.9323 * 10^(-3), 1.2583 * 10^(-3), 1.2308 * 10^(-4)];
ϑ = ϑ_range[idx]

dist = GeneralizedGamma(r, β, ϑ);

# %%
PA_offset = ceil(Int64, order_PA / 2)
Ns = Nx - 2 * PA_offset

PA = PolyAnnil(xgrid, order_PA; istruncated=true)
@assert size(PA.P) == (Ns, Nx)

S = LinearMaps.FunctionMap{Float64,true}((s, x) -> mul!(s, PA.P, x), (x, s) -> mul!(x, PA.P', s),
    Ns, Nx; issymmetric=false, isposdef=false)

θinit_vec = fill(θinit, Ns)
Cθ = LinearMap(Diagonal(θinit_vec))
sys_ys = ObsConstraintSystem(H, S, Cθ, Cϵ, CX);

# %%
hlocenkf = HLocEnKF(Ne, ϵy, sys_ys, Loc, dist, θinit_vec, Δtdyn, Δtobs; Niter, θinit)

# %%
X_hlocenkf, θ_hlocenkf = seqassim_trixi(data, Tf, ϵxβ_filter, hlocenkf, deepcopy(X), model.Ny, model.Nx, t0, sys_burgers);

# %%
make_figs && with_theme(my_theme) do
    t_start = 1
    tsnap = Observable(t_start)
    x_tsnap = @lift(data.xt[:, $tsnap])
    y_tsnap = @lift(data.yt[:, $tsnap])
    ut = t -> map(x -> x[], vec(data.xt[:, round(Int, t / Δtobs)]))
    ys = @lift(ut(($tsnap) * Δtobs))
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
    scatter!(ax1, xgrid[1:Δy:end], y_tsnap)

    axislegend(ax1)


    framerate = 10
    timestamps = range(t_start, Tf, step=1)

    anim = Makie.Record(fig, timestamps; framerate=framerate) do t
        tsnap[] = t
    end
    save("figs/assim_hlenkf.mp4", anim)
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
    
    save("figs/heatmap_inviscid_burgers.png", fig)
    
    fig
end;

# %%
mesh_weights = vec(sys_burgers.mesh.md.wJq);

# %%
weighted_norm = (x,w)-> sqrt( sum( dim_idx->w[dim_idx]*abs2(x[dim_idx]), eachindex(x,w) ) )
rel_norms = map(Base.Fix2(weighted_norm, mesh_weights), eachcol(data.xt))

errs_locenkf2 = map(j -> CRPS(X_locenkf[j+1], @view(data.xt[:, j]), :norm2, mesh_weights), axes(data.xt, 2))
errs_hlocenkf2 = map(j -> CRPS(X_hlocenkf[j+1], @view(data.xt[:, j]), :norm2, mesh_weights), axes(data.xt, 2))

rmse_locenkf, rmse_hlocenkf = [mean(i -> err[i].rmse / rel_norms[i], eachindex(err,rel_norms)) for err in [errs_locenkf2, errs_hlocenkf2]]
crps_locenkf, crps_hlocenkf = [mean(i -> err[i].crps / rel_norms[i], eachindex(err,rel_norms)) for err in [errs_locenkf2, errs_hlocenkf2]]
make_figs && @info "2-Norm results" rmse_locenkf crps_locenkf "======================" rmse_hlocenkf crps_hlocenkf;

# %%
weighted_norm = (x,w)-> sum( dim_idx->w[dim_idx]*abs(x[dim_idx]), eachindex(x,w) )
rel_norms = map(Base.Fix2(weighted_norm, mesh_weights), eachcol(data.xt))

errs_locenkf2 = map(j -> CRPS(X_locenkf[j+1], @view(data.xt[:, j]), :norm1, mesh_weights), axes(data.xt, 2))
errs_hlocenkf2 = map(j -> CRPS(X_hlocenkf[j+1], @view(data.xt[:, j]), :norm1, mesh_weights), axes(data.xt, 2))

rmse_locenkf, rmse_hlocenkf = [mean(i -> err[i].rmse / rel_norms[i], eachindex(err,rel_norms)) for err in [errs_locenkf2, errs_hlocenkf2]]
crps_locenkf, crps_hlocenkf = [mean(i -> err[i].crps / rel_norms[i], eachindex(err,rel_norms)) for err in [errs_locenkf2, errs_hlocenkf2]]
make_figs && @info "1-Norm results" rmse_locenkf crps_locenkf "======================" rmse_hlocenkf crps_hlocenkf;

# %%
mass_true, energy_true = [weight_sum_reduction.(eachcol(data.xt), fcn, (mesh_weights,)) for fcn in (abs, abs2)]
mass_locenkf, energy_locenkf = [reduce(hcat, weight_sum_reduction.(eachcol(x), fcn, (mesh_weights,)) for x in X_locenkf) for fcn in (abs, abs2)]
mass_hlocenkf, energy_hlocenkf = [reduce(hcat, weight_sum_reduction.(eachcol(x), fcn, (mesh_weights,)) for x in X_hlocenkf) for fcn in (abs, abs2)]
mass_err_locenkf, energy_err_locenkf = [mean(t_idx -> abs(mean(enkf[:,t_idx+1]) - truth[t_idx])/truth[t_idx], eachindex(truth)) for (truth, enkf) in [(mass_true, mass_locenkf), (energy_true, energy_locenkf)]]
mass_err_hlocenkf, energy_err_hlocenkf = [mean(t_idx -> abs(mean(enkf[:,t_idx+1]) - truth[t_idx])/truth[t_idx], eachindex(truth)) for (truth, enkf) in [(mass_true, mass_hlocenkf), (energy_true, energy_hlocenkf)]]
make_figs && @info "Summary stat results" mass_err_locenkf energy_err_locenkf "======================" mass_err_hlocenkf energy_err_hlocenkf;

# %%
