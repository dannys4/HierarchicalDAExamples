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
using Trixi
using LinearAlgebra
using OrdinaryDiffEq
using CairoMakie
using HierarchicalDA
using StaticArrays
using LinearMaps
using TransportBasedInference2
using Distributions
using SparseArrays

using Trixi: entropy2cons

# %%
my_theme = Theme()

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
# PDE solution parameters
polydeg = 6
advection_velocity = 0.1

N_elements = 100
coordinates_min, coordinates_max = -1., 1.

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

mesh = DGMultiMesh(solver, (N_elements,); periodicity=true, coordinates_min, coordinates_max)
initial_condition = initial_condition_sawtooth

semi = SemidiscretizationHyperbolic(mesh,
    equations,
    initial_condition,
    solver, boundary_conditions=boundary_condition_periodic)
xgrid = vec(mesh.md.xq);

# %%
# Data assimilation setup
Δy = 50
Nx = length(xgrid)
Ny = ceil(Int64, Nx / Δy)
Ne = 50
Δtdyn = 0.05
Δtobs = 0.25
t0, tf = 0.0, 25.0
σx_data = 1e-5
σy = 0.2

# %%
Tf = Int((tf - t0) / Δtobs)
π0 = MvNormal(zeros(Nx), Matrix(1.0 * I, Nx, Nx))

# %%
h(x, t) = x[1:Δy:end]
H = LinearMap(sparse(Matrix(1.0 * I, Nx, Nx)[1:Δy:end, :]))
F = StateSpace(x -> x, h)
sys_advection = TrixiSystem(equations, solver, mesh, semi)
ϵx_true = AdditiveInflation(Nx, zeros(Nx), σx_data)
ϵy = AdditiveInflation(Ny, zeros(Ny), σy)
model = Model(Nx, Ny, Δtdyn, Δtobs, ϵx_true, ϵy, π0, 0, 0, 0, F)
f0 = SmoothPeriodic(xgrid, 0.8)

# %%
u0 = initial_condition_sawtooth_fcn.(xgrid, (0,))
data = generate_data_trixi(model, u0, Tf, sys_advection)

# %%
X0 = zeros(model.Ny + model.Nx, Ne)
for i = 1:Ne
    regenerate!(f0)
    X0[Ny+1:Ny+Nx, i] = 0.5*(f0.(xgrid) .+ 1.)
end

# %%
β_infl = 1.02
σx_filter = 0.15

Lrad = 7

# %%
Nx = length(xgrid)
yidx = 1:Δy:Nx

# # Create Localization structure
Gxx(i, j) = periodicmetric!(i, j, Nx)
Gxy(i, j) = periodicmetric!(i, yidx[j], Nx)
Gyy(i, j) = periodicmetric!(yidx[i], yidx[j], Nx)

Loc = Localization(Lrad, Gxx, Gxy, Gxx)
ϵxβ_enkf = MultiAddInflation(Nx, β_infl, zeros(Nx), σx_filter)

# %%
Cϵ = LinearMap(ϵy.Σ)
# This CX is replaced with the estimated state cov at each step
CX = LinearMap(I(Nx))

# %%
sys_y = ObsSystem(H, Cϵ, CX)
Lenkf = LocEnKF(Ne, ϵy, sys_y, Loc, Δtdyn, Δtobs)

X_locenkf = seqassim_trixi(data, Tf, ϵxβ_enkf, Lenkf, deepcopy(X0), model.Ny, model.Nx, t0, sys_advection);

# %%
with_theme(my_theme) do
    t_start = 3
    tsnap = Observable(t_start)
    cols = Makie.wong_colors()

    x_tsnap = @lift(data.xt[:, $tsnap])
    x_tsnap_plus = @lift(data.xt[:, $tsnap] .+ σx_filter)
    y_tsnap = @lift(data.yt[:, $tsnap])
    X_Lenkf_tsnap = @lift(vec(mean(X_Lenkf[$tsnap+1]; dims=2)))
    # X_Lens_tsnap = [@lift(X_Lenkf[$tsnap+1][:, j]) for j in 1:Ne]

    fig = Figure()

    ax1 = Axis(fig[1, 1], title="Unadjusted EnKF")

    lines!(ax1, xgrid, x_tsnap, linewidth=3, label="Truth")
    lines!(ax1, xgrid, x_tsnap_plus, linewidth=3, label="Truth+σ")
    lines!(ax1, xgrid, X_Lenkf_tsnap, linewidth=3, label="EnKF")
    # for j in 1:Ne
    #     lines!(ax1, xgrid, X_Lens_tsnap[j], linewidth=0.9, color=(cols[1+(j%length(cols))], 0.2))
    # end
    scatter!(ax1, xgrid[1:Δy:end], y_tsnap)

    axislegend(ax1)


    framerate = 10
    timestamps = range(t_start, Tf, step=1)

    anim = CairoMakie.Makie.Record(fig, timestamps; framerate=framerate) do t
        tsnap[] = t
    end
    save("figs/assim_enkf.mp4", anim)
    anim
end

# %%
order_PA = 3
GSBL_idx = 4
θinit = 1.
Niter=5

# %%
## Selecion of hyper-prior parameters
# power parameter
r_range = [1.0, 0.5, -0.5, -1.0];
r = r_range[GSBL_idx] # select parameter 
# shape parameter
β_range = [1.501, 3.0918, 2.0165, 1.0017];
β_dist = β_range[GSBL_idx] # shape parameter
# rate parameters 
ϑ_range = [5 * 10^(-2), 5.9323 * 10^(-3), 1.2583 * 10^(-3), 1.2308 * 10^(-4)];
ϑ = ϑ_range[GSBL_idx]

dist = GeneralizedGamma(r, β_dist, ϑ);

# %%
PA_offset = ceil(Int, order_PA / 2)
Ns = Nx - 2PA_offset
PA = PolyAnnil(xgrid, order_PA; istruncated=true)
S = LinearMaps.FunctionMap{Float64,true}((s, x) -> mul!(s, PA.P, x), (x, s) -> mul!(x, PA.P', s), Ns, Nx; issymmetric=false, isposdef=false)
xgrid_S = xgrid[PA_offset+1:end-PA_offset];

# %%
θinit_vec = fill(θinit, Ns)
Cθ = LinearMap(Diagonal(θinit_vec))
sys_ys = ObsConstraintSystem(H, S, Cθ, Cϵ, CX)

# %%
hlocenkf = HLocEnKF(Ne, ϵy, sys_ys, Loc, dist, θinit_vec, Δtdyn, Δtobs; Niter, θinit)

# %%
X_hlocenkf, θhist = seqassim_trixi(data, Tf, ϵxβ_enkf, hlocenkf, deepcopy(X0), model.Ny, model.Nx, t0, sys_advection);

# %%
with_theme(my_theme) do
    t_start = 4
    tsnap = Observable(t_start)
    x_tsnap = @lift(data.xt[:, $tsnap])
    y_tsnap = @lift(data.yt[:, $tsnap])
    X_hlocenkf_tsnap = @lift(vec(mean(X_hlocenkf[$tsnap+1]; dims=2)))
    X_ens_tsnap = [@lift(X_hlocenkf[$tsnap+1][:, j]) for j in 1:Ne]
    theta_tsnap = @lift(θhist[$tsnap+1])
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
    scatter!(ax1, xgrid[1:Δy:end], y_tsnap)

    axislegend(ax1)


    framerate = 10
    timestamps = range(t_start, Tf, step=1)

    anim = CairoMakie.Makie.Record(fig, timestamps; framerate=framerate) do t
        tsnap[] = t
    end
    save("figs/assim_hlenkf.mp4", anim)
    anim
end

# %%
errs_locenkf2 = map(j -> CRPS(X_locenkf[j+1], @view(data.xt[:, j]), :norm2), axes(data.xt, 2))
errs_hlocenkf2 = map(j -> CRPS(X_hlocenkf[j+1], @view(data.xt[:, j]), :norm2), axes(data.xt, 2))
rel_norms = map(norm, eachcol(data.xt))
rmse_locenkf, rmse_hlocenkf = [mean(i -> err[i].rmse / rel_norms[i], eachindex(err)) for err in [errs_locenkf2, errs_hlocenkf2]]
crps_locenkf, crps_hlocenkf = [mean(i -> err[i].crps / rel_norms[i], eachindex(err)) for err in [errs_locenkf2, errs_hlocenkf2]]
@info "2-Norm results" rmse_locenkf crps_locenkf
@info "" rmse_hlocenkf crps_hlocenkf

# %%
mesh_wts = vec(mesh.md.wJq)
mass_true = abs.(data.xt)'mesh_wts
energy_true = abs2.(data.xt)'mesh_wts
avg_mass_locenkf = map(x->mean(xj->mesh_wts'abs.(xj), eachcol(x)), X_locenkf)[2:end]
avg_energy_locenkf = map(x->mean(xj->mesh_wts'abs2.(xj), eachcol(x)), X_locenkf)[2:end]
avg_mass_hlocenkf = map(x->mean(xj->mesh_wts'abs.(xj), eachcol(x)), X_hlocenkf)[2:end]
avg_energy_hlocenkf = map(x->mean(xj->mesh_wts'abs2.(xj), eachcol(x)), X_hlocenkf)[2:end]
mass_err_locenkf = norm(mass_true - avg_mass_locenkf)/norm(mass_true)
mass_err_hlocenkf = norm(mass_true - avg_mass_hlocenkf)/norm(mass_true)
energy_err_locenkf = norm(energy_true - avg_energy_locenkf)/norm(energy_true)
energy_err_hlocenkf = norm(energy_true - avg_energy_hlocenkf)/norm(energy_true)
@show mass_err_locenkf energy_err_locenkf mass_err_hlocenkf energy_err_hlocenkf

# %%
map(x->mean(xj->mesh_wts'abs.(xj), eachcol(x)), X_hlocenkf)

# %%

# %%
