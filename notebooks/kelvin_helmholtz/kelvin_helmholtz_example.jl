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
Pkg.activate(joinpath(@__DIR__, "../.."))
Pkg.precompile()

# %%
using Revise
using LinearAlgebra, Statistics, SparseArrays
using FFTW, Distributions, JLD2
using LinearMaps
using CairoMakie, Trixi, OrdinaryDiffEq
using HierarchicalDA, TransportBasedInference2

# %%
my_theme = Theme()

# %%
Ncells_dim = 16
polydeg = 3
t0, tf = 0., 1.0
order_PA = 3

# %%
Δtobs = 0.05
Δtdyn = 0.05
Δy = 20
Tf = ceil(Int, (tf - t0) / Δtobs)

# %%
Lrad = 10
Ne = 50
σx_filter = 0.02
β_infl = 1.02

# %%

"""
    initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)

A version of the classical Kelvin-Helmholtz instability based on
- Andrés M. Rueda-Ramírez, Gregor J. Gassner (2021)
  A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations
  of the Euler Equations
  [arXiv: 2102.06017](https://arxiv.org/abs/2102.06017)
"""
function initial_condition_kelvin_helmholtz_instability(x, t,
    equations::CompressibleEulerEquations2D)
    # change discontinuity to tanh
    # typical resolution 128^2, 256^2
    # domain size is [-1,+1]^2
    slope = 15
    amplitude = 0.02
    B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
    rho = 0.5 + 0.75 * B
    v1 = 0.5 * (B - 1)
    v2 = 0.1 * sin(2 * pi * x[1])
    p = 1.0
    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_kelvin_helmholtz_instability
cells_per_dimension = (Ncells_dim, Ncells_dim)

dg = DGMulti(polydeg=polydeg, element_type=Quad(), approximation_type=Polynomial(),
    surface_integral=SurfaceIntegralWeakForm(FluxLaxFriedrichs()),
    volume_integral=VolumeIntegralFluxDifferencing(flux_ranocha))

equations = CompressibleEulerEquations2D(1.4)
mesh = DGMultiMesh(dg, cells_per_dimension; periodicity=true)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)
sys_euler = TrixiSystem(equations, dg, mesh, semi)

# %%
Nvar = nvariables(equations)
# PA_offset = ceil(Int64, order_PA / 2)
PA = VerticalPolyAnnil2D(mesh, order_PA, Nvar)
S = LinearMap(PA.P)
H = create_observation_operator(
    mesh, polydeg + 1, offset=(polydeg + 1) ÷ 2,
    Nvar=Nvar
)
Nx, Ny = size(PA.P, 1), size(H, 1)
Nx_var = Nx ÷ Nvar

# %%
π0 = MvNormal(I(Nx))

# %%
σx_data = 0.0#Δtobs*1.0
σx_filter = 0.05#copy(σx_true)
@show σx_data σx_filter

σy = 0.15

ϵx_data = AdditiveInflation(Nx, σx_data)
ϵx_filter = AdditiveInflation(Nx, σx_filter)

ϵy = AdditiveInflation(Ny, σy);

# %%
h(x, t) = H * x
F = StateSpace(identity, h)
model = Model(Nx, Ny, Δtdyn, Δtobs, ϵx_data, ϵy, π0, 0, 0, 0, F)

# %%
thresholds = (5e-6, 5e-6)
variables = (Trixi.density, Trixi.pressure)
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=thresholds,
    variables=variables)
ode_solver = SSPRK43(stage_limiter!)

# %%
# Test sol2vec
x0_sol = map(xy -> initial_condition_kelvin_helmholtz_instability(xy, 0., equations), zip(mesh.md.xyzq...))
x0 = sol2vec(x0_sol, equations)

# %%
data = generate_data_trixi(model, x0, Tf, sys_euler; ode_solver, cfl=0.9, record_first=false)

# %%
L_f0 = 2.0
alpha_f0 = 0.75
grid1d = unique(x -> round(x, digits=5), mesh.md.yq)
f0_row = SmoothPeriodic(grid1d, alpha_f0; L=L_f0, Nvar)
f0_col = SmoothPeriodic(grid1d, alpha_f0; L=L_f0, Nvar)
# Need prim vars ρ and p to be positive
pos_vars = ["rho", "p"]
transform_fcn = [var in pos_vars ? exp : identity for var in Trixi.varnames(cons2prim, equations)]
x0_ens = reduce(hcat, HierarchicalDA.sample_initial_state(f0_row, f0_col, mesh; Nvar, transform_fcn) for _ in 1:Ne)
noise_level_t0 = 0.05
for c_idx in CartesianIndices(x0_ens)
    (state_idx, ens_idx) = Tuple(c_idx)
    x0_ens[c_idx] = muladd(noise_level_t0, x0_ens[c_idx], (1 - noise_level_t0) * x0[state_idx])
end

# %%
Loc = Localization(mesh, Lrad; Nvar, isperiodic=false)

# %%
Cϵ = LinearMap(ϵy.Σ, Ny)
ĈX = LocalizedEmpiricalCov(x0_ens, Loc; with_matrix=false)

##
CX_init = LinearMaps.FunctionMap{Float64,true}(ĈX, Nx, issymmetric=true)
sparse_pattern = Int.(filter(!iszero, H * (1:size(H, 2))))
sys_y = ObsSystem(H, Cϵ, CX_init; use_workspace=true, sparse_pattern);

# %%
filter_inflation = MultiAddInflation(Nx, β_infl, zeros(Nx), σx_filter)
locenkf = LocEnKF(identity, Ne, ϵy, sys_y, Loc, Δtdyn, Δtobs, isfiltered=true, isiterative=true);

# %%
store_state_path = joinpath(@__DIR__, "data")
X_locenkf = seqassim_trixi(data, 3, filter_inflation, locenkf, x0_ens, model.Ny, model.Nx, t0, sys_euler; ode_solver, cfl=0.2, store_state_path);

# %%
ex_sol = vec2sol(X_locenkf[1][:,2], equations, sys_euler.semi)
sol_true = vec2sol(data.xt[:,3], equations, sys_euler.semi)
pd_ex = Trixi.PlotData2D(ex_sol, sys_euler.semi)
pd_true = Trixi.PlotData2D(sol_true, sys_euler.semi)

##
plot(pd_true)
plot(pd_ex)

