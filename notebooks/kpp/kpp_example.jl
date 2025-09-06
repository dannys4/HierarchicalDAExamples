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
polydeg = 4
Ncells_dim = 32
sys_kpp = setup_kpp(polydeg, Ncells_dim)

# %%
t0, tf = 0., 0.1
order_PA = 3

Δtobs = 0.05
Δtdyn = 0.05
Δy = 20
Tf = ceil(Int, (tf - t0) / Δtobs)

# %%
Lrad = 5
Ne = 50
σx_filter = 0.02
β_infl = 1.02

# %%
Nvar = nvariables(sys_kpp.equations)
# PA_offset = ceil(Int64, order_PA / 2)
PA = VerticalPolyAnnil2D(sys_kpp, order_PA, Nvar)
S = LinearMap(PA.P)
H = create_observation_operator2d(
    sys_kpp, polydeg + 1; offset=(polydeg + 1) ÷ 2, Nvar
)
Nx, Ny = size(PA.P, 1), size(H, 1)
Nx_var = Nx ÷ Nvar

# %%
π0 = MvNormal(I(Nx))

# %%
π0 = MvNormal(I(Nx))
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
ode_solver = SSPRK43()
x0_sol = map(xy -> initial_condition_kpp(xy, 0., sys_kpp.equations), zip(sys_kpp.mesh.md.xyzq...))
x0 = sol2vec(x0_sol, sys_kpp.equations)


# %%
data = generate_data_trixi(model, x0, Tf, sys_kpp; ode_solver, cfl=0.9)

# %%
L_f0 = 2.0
alpha_f0 = 0.75
grid1d = unique(x -> round(x, digits=3), sys_kpp.mesh.md.xq)
f0_row = SmoothPeriodic(grid1d, alpha_f0; L=L_f0, Nvar)
f0_col = SmoothPeriodic(grid1d, alpha_f0; L=L_f0, Nvar)
# Need prim vars ρ and p to be positive
x0_ens = reduce(hcat, sample_initial_state2d(sys_kpp, f0_row, f0_col; Nvar) for _ in 1:Ne)
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
CX_init = LinearMaps.FunctionMap{Float64,true}(ĈX, Nx, issymmetric=true)
sparse_pattern = Int.(filter(!iszero, H * (1:size(H, 2))))
sys_y = ObsSystem(H, Cϵ, CX_init; use_workspace=true, sparse_pattern);

# %%
filter_inflation = MultiAddInflation(Nx, β_infl, zeros(Nx), σx_filter)
locenkf = LocEnKF(identity, Ne, ϵy, sys_y, Loc, Δtdyn, Δtobs, isfiltered=true, isiterative=true);

# %%
store_state_path = joinpath(@__DIR__, "data")
X_locenkf = seqassim_trixi(data, Tf, filter_inflation, locenkf, x0_ens, model.Ny, model.Nx, t0, sys_kpp; ode_solver, cfl=0.8, store_state_path);

# %%
tspan = (0.0, 1.0)
ode = semidiscretize(sys_kpp.semi, tspan)

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(sys_kpp.semi, interval=100, uEltype=Float64)
stepsize_callback = StepsizeCallback(; cfl=0.8)
alive_callback = AliveCallback(analysis_interval=200)
callbacks = CallbackSet(summary_callback,
    analysis_callback, alive_callback,
    stepsize_callback
)

# %%
sol_dgmulti = solve(ode, SSPRK43(); ode_default_options()..., callback=callbacks)

##
pd_sol = PlotData2D(sol_dgmulti[end], sys_kpp.semi)
plot(pd_sol)