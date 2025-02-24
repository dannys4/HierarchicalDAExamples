# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Julia 1.11.3
#     language: julia
#     name: julia-1.11
# ---

using Pkg
Pkg.activate("..")

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

my_theme = theme_latexfonts()
update_theme!(my_theme, Axis=(;linewidth=3.))

equations = CompressibleEulerEquations1D(1.4)

with_theme(my_theme) do
    fig = Figure()
    ax = Axis(fig[1,1], title="Initial Condition", ylabel=L"u(0,x)", xlabel=L"x")
    xgrid = -5:0.01:5
    u0 = reduce(hcat, initial_condition_shu_osher.(xgrid, (0.,), equations))
    lines!(xgrid, u0[1,:], label=L"\rho")
    lines!(xgrid, u0[2,:], label=L"\rho v_1")
    lines!(xgrid, u0[3,:], label=L"\rho e")
    axislegend()
    fig
end

# +
polydeg = 3
Ncells = 100
Nvar = 3

Nxvar = (polydeg+1)*Ncells
Nx = Nvar*Nxvar
Δ = 20

Nyvar = ceil(Int64, Nxvar/Δ)
Ny = Nvar*Nyvar

# Define Trixi system for inviscid Burgers equation
sys_euler = setup_euler(polydeg, Ncells);

xgrid = vec(sys_euler.mesh.md.xq);

# +
order_PA = 3

Nsvar = Nxvar - 2*ceil(Int64, order_PA/2)
Ns = Nvar*Nsvar

PA = PolyAnnil(xgrid, order_PA; Nvar = Nvar, istruncated = true)

@assert size(PA.P) == (Ns, Nx)

S = LinearMaps.FunctionMap{Float64,true}((s,x)->mul!(s, PA.P, x), (x,s)->mul!(x, PA.P', s),
Ns, Nx; issymmetric=false, isposdef=false)

xs = xgrid[ceil(Int64, order_PA/2)+1:end-ceil(Int64, order_PA/2)];

# +
idxρ = 1:length(xgrid)
idxv = length(xgrid) .+ collect(1:length(xgrid))
idxp = 2*length(xgrid) .+ collect(1:length(xgrid));

idxρy = 1:ceil(Int64, Nx/(Δ*Nvar))
idxvy = ceil(Int64, Nx/(Δ*Nvar)) .+ collect(1:ceil(Int64, Nx/(Δ*Nvar)))
idxpy = 2*ceil(Int64, Nx/(Δ*Nvar)) .+ collect(1:ceil(Int64, Nx/(Δ*Nvar)))

idxρs = idxρ[ceil(Int64, order_PA/2)+1:end-ceil(Int64, order_PA/2)]
idxvs = idxv[ceil(Int64, order_PA/2)+1:end-ceil(Int64, order_PA/2)]
idxps = idxp[ceil(Int64, order_PA/2)+1:end-ceil(Int64, order_PA/2)];
# -

Δtdyn = 0.02
Δtobs = 0.02

t0 = 0.0
Tf = 200
Tspin = 1000
tf = t0 + Tf*Δtobs

π0 = MvNormal(zeros(Nx), Matrix(1.0*I, Nx, Nx))

# +
σx_true = 0.01#Δtobs*1.0
σx = 0.01#copy(σx_true)
@show σx


σy = 0.1

ϵx_true = AdditiveInflation(Nx, zeros(Nx), σx_true)
ϵx = AdditiveInflation(Nx, zeros(Nx), σx)

ϵy = AdditiveInflation(Ny, zeros(Ny), σy)
# -

h(x, t) = x[unroll(1:Δ:length(xgrid), length(xgrid), Nvar)]
H = LinearMap(sparse(Matrix(1.0*I, Nx, Nx)[unroll(1:Δ:length(xgrid), length(xgrid), Nvar),:]))
F = StateSpace(x->x, h)

model = Model(Nx, Ny, Δtdyn, Δtobs, ϵx_true, ϵy, π0, 0, 0, 0, F);

# Define function class for the initial condition
αk = 1.0
f0 = SmoothPeriodic(xgrid, αk; L = 10.0, Nvar = Nvar)

# +
xshuosher = zeros(Nx)
x0 = zeros(Nx)

for (i, xi) in enumerate(xgrid)
    x̃i = cons2prim(initial_condition_shu_osher(xi, 0.0, sys_euler.equations), sys_euler.equations)
    for k=1:Nvar
        xshuosher[((polydeg+1)*Ncells)*(k-1) + i] = x̃i[k]
    end
end

x0 = copy(xshuosher) + 0.01*f0(xgrid);
# x0 = f0(xgrid);
# -

@time data = generate_data_trixi(model, x0, Tf, sys_euler)

size(data.xt[idxρs,1]), size(xs)

with_theme(my_theme) do
	N_T = length(data.tt)
    p_t = t->data.xt[idxps,t]
    ρ_t = t->data.xt[idxρs,t]
    v_t = t->data.xt[idxvs,t]
    
	time = Observable(1)
    ps = @lift(p_t($time))
    ρs = @lift(ρ_t($time))
    vs = @lift(v_t($time))
    
    fig = Figure()
    ax = Axis(fig[1,1], xlabel=L"x", ylabel=L"u(t,x)", title=@lift("t = $($time/N_T)"))
	lines!(xs, ps, label=L"p")
    lines!(xs, ρs, label=L"\rho")
    lines!(xs, vs, label=L"v")
    axislegend()
	# hlines!(vec(mesh_x), fill(0.05, length(mesh_x)))
	timestamps = 1:N_T
	anim = CairoMakie.Makie.Record(fig, timestamps; framerate=N_T÷4) do t
		time[] = t
	end
	save("solution.gif", anim)
	anim
end


