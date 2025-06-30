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
Pkg.activate("..")

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

# %%
my_theme = Theme()

# %%
polydeg = 3
Ncells = 100

Nx = (polydeg+1)*Ncells
Δy = 20
Ny = ceil(Int64, Nx/Δy)

# Define Trixi system for inviscid Burgers equation
sys_burgers = setup_burgers(polydeg, Ncells);

xgrid = vec(sys_burgers.mesh.md.xq);

# %%
order_PA = 3

PA_offset = ceil(Int64, order_PA/2)
Ns = Nx - 2*PA_offset

PA = PolyAnnil(xgrid, order_PA; istruncated = true)
@assert size(PA.P) == (Ns, Nx)

S = LinearMaps.FunctionMap{Float64,true}((s,x)->mul!(s, PA.P, x), (x,s)->mul!(x, PA.P', s),
Ns, Nx; issymmetric=false, isposdef=false)

xs = xgrid[PA_offset+1:end-PA_offset];

# %%
Δtdyn = 0.01
Δtobs = 0.05

# %%
t0 = 0.0
Tf = 200
Tspin = 100
tf = t0 + Tf*Δtobs

# %%
π0 = MvNormal(zeros(Nx), Matrix(1.0*I, Nx, Nx))

# %%
σx_data = 0.0#Δtobs*1.0
σx_filter = 0.05#copy(σx_true)
@show σx_data σx_filter

σy = 0.15

ϵx_data = AdditiveInflation(Nx, zeros(Nx), σx_data)
ϵx_filter = AdditiveInflation(Nx, zeros(Nx), σx_filter)

ϵy = AdditiveInflation(Ny, zeros(Ny), σy);

# %%
h(x, t) = x[1:Δy:end]
H = LinearMap(sparse(Matrix(1.0*I, Nx, Nx)[1:Δy:end,:]))
F = StateSpace(x->x, h)

# %%
model = Model(Nx, Ny, Δtdyn, Δtobs, ϵx_data, ϵy, π0, 0, 0, 0, F);

# %%
x0 = vec(1/2 .+ 0.5*sin.(3*π*sys_burgers.mesh.md.xq));

# %%
lines(xgrid, x0)

# %%
@time data = generate_data_trixi(model, x0, Tf, sys_burgers)

# %%
heatmap( Δtobs*(1:Tf), xgrid, data.xt', axis=(;xlabel=L"t",ylabel=L"x",title=L"Solution of inviscid burgers, $u(x,t)$"))

# %%
fig = Figure()
ax = Axis(fig[1,1])

lines!(ax, xgrid, data.xt[:,1])
lines!(ax, xgrid, data.xt[:,100])
scatter!(ax, xgrid[1:Δy:end], data.yt[:,100])

fig

# %%
fig = Figure()
PA_t_idx = 100
ax = Axis(fig[1,1], title=L"Polynomial annhilator, $\mathbf{S}u(x,%$(Δtobs*PA_t_idx) )$")

# lines!(ax, xgrid, data.xt[:,1])
lines!(ax, xs, PA*data.xt[:,PA_t_idx])

fig

# %%
idx = 4

## Selecion of hyper-prior parameters
# power parameter
r_range = [ 1.0, .5, -.5, -1.0 ]; 
r = r_range[idx] # select parameter 
# shape parameter
β_range = [ 1.501, 3.0918, 2.0165, 1.0017 ]; 
β = β_range[idx] # shape parameter
# rate parameters 
ϑ_range = [ 5*10^(-2), 5.9323*10^(-3), 1.2583*10^(-3), 1.2308*10^(-4) ]; 
ϑ = ϑ_range[idx]

dist = GeneralizedGamma(r, β, ϑ);

# %%
lines(rand(dist, Ns))

# %%
# Define function class for the initial condition
αk = 0.7
f0 = SmoothPeriodic(xgrid, αk; L = 1.0);
Ne = 50
X = zeros(model.Ny + model.Nx, Ne)

for i=1:Ne
    regenerate!(f0)
    X[Ny+1:Ny+Nx,i] = f0.(xgrid)/4 .+ 0.5#initial_condition(αk, Δx, Nx)
end

# %%
fig = Figure()

ax = Axis(fig[1,1])

for i=1:10
    lines!(ax, xgrid, X[Ny+1:Ny+Nx,i])
end
lines!(xgrid, mean(X[Ny+1:Ny+Nx,:]; dims = 2)[:,1], linewidth = 5, linestyle = :dash)

lines!(ax, xgrid, x0, linewidth = 3)

fig

# %%
θinit = rand(dist, Ns);

# %%
Cθ = LinearMap(Diagonal(deepcopy(θinit)))
Cϵ = LinearMap(ϵy.Σ)
CX = LinearMap(Diagonal(1.0 .+ rand(Nx)))
sys_ys = ObsConstraintSystem(H, S, Cθ, Cϵ, CX);

# %%
sys_y = ObsSystem(H, Cϵ, CX);

# %%
yidx = 1:Δy:Nx
idx = vcat(collect(1:length(yidx))', collect(yidx)')

# Create Localization structure
Gxx(i,j) = periodicmetric!(i,j, Nx)
Gxy(i,j) = periodicmetric!(i,yidx[j], Nx)
Gyy(i,j) = periodicmetric!(yidx[i],yidx[j], Nx)

Lrad = 10
Loc = Localization(Lrad, Gxx, Gxy, Gxx)

# %%
β = 1.02
ϵxβ_filter = MultiAddInflation(Nx, β, zeros(Nx), σx_filter)

# %%
enkf = EnKF(Ne, ϵy, sys_y, Δtdyn, Δtobs)

# %%
locenkf = LocEnKF(Ne, ϵy, sys_y, Loc, Δtdyn, Δtobs)

# %%
henkf = HEnKF(Ne, ϵy, sys_ys, dist, deepcopy(θinit), Δtdyn, Δtobs)

# %%
hlocenkf = HLocEnKF(Ne, ϵy, sys_ys, Loc, dist, deepcopy(θinit), Δtdyn, Δtobs, Niter=5, θinit=1.)

# %%
X_locenkf = seqassim_trixi(data, Tf, ϵxβ_filter, locenkf, deepcopy(X), model.Ny, model.Nx, t0, sys_burgers);

# %%
X_hlocenkf, θ_hlocenkf = seqassim_trixi(data, Tf, ϵxβ_filter, hlocenkf, deepcopy(X), model.Ny, model.Nx, t0, sys_burgers);

# %%
with_theme(my_theme) do
	t_start = 1
	tsnap = Observable(t_start)
	x_tsnap = @lift(data.xt[:,$tsnap])
	y_tsnap = @lift(data.yt[:,$tsnap])
    ut = t->map(x->x[], vec(data.xt[:,round(Int,t/Δtobs)]))
	ys = @lift(ut(($tsnap)*Δtobs))
	X_hlocenkf_tsnap = @lift(vec(mean(X_hlocenkf[$tsnap+1]; dims = 2)))
	X_ens_tsnap = [@lift(X_hlocenkf[$tsnap+1][:,j]) for j in 1:Ne]
    theta_tsnap = @lift(θ_hlocenkf[$tsnap+1])
    cols = Makie.wong_colors()
	
	fig = Figure()
	
	ax1 = Axis(fig[1,1], title="Hierarchical Localized EnKF")
	
	# scatter!(ax1, xgrid, x_tsnap, label = "Truth")
	lines!(ax1, xgrid, X_hlocenkf_tsnap, linewidth = 3, label = "HLocEnKF")
	lines!(ax1, xgrid, ys, linewidth = 3, label = "State")
	lines!(ax1, xgrid[PA_offset+1:end-PA_offset], theta_tsnap, linewidth = 3, label = "θ")
	for j in 1:Ne
		lines!(ax1, xgrid, X_ens_tsnap[j], linewidth=0.9, color=( cols[1+(j % length(cols))], 0.2))
	end
	scatter!(ax1, xgrid[1:Δy:end], y_tsnap)
	
	axislegend(ax1)
	
	
	framerate = 10
	timestamps = range(t_start, Tf, step = 1)
	
	anim = Makie.Record(fig, timestamps; framerate = framerate) do t
	    tsnap[] = t
	end
	save("assim_hlenkf.gif", anim)
	anim
end

# %%
with_theme(my_theme) do
	t_start = 1
	tsnap = Observable(t_start)
	x_tsnap = @lift(data.xt[:,$tsnap])
	y_tsnap = @lift(data.yt[:,$tsnap])
    ut = t->map(x->x[], vec(data.xt[:,round(Int,t/Δtobs)]))
	ys = @lift(ut(($tsnap)*Δtobs))
	X_locenkf_tsnap = @lift(vec(mean(X_locenkf[$tsnap+1]; dims = 2)))
	X_ens_tsnap = [@lift(X_locenkf[$tsnap+1][:,j]) for j in 1:Ne]
    cols = Makie.wong_colors()
	
	fig = Figure()
	
	ax1 = Axis(fig[1,1], title="Localized EnKF")
	
	# scatter!(ax1, xgrid, x_tsnap, label = "Truth")
	lines!(ax1, xgrid, X_locenkf_tsnap, linewidth = 3, label = "LocEnKF")
	lines!(ax1, xgrid, ys, linewidth = 3, label = "State")
	for j in 1:Ne
		lines!(ax1, xgrid, X_ens_tsnap[j], linewidth=0.9, color=( cols[1+(j % length(cols))], 0.2))
	end
	scatter!(ax1, xgrid[1:Δy:end], y_tsnap)
	
	axislegend(ax1)
	
	
	framerate = 10
	timestamps = range(t_start, Tf, step = 1)
	
	anim = Makie.Record(fig, timestamps; framerate = framerate) do t
	    tsnap[] = t
	end
	save("assim_lenkf.gif", anim)
	anim
end

# %%
fig = Figure(fontsize = 20, size = (1200, 400))


ax1 = Axis(fig[1,1], 
           title = L"\text{Truth}",
           xlabel = L"t",
           ylabel = L"x",)

h1 = heatmap!(ax1, data.tt, xgrid, data.xt')

Colorbar(fig[1, 4], h1, label = L"u(x, t)")


ax2 = Axis(fig[1,2], 
           title = L"\text{EnKF}",
           xlabel = L"t",
           ylabel = L"x",)
h2 = heatmap!(ax2, data.tt, xgrid, mean_hist(X_locenkf)[:,2:end]')


ax3 = Axis(fig[1,3],
           title = L"\text{GSBL EnKF}",
           xlabel = L"t",
           ylabel = L"x",)
h3 = heatmap!(ax3, data.tt, xgrid, mean_hist(X_hlocenkf)[:,2:end]')

save("heatmap_inviscid_burgers.pdf", fig)

fig

# %%
