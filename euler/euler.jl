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
Δ = 25

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
PA_offset = ceil(Int, order_PA/2)
xs = xgrid[PA_offset+1:end-PA_offset]

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
Tf = 100
Tspin = 1000
tf = t0 + Tf*Δtobs

π0 = MvNormal(zeros(Nx), Matrix(1.0*I, Nx, Nx))

# +
σx_true = 0.0#Δtobs*1.0
σx_filter = 0.1#copy(σx_true)
@show σx_true σx_filter


σy = 0.1

ϵx_true = AdditiveInflation(Nx, zeros(Nx), σx_true)
ϵx_filter = AdditiveInflation(Nx, zeros(Nx), σx_filter)

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
    ax = Axis(fig[1,1], xlabel=L"x", ylabel=L"u(t,x)", title=@lift("Shu-Osher, t = $($time/N_T)"))
	lines!(xs, ps, label=L"p", linewidth=3)
    lines!(xs, ρs, label=L"\rho", linewidth=3)
    lines!(xs, vs, label=L"v", linewidth=3)
    axislegend()
	# hlines!(vec(mesh_x), fill(0.05, length(mesh_x)))
	timestamps = 1:N_T
	anim = CairoMakie.Makie.Record(fig, timestamps; framerate=N_T÷4) do t
		time[] = t
	end
	save("solution.mp4", anim)
	anim
end

with_theme(my_theme) do 
    fig = Figure(size=(2100,700))
    for (i, idx_i) in enumerate([(idxρ, idxρy), (idxv, idxvy), (idxp, idxpy)])
        idx_x, idx_y = idx_i
        axi = Axis(fig[1,i])
        lines!(axi, xgrid, data.xt[idx_x,1])
        lines!(axi, xgrid, data.xt[idx_x,100])
        scatter!(axi, xgrid[1:Δ:end], data.yt[idx_y,100])
    end
    fig
end

# +
idx = 3

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

# r = -0.5
# β = 0.5
# ϑ = 0.01
dist = GeneralizedGamma(r, β, ϑ);

# +
Ne = 40
X = zeros(model.Ny + model.Nx, Ne)

for i=1:Ne
    regenerate!(f0)
    X[Ny+1:Ny+Nx,i] = xshuosher + 0.02*f0(xgrid)#initial_condition(αk, Δx, Nx)
end
# -

with_theme(my_theme) do
    fig = Figure()
    ax = Axis(fig[1,1])
    for i=1:10
        lines!(ax, xgrid, X[Ny .+ idxp,i])
    end
    
    lines!(ax, xgrid, x0[idxv])
    
    fig
end

θinit = rand(dist, Ns);

Cθ = LinearMap(Diagonal(deepcopy(θinit)))
Cϵ = LinearMap(ϵy.Σ)
CX = LinearMap(Diagonal(1.0 .+ rand(Nx)))
sys_ys = ObsConstraintSystem(H, S, Cθ, Cϵ, CX)

sys_y = ObsSystem(H, Cϵ, CX)

# +
yidx = 1:Δ:Nx
idx = vcat(collect(1:length(yidx))', collect(yidx)')

# @assert length(yidx) == Ny

# # Create Localization structure
Gxx(i,j) = periodicmetric!(mod(i,Nxvar)-1,mod(j, Nxvar)-1, Nxvar)
Gxy(i,j) = periodicmetric!(i,yidx[j], Nxvar)
Gyy(i,j) = periodicmetric!(yidx[i],yidx[j], Nxvar)

Lrad = 10
Loc = Localization(Lrad, Gxx, Gxy, Gxx)
# -

β = 1.01
filter_inflation = MultiAddInflation(Nx, β, zeros(Nx), σx_filter)

locenkf = LocEnKF(x-> abs.(x), Ne, ϵy, sys_y, Loc, Δtdyn, Δtobs)

# +
# X_locenkf = seqassim_trixi(data, Tf, filter_inflation, locenkf, deepcopy(X), model.Ny, model.Nx, t0, sys_euler);
# -

hlocenkf = HLocEnKF(x-> abs.(x), Ne, ϵy, sys_ys, Loc, dist, deepcopy(θinit), Δtdyn, Δtobs, Niter=5, θinit=1.)

X_hlocenkf, θ_hlocenkf = seqassim_trixi(data, Tf, filter_inflation, hlocenkf, deepcopy(X), model.Ny, model.Nx, t0, sys_euler);

rmse_hlocenkf = mean(map(i->norm(data.xt[:,i]-mean(X_hlocenkf[i+1]; dims = 2))/sqrt(Nx), 1:Tf))
rmse_locenkf  = mean(map(i->norm(data.xt[:,i]-mean(X_locenkf[ i+1]; dims = 2))/sqrt(Nx), 1:Tf))
@info "" rmse_hlocenkf rmse_locenkf

# +
fig = Figure()
tsnap = 80
idx = 10
ax = Axis(fig[1,1])
lines!(ax, xgrid, data.xt[idxρ,tsnap], linewidth = 3, label = "Truth")

# lines!(ax, xgrid, X[Ny+1:Ny+Nx,2])
# lines!(ax, xgrid, X_enkf[tsnap+1][:,idx], linewidth = 3, label = "EnKF")
# lines!(ax, xgrid, mean(X_enkf[tsnap+1]; dims = 2)[:,1], linewidth = 3, label = "EnKF")

# lines!(ax, xgrid, X_locenkf[tsnap+1][:,1][idxρ,1], linewidth = 3, label = "LocEnKF")
# lines!(ax, xgrid, mean(X_locenkf[tsnap+1]; dims = 2)[idxρ,1], linewidth = 3, label = "LocEnKF")


# lines!(ax, xgrid, X_henkf[tsnap+1][:,2], linewidth = 3, label = "HEnKF")
# lines!(ax, xgrid, X_henkf[tsnap+1][:,2])
# lines!(ax, xgrid, mean(X_henkf[tsnap+1]; dims = 2)[:,1], linewidth = 3, label = "HEnKF")

for j in 1:Ne
    lines!(ax, xgrid, X_hlocenkf[tsnap+1][:,j][idxρ,1], linewidth = 0.8)
end
# lines!(ax, xgrid, mean(X_hlocenkf[tsnap+1]; dims = 2)[idxρ,1], linewidth = 3, label = "HLocEnKF")

# ax2 = Axis(fig[1,2])

# fig[1, 2] = Legend(fig, ax, "Filters", framevisible = false)

# lines!(ax, xgrid[1:2:end], data.yt[idxpy,tsnap], linewidth = 3)


# scatter!(ax, xgrid[1:Δ:end], data.yt[:,tsnap])
# lines!(ax, xs, PA.P*X_enkf[tsnap+1][:,2])

axislegend()
fig
# -
false && with_theme(my_theme) do
    cols = Makie.wong_colors()
	N_T = length(data.tt)
    p_t = t->data.xt[idxps,t]
    ρ_t = t->data.xt[idxρs,t]
    v_t = t->data.xt[idxvs,t]
    J_Ens = 5
    locenkf_p_t = t->X_locenkf[t][:,J_Ens][idxps]
    locenkf_ρ_t = t->X_locenkf[t][:,J_Ens][idxρs]
    locenkf_v_t = t->X_locenkf[t][:,J_Ens][idxvs]
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
    ax_p = Axis(fig[1,1], xlabel=L"x", title=L"p", aspect=1.)
    ax_ρ = Axis(fig[1,2], xlabel=L"x", title=L"\rho", aspect=1.)
    ax_v = Axis(fig[1,3], xlabel=L"x", title=L"v", aspect=1.)
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
	anim = CairoMakie.Makie.Record(fig, timestamps; framerate=N_T÷4) do t
		time[] = t
	end
	save("loc_enkf_result.mp4", anim)
	anim
end

with_theme(my_theme) do
    cols = Makie.wong_colors()
	N_T = length(data.tt)
    p_t = t->data.xt[idxps,t]
    ρ_t = t->data.xt[idxρs,t]
    v_t = t->data.xt[idxvs,t]
    J_Ens = 5
    hlocenkf_p_t = t->X_hlocenkf[t][:,J_Ens][idxps]
    hlocenkf_ρ_t = t->X_hlocenkf[t][:,J_Ens][idxρs]
    hlocenkf_v_t = t->X_hlocenkf[t][:,J_Ens][idxvs]
    # Offset each to account for previous offsets in data
    θ_p_t = t->(θ_hlocenkf[t][idxps[PA_offset+1:end-PA_offset] .- 4PA_offset])
    θ_ρ_t = t->(θ_hlocenkf[t][idxρs[PA_offset+1:end-PA_offset] .- 0PA_offset])
    θ_v_t = t->(θ_hlocenkf[t][idxvs[PA_offset+1:end-PA_offset] .- 2PA_offset])
    
	time = Observable(1)
    ps = @lift(p_t($time))
    ρs = @lift(ρ_t($time))
    vs = @lift(v_t($time))
    θ_ps = @lift(θ_p_t($time)/norm(θ_p_t($time)))
    θ_ρs = @lift(θ_ρ_t($time)/norm(θ_ρ_t($time)))
    θ_vs = @lift(θ_v_t($time)/norm(θ_v_t($time)))
    hlocenkf_ps = @lift(hlocenkf_p_t($time))
    hlocenkf_ρs = @lift(hlocenkf_ρ_t($time))
    hlocenkf_vs = @lift(hlocenkf_v_t($time))
    xs_theta = xs[1+PA_offset:end-PA_offset]

    px_size = 600
    fig = Figure(size=(3px_size, px_size))
    ax_p = Axis(fig[1,1], xlabel=L"x", title=L"p", aspect=1.)
    ax_ρ = Axis(fig[1,2], xlabel=L"x", title=L"\rho", aspect=1.)
    ax_v = Axis(fig[1,3], xlabel=L"x", title=L"v", aspect=1.)
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
    
	timestamps = 1:N_T
	anim = CairoMakie.Makie.Record(fig, timestamps; framerate=N_T÷4) do t
		time[] = t
	end
	save("hloc_enkf_result.mp4", anim)
	anim
end


