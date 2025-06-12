# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Julia 1.11.5
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
update_theme!(my_theme, linewidth=3.)

equations = CompressibleEulerEquations1D(1.4)

# +
polydeg = 2
Ncells = 200
Nvar = 3

Nxvar = (polydeg+1)*Ncells
Nx = Nvar*Nxvar
Δy = 100

Nyvar = ceil(Int64, Nxvar/Δy)
Ny = Nvar*Nyvar

# Define Trixi system for inviscid Burgers equation
sys_euler = setup_euler(polydeg, Ncells);

xgrid = GridFromMesh(sys_euler);

# +
order_PA = 3

Nsvar = Nxvar - 2*ceil(Int64, order_PA/2)
Ns = Nvar*Nsvar

PA = PolyAnnil(xgrid, order_PA; Nvar = Nvar, istruncated = true)

@assert size(PA.P) == (Ns, Nx)

S = LinearMaps.FunctionMap{Float64,true}((s,x)->mul!(s, PA.P, x), (x,s)->mul!(x, PA.P', s),
Ns, Nx; issymmetric=false, isposdef=false)
PA_offset = ceil(Int, order_PA/2)
xs = xgrid[PA_offset+1:end-PA_offset];

# +
idxρ = 1:length(xgrid)
idxv = length(xgrid) .+ collect(1:length(xgrid))
idxp = 2*length(xgrid) .+ collect(1:length(xgrid));

idxρy = 1:ceil(Int64, Nx/(Δy*Nvar))
idxvy = ceil(Int64, Nx/(Δy*Nvar)) .+ collect(1:ceil(Int64, Nx/(Δy*Nvar)))
idxpy = 2*ceil(Int64, Nx/(Δy*Nvar)) .+ collect(1:ceil(Int64, Nx/(Δy*Nvar)))

idxρs = idxρ[ceil(Int64, order_PA/2)+1:end-ceil(Int64, order_PA/2)]
idxvs = idxv[ceil(Int64, order_PA/2)+1:end-ceil(Int64, order_PA/2)]
idxps = idxp[ceil(Int64, order_PA/2)+1:end-ceil(Int64, order_PA/2)];

# +
Δtdyn = 0.02
Δtobs = 0.04

t0 = 0.0
Tf = 50
Tspin = 1000
tf = t0 + Tf*Δtobs

π0 = MvNormal(zeros(Nx), Matrix(1.0*I, Nx, Nx))

# +
σx_true = 0.0#Δtobs*1.0
@show σx_true

σy = 0.1

ϵx_true = AdditiveInflation(Nx, zeros(Nx), σx_true)
ϵx_filter = AdditiveInflation(Nx, zeros(Nx), σx_filter)

ϵy = AdditiveInflation(Ny, zeros(Ny), σy);

# +
h(x, t) = x[unroll(1:Δy:length(xgrid), length(xgrid), Nvar)]
H = LinearMap(sparse(Matrix(1.0*I, Nx, Nx)[unroll(1:Δy:length(xgrid), length(xgrid), Nvar),:]))
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
        xshuosher[Nxvar*(k-1) + i] = x̃i[k]
    end
end

x0 = xshuosher;# + 0.01*f0(xgrid);
# -

thresholds = (5e-6, 5e-6)
variables = (Trixi.density, Trixi.pressure)
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=thresholds,
                                                     variables=variables)
ode_solver = SSPRK43(stage_limiter!)
# ode_solver = SSPRK43()
# ode_solver = RDPK3SpFSAL49(stage_limiter!)
# ode_solver = CarpenterKennedy2N54(stage_limiter!, williamson_condition=false)

data = generate_data_trixi(deepcopy(model), deepcopy(x0), Tf, deepcopy(sys_euler); ode_solver, cfl=0.9)

false && with_theme(my_theme) do
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
    title_times = round.(data.tt, digits=2)
    ax = Axis(fig[1,1], xlabel=L"x", ylabel=L"u(t,x)", title=@lift("Shu-Osher, t = $(title_times[$time])"))
    lines!(xs, ρs, label=L"\rho", linewidth=3)
	lines!(xs, ps, label=L"p", linewidth=3)
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
        lines!(axi, xgrid, data.xt[idx_x,1], linewidth=3)
        scatter!(axi, xgrid[1:Δy:end], data.yt[idx_y,1], markersize=18)
        errorbars!(axi, xgrid[1:Δy:end], data.yt[idx_y,1], fill(2σy,length(idx_y)))
        lines!(axi, xgrid, data.xt[idx_x,div(end,3)], linewidth=3)
        scatter!(axi, xgrid[1:Δy:end], data.yt[idx_y,div(end,3)], markersize=18)
        errorbars!(axi, xgrid[1:Δy:end], data.yt[idx_y,div(end,3)], fill(2σy,length(idx_y)))
    end
    fig
end

# +
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

# r = -0.5
# β = 0.5
# ϑ = 0.01
dist = GeneralizedGamma(r, β, ϑ);

# +
Ne = 40
X = zeros(model.Ny + model.Nx, Ne)

for i=1:Ne
    regenerate!(f0)
    X[Ny+1:Ny+Nx,i] = max.(1e-5,xshuosher + 0.1*f0(xgrid))#initial_condition(αk, Δx, Nx)
end
# -

θinit = rand(dist, Ns);

Cθ = LinearMap(Diagonal(deepcopy(θinit)))
Cϵ = LinearMap(ϵy.Σ)
CX = LinearMap(I(Nx))
sys_ys = ObsConstraintSystem(H, S, Cθ, Cϵ, CX)

sys_y = ObsSystem(H, Cϵ, CX)

# +
yidx = 1:Δy:Nx
idx = vcat(collect(1:length(yidx))', collect(yidx)')

# @assert length(yidx) == Ny

# # Create Localization structure
Gxx(i,j) = periodicmetric!(mod(i,Nxvar)-1,mod(j, Nxvar)-1, Nxvar)
Gxy(i,j) = periodicmetric!(i,yidx[j], Nxvar)
Gyy(i,j) = periodicmetric!(yidx[i],yidx[j], Nxvar)

Lrad = 7
Loc = Localization(Lrad, Gxx, Gxy, Gxx)
# -

σx_filter = 0.02
β = 1.00
filter_inflation = MultiAddInflation(Nx, β, zeros(Nx), σx_filter)

locenkf = LocEnKF(x-> max.((1e-4,), x), Ne, ϵy, sys_y, Loc, Δtdyn, Δtobs, isfiltered = true)

X_locenkf = seqassim_trixi(data, Tf, filter_inflation, locenkf, deepcopy(X), model.Ny, model.Nx, t0, sys_euler; ode_solver, cfl = 0.9);

hlocenkf = HLocEnKF(x->max.((1e-4,), x), Ne, ϵy, sys_ys, Loc, dist, deepcopy(θinit), Δtdyn, Δtobs, Niter=5, θinit=1., isfiltered=true)

X_hlocenkf, θ_hlocenkf = seqassim_trixi(data, Tf, filter_inflation, hlocenkf, deepcopy(X), model.Ny, model.Nx, t0, sys_euler; ode_solver, cfl=0.9);

function __RMSE(X_ens::AbstractMatrix, x::AbstractVector)
    d,N = size(X_ens)
    @assert d == length(x)
    mean(axes(X_ens,2)) do ens_idx
        norm(@view(X_ens[:,ens_idx]) - x)
    end
end
rel_norm = mean(i->norm(data.xt[:,i]), axes(data.xt,2))
# rmse_hlocenkf = mean(i->mean(j->norm(data.xt[:,i] - mean(X_hlocenkf[i+1]; dims = 2)), axes(data.xt,2))
# rmse_locenkf  = mean(i->norm(data.xt[:,i] - mean(X_locenkf[ i+1]; dims = 2)), axes(data.xt,2))
rmse_hlocenkf = mean(__t->__RMSE(X_hlocenkf[__t+1], @view(data.xt[:,__t])), axes(data.xt,2))
rmse_locenkf  = mean(__t->__RMSE(X_locenkf[ __t+1], @view(data.xt[:,__t])), axes(data.xt,2))
rel_rmse_hlocenkf = rmse_hlocenkf/rel_norm
rel_rmse_locenkf  = rmse_locenkf/rel_norm
@info "" rel_rmse_hlocenkf rel_rmse_locenkf

errs_locenkf2  = map(j->CRPS(X_locenkf[j+1] , @view(data.xt[:,j]), :norm2), axes(data.xt,2))
errs_hlocenkf2 = map(j->CRPS(X_hlocenkf[j+1], @view(data.xt[:,j]), :norm2), axes(data.xt,2))
rel_norms = map(norm, eachcol(data.xt))
rmse_locenkf, rmse_hlocenkf = [mean(i->err[i].rmse/rel_norms[i], eachindex(err)) for err in [errs_locenkf2, errs_hlocenkf2]]
crps_locenkf, crps_hlocenkf = [mean(i->err[i].crps/rel_norms[i], eachindex(err)) for err in [errs_locenkf2, errs_hlocenkf2]]
@info "2-Norm results" rmse_locenkf  crps_locenkf
@info "" rmse_hlocenkf crps_hlocenkf

errs_locenkf1  = map(j->CRPS(X_locenkf[j+1] , @view(data.xt[:,j]), :norm1), axes(data.xt,2))
errs_hlocenkf1 = map(j->CRPS(X_hlocenkf[j+1], @view(data.xt[:,j]), :norm1), axes(data.xt,2))
rel_norms = map(norm, eachcol(data.xt))
rmse_locenkf, rmse_hlocenkf = [mean(i->err[i].rmse/rel_norms[i], eachindex(err)) for err in [errs_locenkf1, errs_hlocenkf1]]
crps_locenkf, crps_hlocenkf = [mean(i->err[i].crps/rel_norms[i], eachindex(err)) for err in [errs_locenkf1, errs_hlocenkf1]]
@info "1-Norm results" rmse_locenkf  crps_locenkf
@info "" rmse_hlocenkf crps_hlocenkf

# +
fig = Figure(size=(1010,500))
tsnap = 50
idx = 10
ax1 = Axis(fig[1,1])
ax2 = Axis(fig[1,2])
lines!(ax1, xgrid, data.xt[idxρ,tsnap], linewidth = 3, label = "Truth")
lines!(ax2, xgrid, data.xt[idxρ,tsnap], linewidth = 3, label = "Truth")

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
    col = cols[mod1(j,length(cols))]
    lines!(ax1, xgrid, X_locenkf[tsnap+1][:,j][idxρ,1], linewidth = 0.8, label=ifelse(j==1,"Loc-EnKF",nothing), color=(col,0.2))
    lines!(ax2, xgrid, X_hlocenkf[tsnap+1][:,j][idxρ,1], linewidth = 0.8, label=ifelse(j==1,"GSBL-EnKF",nothing), color=(col,0.2))
end
# lines!(ax, xgrid, mean(X_hlocenkf[tsnap+1]; dims = 2)[idxρ,1], linewidth = 3, label = "HLocEnKF")

# ax2 = Axis(fig[1,2])

# fig[1, 2] = Legend(fig, ax, "Filters", framevisible = false)

# lines!(ax, xgrid[1:2:end], data.yt[idxpy,tsnap], linewidth = 3)


# scatter!(ax, xgrid[1:Δy:end], data.yt[:,tsnap])
# lines!(ax, xs, PA.P*X_enkf[tsnap+1][:,2])

axislegend(ax1)
axislegend(ax2)
fig
# -
with_theme(my_theme) do
    cols = Makie.wong_colors()
	N_T = length(data.tt)
    p_t = t->data.xt[idxps,t]
    ρ_t = t->data.xt[idxρs,t]
    v_t = t->data.xt[idxvs,t]
    J_Ens = 5
    locenkf_p_t = t->X_locenkf[t+1][:,J_Ens][idxps]
    locenkf_ρ_t = t->X_locenkf[t+1][:,J_Ens][idxρs]
    locenkf_v_t = t->X_locenkf[t+1][:,J_Ens][idxvs]
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
    θ_ρ_t = t->(θ_hlocenkf[t][idxρs[PA_offset+1:end-PA_offset] .- 0PA_offset])
    θ_v_t = t->(θ_hlocenkf[t][idxvs[PA_offset+1:end-PA_offset] .- 2PA_offset])
    θ_p_t = t->(θ_hlocenkf[t][idxps[PA_offset+1:end-PA_offset] .- 4PA_offset])
    
	time = Observable(2)
    ps = @lift(p_t($time-1))
    ρs = @lift(ρ_t($time-1))
    vs = @lift(v_t($time-1))
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
    
	timestamps = 2:N_T
	anim = CairoMakie.Makie.Record(fig, timestamps; framerate=N_T÷4) do t
		time[] = t
	end
	save("hloc_enkf_result.mp4", anim)
	anim
end


