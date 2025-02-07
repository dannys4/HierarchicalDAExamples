### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 633dd2fa-2388-4b10-8aea-83754fba1e99
begin
	using Pkg
	Pkg.activate("..")
end

# ╔═╡ beac8a48-22f0-4bca-b60f-d956c92b7706
using Revise

# ╔═╡ f61f50da-e344-11ef-2c68-b70ecb2e5ab2
begin
	using Trixi
	using LinearAlgebra
	using OrdinaryDiffEq
	using CairoMakie
	using ProgressLogging
	using HierarchicalDA
	using StaticArrays
	using LinearMaps
	using TransportBasedInference2
	using Distributions
	using SparseArrays
end

# ╔═╡ f4eac16c-f5bc-44ff-88d3-81c533b16075
begin
	using Trixi: entropy2cons
	function Trixi.entropy2cons(t, ::LinearScalarAdvectionEquation1D)
		t
	end
end

# ╔═╡ 243f2bf0-cd29-4dcd-a8e6-3bead9a48b74
my_theme = Theme()

# ╔═╡ 34c27970-71f0-4fe3-9ff3-7c3cdff7b434
begin
	function initial_condition_sawtooth_fcn(x, t;
		a=-1, b=1, N_saw=4, u_a = 0.0, u_b = 0.95)
		normalized_x = (x[] - a)/(b - a)
		which_seg = normalized_x * N_saw
		per_x = (which_seg - floor(which_seg))
	    u_a + per_x*(u_b - u_a)
	end
	function initial_condition_sawtooth(x, t, _::LinearScalarAdvectionEquation1D)
		SVector(initial_condition_sawtooth_fcn(x, t))
	end
end

# ╔═╡ 9c9595eb-a46b-473b-9a03-5feae5de8873
with_theme(my_theme) do
	fig = Figure()
	ax = Axis(fig[1,1], title="Initial condition", ylabel=L"u(0,x)", xlabel=L"x")
	lines!(-1:0.01:1,initial_condition_sawtooth_fcn.(-1:0.01:1.,(nothing,)))
	fig
end

# ╔═╡ 7d827a01-37fb-4345-86f7-fd93139db8d0
begin
	polydeg = 8
	advection_velocity = 0.1
	equations = LinearScalarAdvectionEquation1D(advection_velocity)
	volume_flux = flux_central
	surface_flux = flux_lax_friedrichs
	# basis = LobattoLegendreBasis(polydeg)
	basis = DGMultiBasis(Trixi.Line(), polydeg, approximation_type=GaussSBP())
	
	indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = first)

	surface_integral = SurfaceIntegralWeakForm(surface_flux)
	# volume_integral = VolumeIntegralWeakForm()
	# volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
	volume_integral = VolumeIntegralShockCapturingHG(indicator_sc)
	# solver = DGSEM(basis, surface_flux, volume_integral)
	solver = DGMulti(basis; surface_integral, volume_integral)
	
	N_elements = 100
	coordinates_min, coordinates_max = -1., 1.
	# mesh = StructuredMesh((N_elements,), coordinates_min, coordinates_max)
	mesh = DGMultiMesh(solver, (N_elements,); periodicity=true, coordinates_min, coordinates_max)
	initial_condition = initial_condition_sawtooth
	
	semi = SemidiscretizationHyperbolic(mesh,
	                                    equations,
	                                    initial_condition,
	                                    solver, boundary_conditions=boundary_condition_periodic)
	tspan = (0.0, 10.0-eps())
	ode = semidiscretize(semi, tspan)
end

# ╔═╡ 9fd61ec9-c7a6-46b7-bdfe-15f8cf58aee2
begin
	stepsize_callback = StepsizeCallback(cfl = 0.2)
	# stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (0.0,), variables=(first,))
	sol = solve(ode, SSPRK43(), adaptive=true, callback = stepsize_callback)
end

# ╔═╡ 2b458b0f-12ce-4092-a62a-199d62bad3d9
begin
	if mesh isa StructuredMesh
		dx = (coordinates_max - coordinates_min) / N_elements
		mesh_x = Matrix{Float64}(undef, length(basis.nodes), N_elements)
		for element in 1:N_elements
		    x_l = -1 + (element - 1) * dx + dx/2
		    for i in eachindex(basis.nodes) # basis points in [-1, 1]
		        ξ = basis.nodes[i]
		        mesh_x[i, element] = x_l + dx/2 * ξ
		    end
		end
	elseif mesh isa DGMultiMesh
		# Get quadrature points
		mesh_x = mesh.md.xq
	else
		throw(ArgumentError("Unknown mesh type $(typeof(mesh))"))
	end
	xgrid = vec(mesh_x)
end

# ╔═╡ 4ddffb96-f4b7-4000-97a5-93b9ef4f7952
# with_theme(my_theme) do
# 	ut = t->map(x->x[], vec(sol(t)))
# 	time = Observable(0.0)
# 	ys = @lift(ut($time))
# 	fig, ax, pl = lines(xgrid, ys, axis=(;xlabel=L"x", ylabel=L"u(t,x)", title=@lift("t = $($time)")))
# 	# hlines!(vec(mesh_x), fill(0.05, length(mesh_x)))
# 	N_T = 101
# 	timestamps = range(0,tspan[end], N_T)
# 	anim = CairoMakie.Makie.Record(fig, timestamps; framerate=N_T÷4) do t
# 		time[] = t
# 	end
# 	save("solution.gif", anim)
# 	anim
# end

# ╔═╡ 8e466589-9814-4ed6-906e-500197474a92
begin
	order_PA = 3
	Nx = length(xgrid)
	Ns = Nx - 2ceil(Int, order_PA/2)
	PA = PolyAnnil(xgrid, order_PA; istruncated = true)
	sys_advection = TrixiSystem(equations, solver, mesh, semi)
	S = LinearMaps.FunctionMap{Float64,true}((s,x)->mul!(s, PA.P, x), (x,s)->mul!(x, PA.P', s), Ns, Nx; issymmetric=false, isposdef=false)
	xgrid_S = xgrid[ceil(Int, order_PA/2)+1:end-ceil(Int, order_PA/2)]
end

# ╔═╡ 14885cc2-2a74-4b95-a112-fddef6ac240a
size(PA.P), size(xgrid), (Ns, Nx)

# ╔═╡ b4d79689-309b-44ca-b783-6070e8b364ea
begin
	Δ = 40
	Ny = ceil(Int64, Nx/Δ)
	Δtdyn = 0.05
	Δtobs = 0.25
	t0 = 0.0
	Tf = 20
	Tspin = 1000
	tf = t0 + Tf*Δtobs
	π0 = MvNormal(zeros(Nx), Matrix(1.0*I, Nx, Nx))
	σx_true = 0.005
	σx = 0.01
	σy = 0.05
	
	ϵx_true = AdditiveInflation(Nx, zeros(Nx), σx_true)
	ϵx = AdditiveInflation(Nx, zeros(Nx), σx)
	ϵy = AdditiveInflation(Ny, zeros(Ny), σy)
end

# ╔═╡ fa3b9910-b166-4651-8103-e27d22b4aab9
begin
	h(x, t) = x[1:Δ:end]
	H = LinearMap(sparse(Matrix(1.0*I, Nx, Nx)[1:Δ:end,:]))
	F = StateSpace(x->x, h)
	model = Model(Nx, Ny, Δtdyn, Δtobs, ϵx_true, ϵy, π0, 0, 0, 0, F)
end

# ╔═╡ 2bf04647-61ce-4e04-aa5b-6d51128847a6
begin
	u0 = initial_condition_sawtooth_fcn.(xgrid, (0,))
	data = generate_data_trixi(model, u0, Tf, sys_advection)
end

# ╔═╡ 9d041faa-a058-4552-842d-7305e114e343
with_theme(my_theme) do
	fig = Figure()
	ax = Axis(fig[1,1], xlabel=L"x", title="Generated data")
	lines!(xgrid, reduce(vcat, vec(ode.u0)), linewidth=3, label="Initial condition", linestyle=:dash)
	scatter!(xgrid, data.xt[:,1], label="State 1")
	scatter!(h(xgrid,nothing), data.yt[:,1], label="Obs 1")
	mid_T = Tf÷2
	scatter!(xgrid, data.xt[:,mid_T], label="State $mid_T", marker=:rect, markersize=14)
	scatter!(h(xgrid,nothing), data.yt[:,mid_T], label="Obs $mid_T", marker=:rect, markersize=14)
	axislegend()
	fig
end

# ╔═╡ 1e253c95-fad3-4510-b2fc-96f63612208b
f0 = SmoothPeriodic(xgrid, 0.8)

# ╔═╡ 5aa16815-27e1-415c-a8e7-73243faa748f
lines(xgrid, f0.(xgrid))

# ╔═╡ c3fd8f87-1b79-455f-bc1d-bcae48cb5a68
begin
	Ne = 100
	X0 = zeros(model.Ny + model.Nx, Ne)
	
	for i=1:Ne
	    regenerate!(f0)
	    X0[Ny+1:Ny+Nx,i] = exp.(f0.(xgrid)/2) .- 0.5#initial_condition(αk, Δx, Nx)
	end
end

# ╔═╡ ee5de05d-b9ef-4224-83b7-9ac2deec1adf
with_theme(my_theme) do
	fig = Figure()
	
	ax = Axis(fig[1,1], title="Initial ensemble", xlabel=L"x")
	
	for i=1:10
	    lines!(ax, xgrid, X0[Ny+1:Ny+Nx,i], linewidth=1.)
	end
	lines!(xgrid, mean(X0[Ny+1:Ny+Nx,:]; dims = 2)[:,1], linewidth = 5, linestyle = :dash, label="Ensemble mean")
	
	lines!(ax, xgrid, u0, linewidth = 5, label="Truth")
	axislegend()
	fig
end

# ╔═╡ 1f4bd9d4-57a3-42aa-ad14-2f8bb3cac5fa
begin
	idx = 4
	
	## Selecion of hyper-prior parameters
	# power parameter
	r_range = [ 1.0, .5, -.5, -1.0 ]; 
	r = r_range[idx] # select parameter 
	# shape parameter
	β_range = [ 1.501, 3.0918, 2.0165, 1.0017 ]; 
	β_dist = β_range[idx] # shape parameter
	# rate parameters 
	ϑ_range = [ 5*10^(-2), 5.9323*10^(-3), 1.2583*10^(-3), 1.2308*10^(-4) ]; 
	ϑ = ϑ_range[idx]
	# ϑ = 1e-5
	
	# r = -0.5
	# β = 0.5
	# ϑ = 0.01
	dist = GeneralizedGamma(r, β_dist, ϑ);
end

# ╔═╡ ccdf262b-e080-4571-b4eb-3d236adfb4c3
begin
	yidx = 1:Δ:Nx
	# idx = vcat(collect(1:length(yidx))', collect(yidx)')
	
	# @assert length(yidx) == Ny
	
	# # Create Localization structure
	Gxx(i,j) = periodicmetric!(i,j, Nx)
	Gxy(i,j) = periodicmetric!(i,yidx[j], Nx)
	Gyy(i,j) = periodicmetric!(yidx[i],yidx[j], Nx)
	
	Lrad = 7
	Loc = Localization(Lrad, Gxx, Gxy, Gxx)
	β_infl = 1.01
	ϵxβ = MultiAddInflation(Nx, β_infl, zeros(Nx), σx)
end

# ╔═╡ 30ee368d-d1c6-428e-b24b-193e4e0a84f0
begin
	Cθ = LinearMap(Diagonal(rand(dist, Ns)))
	Cϵ = LinearMap(ϵy.Σ)
	CX = LinearMap(Diagonal(1.0 .+ rand(Nx)))
	sys_ys = ObsConstraintSystem(H, S, Cθ, Cϵ, CX)
	θinit = rand(dist, Ns)
end

# ╔═╡ 0fd77f8a-5b6a-414d-a340-3d7011dccf0b
hlocenkf = HLocEnKF(Ne, ϵy, sys_ys, Loc, dist, deepcopy(θinit), Δtdyn, Δtobs)

# ╔═╡ a859c590-5b3e-4eec-a5b8-047f2c80fac1
begin
	Trixi.TrixiBase.disable_debug_timings()
	X_hlocenkf, θhist = seqassim_trixi(data, Tf, ϵxβ, hlocenkf, deepcopy(X0), model.Ny, model.Nx, t0, sys_advection)
end

# ╔═╡ 461303fb-3211-4c89-a150-1d40cef16c40
with_theme(my_theme) do
	t_start = 3
	tsnap = Observable(t_start)
	
	x_tsnap = @lift(data.xt[:,$tsnap])
	y_tsnap = @lift(data.yt[:,$tsnap])
	X_locenkf_tsnap = @lift(vec(mean(X_hlocenkf[$tsnap+1]; dims = 2)))
	X_ens_tsnap = [@lift(X_hlocenkf[$tsnap+1][:,j]) for j in 1:Ne]
	
	fig = Figure()
	
	ax1 = Axis(fig[1,1])
	
	lines!(ax1, xgrid, x_tsnap, linewidth = 3, label = "Truth")
	lines!(ax1, xgrid, X_locenkf_tsnap, linewidth = 3, label = "LocEnKF")
	for j in 1:Ne
		lines!(ax1, xgrid, X_ens_tsnap[j], linewidth=0.9)
	end
	scatter!(ax1, xgrid[1:Δ:end], y_tsnap)
	
	axislegend(ax1)
	
	
	framerate = 10
	timestamps = range(t_start, Tf, step = 1)
	
	anim = CairoMakie.Makie.Record(fig, timestamps; framerate = framerate) do t
	    tsnap[] = t
	end
	save("assim.gif", anim)
	anim
end

# ╔═╡ 31a7b1b8-84d4-4e65-8c4c-069185103eb7
with_theme(my_theme) do
	tsnap = Observable(1)
	
	y_tsnap = @lift(data.yt[:,$tsnap])
	X_locenkf_tsnap = @lift(mean(X_hlocenkf[$tsnap+1]; dims = 2)[:,1])
	θ_tsnap = @lift(θhist[$tsnap])
	
	
	fig = Figure()
	
	ax1 = Axis(fig[1,1])
	ylims!(ax1, (0.,1e-4))
	
	lines!(ax1, xgrid[3:end-2], θ_tsnap, linewidth = 3, label = L"\theta")
	
	axislegend(ax1)
	
	
	framerate = 10
	timestamps = range(1, Tf, step = 1)
	
	anim = CairoMakie.Makie.Record(fig, timestamps; framerate = framerate) do t
	    tsnap[] = t
	end
	save("assim_theta.gif", anim)
	anim
end

# ╔═╡ Cell order:
# ╠═633dd2fa-2388-4b10-8aea-83754fba1e99
# ╠═beac8a48-22f0-4bca-b60f-d956c92b7706
# ╠═f61f50da-e344-11ef-2c68-b70ecb2e5ab2
# ╠═243f2bf0-cd29-4dcd-a8e6-3bead9a48b74
# ╠═34c27970-71f0-4fe3-9ff3-7c3cdff7b434
# ╟─9c9595eb-a46b-473b-9a03-5feae5de8873
# ╠═7d827a01-37fb-4345-86f7-fd93139db8d0
# ╠═f4eac16c-f5bc-44ff-88d3-81c533b16075
# ╠═9fd61ec9-c7a6-46b7-bdfe-15f8cf58aee2
# ╠═2b458b0f-12ce-4092-a62a-199d62bad3d9
# ╠═4ddffb96-f4b7-4000-97a5-93b9ef4f7952
# ╠═8e466589-9814-4ed6-906e-500197474a92
# ╠═14885cc2-2a74-4b95-a112-fddef6ac240a
# ╠═b4d79689-309b-44ca-b783-6070e8b364ea
# ╠═fa3b9910-b166-4651-8103-e27d22b4aab9
# ╠═2bf04647-61ce-4e04-aa5b-6d51128847a6
# ╟─9d041faa-a058-4552-842d-7305e114e343
# ╠═1e253c95-fad3-4510-b2fc-96f63612208b
# ╠═5aa16815-27e1-415c-a8e7-73243faa748f
# ╠═c3fd8f87-1b79-455f-bc1d-bcae48cb5a68
# ╠═ee5de05d-b9ef-4224-83b7-9ac2deec1adf
# ╠═1f4bd9d4-57a3-42aa-ad14-2f8bb3cac5fa
# ╠═ccdf262b-e080-4571-b4eb-3d236adfb4c3
# ╠═30ee368d-d1c6-428e-b24b-193e4e0a84f0
# ╠═0fd77f8a-5b6a-414d-a340-3d7011dccf0b
# ╠═a859c590-5b3e-4eec-a5b8-047f2c80fac1
# ╠═461303fb-3211-4c89-a150-1d40cef16c40
# ╠═31a7b1b8-84d4-4e65-8c4c-069185103eb7
