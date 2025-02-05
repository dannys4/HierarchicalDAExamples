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
end

# ╔═╡ f4eac16c-f5bc-44ff-88d3-81c533b16075
begin
	using Trixi: entropy2cons
	function Trixi.entropy2cons(t, ::LinearScalarAdvectionEquation1D)
		t
	end
end

# ╔═╡ c7ae48f6-b72b-4103-853f-9e4052dc28ff
using HierarchicalDA, LinearMaps, TransportBasedInference2, Distributions, SparseArrays

# ╔═╡ 67b1362b-a400-4550-8c91-442acac4ba16
using StaticArrays

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
lines(-1:0.01:1,initial_condition_sawtooth_fcn.(-1:0.01:1.,(nothing,)), axis=(;xlabel=L"x", ylabel=L"u(0,x)", title="Initial condition"))

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

# ╔═╡ cd3c9d4c-54f7-4910-8950-6f6426613924
@which solve(ode, SSPRK43(), adaptive=true, dense=false, save_everystep=false, callback = stepsize_callback)

# ╔═╡ df929046-fb9f-4769-8a15-f5ff13ecaf23
typeof(ode)

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
begin
	ut = t->map(x->x[], vec(sol(t)))
	time = Observable(0.0)
	ys = @lift(ut($time))
	fig, ax, pl = lines(xgrid, ys, axis=(;xlabel=L"x", ylabel=L"u(t,x)", title=@lift("t = $($time)")))
	# hlines!(vec(mesh_x), fill(0.05, length(mesh_x)))
	N_T = 101
	timestamps = range(0,tspan[end], N_T)
	anim = CairoMakie.Makie.Record(fig, timestamps; framerate=N_T÷4) do t
		time[] = t
	end
	save("solution.gif", anim)
	anim
end

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
	Δtdyn = 0.01
	Δtobs = 0.05
	t0 = 0.0
	Tf = 100
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

# ╔═╡ 946ab4ed-d3be-4ac9-8f6f-deeff7e368a5
σy

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

# ╔═╡ 04070ff2-9b7f-4ac7-9a05-3a5d2a12a8b5
# begin
# 	x_ode = Trixi.allocate_coefficients(Trixi.mesh_equations_solver_cache(sys_advection.semi)...)
# 	for idx in eachindex(x_ode)
# 		x_ode[idx] = @SVector[1.]
# 	end
# 	C = x_ode
# 	A = basis.Pq
# 	B = x_ode
# 	α, β = true, false
# 	# LinearAlgebra.generic_matmatmul!(
#  #        C,
#  #        LinearAlgebra.wrapper_char(A),
#  #        LinearAlgebra.wrapper_char(B),
#  #        LinearAlgebra._unwrap(A),
# 	# 	LinearAlgebra._unwrap(B),
#  #        LinearAlgebra.MulAddMul(α, β)
#  #    )
# 	@which mul!(C, A, B)
# end

# ╔═╡ 9d041faa-a058-4552-842d-7305e114e343
begin
	lines(xgrid, reduce(vcat, vec(ode.u0)))
	lines!(xgrid, u0, linestyle=:dash)
	scatter!(xgrid, data.xt[:,end÷2])
	Makie.current_figure()
end

# ╔═╡ 1e253c95-fad3-4510-b2fc-96f63612208b
sys_advection.dg.basis.Pq

# ╔═╡ Cell order:
# ╠═633dd2fa-2388-4b10-8aea-83754fba1e99
# ╠═beac8a48-22f0-4bca-b60f-d956c92b7706
# ╠═f61f50da-e344-11ef-2c68-b70ecb2e5ab2
# ╠═34c27970-71f0-4fe3-9ff3-7c3cdff7b434
# ╟─9c9595eb-a46b-473b-9a03-5feae5de8873
# ╠═7d827a01-37fb-4345-86f7-fd93139db8d0
# ╠═f4eac16c-f5bc-44ff-88d3-81c533b16075
# ╠═9fd61ec9-c7a6-46b7-bdfe-15f8cf58aee2
# ╠═cd3c9d4c-54f7-4910-8950-6f6426613924
# ╠═df929046-fb9f-4769-8a15-f5ff13ecaf23
# ╠═2b458b0f-12ce-4092-a62a-199d62bad3d9
# ╠═4ddffb96-f4b7-4000-97a5-93b9ef4f7952
# ╠═c7ae48f6-b72b-4103-853f-9e4052dc28ff
# ╠═8e466589-9814-4ed6-906e-500197474a92
# ╠═14885cc2-2a74-4b95-a112-fddef6ac240a
# ╠═b4d79689-309b-44ca-b783-6070e8b364ea
# ╠═946ab4ed-d3be-4ac9-8f6f-deeff7e368a5
# ╠═fa3b9910-b166-4651-8103-e27d22b4aab9
# ╠═2bf04647-61ce-4e04-aa5b-6d51128847a6
# ╠═67b1362b-a400-4550-8c91-442acac4ba16
# ╠═04070ff2-9b7f-4ac7-9a05-3a5d2a12a8b5
# ╠═9d041faa-a058-4552-842d-7305e114e343
# ╠═1e253c95-fad3-4510-b2fc-96f63612208b
