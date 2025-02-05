### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 633dd2fa-2388-4b10-8aea-83754fba1e99
begin
	using Pkg
	Pkg.activate("..")
end

# ╔═╡ f61f50da-e344-11ef-2c68-b70ecb2e5ab2
begin
	using Trixi
	using LinearAlgebra
	using OrdinaryDiffEq
	using CairoMakie
end

# ╔═╡ 34c27970-71f0-4fe3-9ff3-7c3cdff7b434
begin
	function initial_condition_sawtooth_fcn(x, t;
		a=-1, b=1, N_saw=3, u_a = 0, u_b = 1)
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
	polydeg = 6
	advection_velocity = 0.1
	equations = LinearScalarAdvectionEquation1D(advection_velocity)
	volume_flux = flux_godunov
	surface_flux = flux_lax_friedrichs
	basis = LobattoLegendreBasis(polydeg)
	
	indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = first)

	volume_integral = VolumeIntegralWeakForm()
	# volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
	volume_integral = VolumeIntegralShockCapturingHG(indicator_sc)
	solver = DGSEM(basis, surface_flux, volume_integral)
	
	N_elements = 100
	coordinates_min, coordinates_max = -1., 1.
	mesh = StructuredMesh((N_elements,), coordinates_min, coordinates_max)
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
	stepsize_callback = StepsizeCallback(cfl = 0.25)
	stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (0.,), variables=(first,))
	sol = solve(ode, CarpenterKennedy2N54(stage_limiter!, williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            callback = stepsize_callback)
end

# ╔═╡ 2b458b0f-12ce-4092-a62a-199d62bad3d9
begin
	dx = (coordinates_max - coordinates_min) / N_elements
	mesh_x = Matrix{Float64}(undef, length(basis.nodes), N_elements)
	for element in 1:N_elements
	    x_l = -1 + (element - 1) * dx + dx/2
	    for i in eachindex(basis.nodes) # basis points in [-1, 1]
	        ξ = basis.nodes[i]
	        mesh_x[i, element] = x_l + dx/2 * ξ
	    end
	end
end

# ╔═╡ 4ddffb96-f4b7-4000-97a5-93b9ef4f7952
begin
	ut = t->map(x->x[], vec(sol(t)))
	time = Observable(0.0)
	ys = @lift(ut($time))
	fig, ax, pl = lines(vec(mesh_x), ys, axis=(;xlabel=L"x", ylabel=L"u(t,x)", title=@lift("t = $($time)")))
	N_T = 101
	timestamps = range(0,tspan[end], N_T)
	anim = CairoMakie.Makie.Record(fig, timestamps; framerate=N_T÷4) do t
		time[] = t
	end
	save("solution.gif", anim)
	anim
end

# ╔═╡ Cell order:
# ╠═633dd2fa-2388-4b10-8aea-83754fba1e99
# ╠═f61f50da-e344-11ef-2c68-b70ecb2e5ab2
# ╠═34c27970-71f0-4fe3-9ff3-7c3cdff7b434
# ╟─9c9595eb-a46b-473b-9a03-5feae5de8873
# ╠═7d827a01-37fb-4345-86f7-fd93139db8d0
# ╠═9fd61ec9-c7a6-46b7-bdfe-15f8cf58aee2
# ╠═2b458b0f-12ce-4092-a62a-199d62bad3d9
# ╠═4ddffb96-f4b7-4000-97a5-93b9ef4f7952
