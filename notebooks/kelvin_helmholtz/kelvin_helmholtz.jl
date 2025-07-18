using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
using OrdinaryDiffEq
using Trixi
using CairoMakie

# %%
polydeg = 3
dg = DGMulti(polydeg=polydeg, element_type=Quad(), approximation_type=Polynomial(),
    surface_integral=SurfaceIntegralWeakForm(FluxLaxFriedrichs()),
    volume_integral=VolumeIntegralFluxDifferencing(flux_ranocha))

equations = CompressibleEulerEquations2D(1.4)

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

cells_per_dimension = (32, 32)
mesh = DGMultiMesh(dg, cells_per_dimension; periodicity=true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)
cfl = 0.5
summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
stepsize_callback = StepsizeCallback(; cfl)
callbacks = CallbackSet(summary_callback,
    analysis_callback,
    alive_callback,
    stepsize_callback
    # save_solution
)

# %%
###############################################################################
# run the simulation

sol = solve(ode,
    SSPRK43(), callback=callbacks);


# %%
framerate, anim_len = 10, 10
timestamps = range(tspan..., length=framerate * anim_len)
sol_times = sol(timestamps)

##
all_data = map(u -> PlotData2D(u, sol.prob.p)["rho"], sol_times)

##
# rhos = map(x -> first.(x.plot_data.data), all_data)
time = Observable(1)
data_t = @lift(all_data[$time])
fig, ax = plot(data_t, axis=(; aspect=1.), plot_mesh=true)

timestamps = 1:length(sol_times)

record(fig, joinpath(@__DIR__, "time_animation.mp4"), timestamps;
    framerate) do t
    time[] = t
end

# %%
plot(PlotData2D(sol[end÷2], sol.prob.p))

# %%
xy_mat = collect(zip(mesh.md.xyzq...))
# xq_tens = reshape(xq, polydeg + 1, polydeg + 1, :)
# yq_tens = reshape(yq, polydeg + 1, polydeg + 1, :)

# %%
which_scatter = [1, 2, 3, 33, 34, 35]
fig = Figure()
ax = Axis(fig[1, 1], aspect=1.)
cols = Makie.wong_colors()
for (c, col) in enumerate(which_scatter)
    scatter!(xy_mat[:, col], color=cols[c])
    for (j, (x, y)) in enumerate(xy_mat[:, col])
        annotation!(ax, x, y, text=string(mesh.md.mapM[j, col]))
    end
end
# scatter!(xy_mat[:, 32+1])
# for (j, (x, y)) in enumerate(xy_mat[:, 32+1])
#     annotation!(ax, x, y, text=string(mesh.md.mapM[j,32+1]))
# end
fig

# %%

using HierarchicalDA, SparseArrays

PA_order = 3
