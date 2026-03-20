using Trixi
using LinearAlgebra
using OrdinaryDiffEq
using HierarchicalDA
using LinearMaps
using TransportBasedInference2
using HierarchicalDA
using Distributions
using SparseArrays
using JLD2
using Dates
using Random

# %%
polydeg, Ncells = 3, 100
sys_burgers = setup_burgers(polydeg, Ncells)
xgrid_a, xgrid_b = -1, 1
xgrid = vec(sys_burgers.mesh.md.xq)

# %%
# xgrid_a, xgrid_b, Ncells = -1, 1, 100
# xgrid = range(xgrid_a, xgrid_b, Ncells + 2)[2:end-1]

# %%
grid_len = xgrid_b - xgrid_a
xgrid_circ = [xgrid; (xgrid[1] + grid_len)]
h_plus = xgrid_circ[2:end] - xgrid_circ[1:end-1]
h_minus = [h_plus[end]; h_plus[1:end-1]]
h_sum = h_plus + h_minus
ratio_plus = h_plus ./ h_sum
ratio_plus = [ratio_plus[2:end]; ratio_plus[1]] # Reorder since it
ratio_minus = h_minus ./ h_sum
denom = @. 0.5 * h_plus * h_minus
d_sq_op = Tridiagonal(ratio_plus, -ones(length(xgrid_circ)), ratio_minus)
d_wrap = sparse(d_sq_op[1:end-1, 1:end-1] ./ denom)
d_wrap[1, end], d_wrap[end, 1] = d_sq_op[end, end-1] / denom[1], d_sq_op[end-1, end] / denom[end]

# %%
sin_eval = sin.(pi * xgrid)
d2_sin = d_wrap * sin_eval
d2_sin_true = -((pi)^2)*sin_eval

# %%
using CairoMakie
fig = Figure()
ax = Axis(fig[1,1], yscale=log10)
# lines!(xgrid, sin_eval, label="Eval")
lines!(xgrid, abs2.(d2_sin - d2_sin_true), label="D2")
# lines!(xgrid, d2_sin_true, label="True d2")
# axislegend()
display(fig)

# %%
# rd = sys_burgers.dg.basis
# md = sys_burgers.mesh.md
# uq = Trixi.allocate_coefficients(Trixi.mesh_equations_solver_cache(sys_burgers.semi)...)
# u = similar(uq)
# duq_dx = similar(uq)

# %%
# %%
# u_quad = Trixi.allocate_coefficients(Trixi.mesh_equations_solver_cache(sys_burgers.semi)...)
# u_itp = similar(u_quad)
# uf_cache1 = similar(u_quad, (2, size(u_quad, 2)))
# uf_cache2 = similar(uf_cache1)
# u_cache = similar(u_quad)
# workspace = (; u_quad, u_itp, u_cache, uf_cache1, uf_cache2)
diff_map = DGMultiDiff1D(sys_burgers, true)

# %%

u_quad_vec = sin.(pi * xgrid)
# du_quad_dx_vec = similar(u_quad_vec)
# dgmulti_diff_1d!(du_quad_dx_vec, u_quad_vec, sys_burgers, true, workspace)
du_quad_dx_vec = diff_map * u_quad_vec

# %%
# diff_op! = (out, in) -> dgmulti_diff_1d!(out, in, sys_burgers, true, workspace)


# %%
using CairoMakie
f, ax, _ = lines(xgrid, du_quad_dx_vec, linewidth=3)
lines!(ax, xgrid, pi * cos.(pi * xgrid), linestyle=:dash, linewidth=3)
f


# %%
struct SmoothSigmoid
    dirichlet_boundary_values::Vector{Tuple{Float64,Float64}}
    grid_left::Vector{Float64}
    grid_right::Vector{Float64}
    Nvar::Int
    shape_scale::Float64
    shape_shift::Float64
end

# %%
using Distributions
x_lo, x_hi = -5., 5.
shift_dist = (x_hi - x_lo) * Normal(0.5, 0.2) + x_lo
log_scale_dist = Normal(-1, 1/2)
sigmoid = x -> 1 / (1 + exp(-x))
xgrid = -5:0.01:5
fig = Figure()
ax = Axis(fig[1,1])
cols = Makie.wong_colors()
# x_norm = (xgrid .- x_lo) / (x_hi - x_lo)
for idx in 1:1000
    # alpha, beta = exp.(0.5*randn(2))
    # dd = Beta(alpha, beta)
    # sample = 1 .- cdf.(dd, x_norm)
    shift, scale = rand(shift_dist), exp(rand(log_scale_dist))
    sample = sigmoid.(scale*(xgrid .- shift))
    sample .-= sample[1]
    sample ./= sample[end]
    sample = 1 .- sample
    if idx % 50 == 0
        lines!(xgrid, sample, color=(cols[2], 1.), linewidth=3)
    else
        lines!(xgrid, sample, color=(cols[1], 0.1), linewidth=3)
    end
end
fig