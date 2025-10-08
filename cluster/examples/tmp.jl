using HierarchicalDA
using JLD2
using CairoMakie
using Trixi
using StaticArrays

# %%
function get_file_path(
    algo::AbstractString,
    time_idx::Union{Int,Nothing},
    dir_path::AbstractString=joinpath(@__DIR__, "data"),
)
    files = filter(x -> startswith(x, algo), readdir(dir_path))
    idxs = map(f -> split(f, "_")[2][2:end], files)
    time_idx_file = nothing
    if isnothing(time_idx)
        int_idxs = parse.((Int,), idxs)
        time_idx_file = sortperm(int_idxs)[end]
    else
        time_idx_file = findfirst(==(string(time_idx)), idxs)
    end
    joinpath(dir_path, files[time_idx_file])
end


# %%
hloc_file_path = get_file_path("HLocEnKF", nothing)
# loc_file_path = get_file_path("LocEnKF", nothing)

# %%
hlocenkf_file = jldopen(hloc_file_path, "r")
X_hlocenkf = hlocenkf_file["X"]
θ_hlocenkf = hlocenkf_file["θ"]
close(hlocenkf_file)

# %%
# @load loc_file_path X
# X_locenkf = X
X = nothing

# %%
polydeg, Ncells_dim = 2, 48
sys_kpp = setup_kpp(polydeg, Ncells_dim)

# %%
plot_ens = SVector{1}.(reshape(@view(X_hlocenkf[:, 3]), (polydeg + 1)^2, Ncells_dim^2))
pd_sol = PlotData2D(plot_ens, sys_kpp.semi)
plot(pd_sol)

# %%
# plot_ens = SVector{1}.(reshape(@view(X_locenkf[:, 1]), (polydeg + 1)^2, Ncells_dim^2))
# pd_sol = PlotData2D(plot_ens, sys_kpp.semi)
# plot(pd_sol)