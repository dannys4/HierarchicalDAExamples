using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using CairoMakie
using HierarchicalDA
using JLD2
using Trixi
using StaticArrays
using Statistics

# %%
function get_file_path(
    algo::AbstractString,
    time_idx::Union{Int,Nothing}=nothing,
    dir_path::AbstractString=joinpath(@__DIR__, "data"),
)
    all_subdirs = readdir(dir_path)
    subdirs = filter(j -> startswith(all_subdirs[j], algo), eachindex(all_subdirs))
    all_subdirs_join = readdir(dir_path, join=true)[subdirs]
    most_recent_subdir_idx = argmax(mtime.(all_subdirs_join))
    subdir_path = all_subdirs_join[most_recent_subdir_idx]
    all_data = readdir(subdir_path)
    all_indices = map(s -> s[2:end-5], all_data)
    time_idx_file = nothing
    if isnothing(time_idx)
        int_idxs = parse.((Int,), all_indices)
        time_idx_file = sortperm(int_idxs)[end]
    else
        time_idx_file = findfirst(==(string(time_idx)), idxs)
    end
    joinpath(subdir_path, all_data[time_idx_file])
end

# %%
polydeg, Ncells_dim = 3, 24
sys_kpp = setup_kpp(polydeg, Ncells_dim)

# %%
hloc_file_path = get_file_path("HLocEnKF")
loc_file_path = get_file_path("LocEnKF")

# %%
hlocenkf_file = jldopen(hloc_file_path, "r")
X_hlocenkf = hlocenkf_file["X"]
θ_hlocenkf = hlocenkf_file["θ"]
close(hlocenkf_file)

# %%
@load loc_file_path X
X_locenkf = X
X = nothing

# %%
itp_hlocenkf = ensemble_to_itp(X_hlocenkf, sys_kpp)
itp_hlocenkf_θ = ensemble_to_itp(reshape(log10.(θ_hlocenkf), :, 2), sys_kpp)
itp_locenkf = ensemble_to_itp(X_locenkf, sys_kpp)

# fig_means = plot_heatmaps(map(x -> mean(x, dims=3)[:, :], [itp_hlocenkf, itp_locenkf]), ["GSBL-DA, mean", "LocEnKF, mean"], sys_kpp)
# save(joinpath(@__DIR__, "figs/kpp_means1.png"), fig_means)

# %%
fig_samples = trixiheatmaps(map(x -> x[:, :, 1], [itp_hlocenkf, itp_locenkf]), ["GSBL-DA sample", "LocEnKF sample"], sys_kpp)
save(joinpath(@__DIR__, "figs/kpp_samples1.png"), fig_samples)


# %%
fig_theta = trixiheatmaps(eachslice(itp_hlocenkf_θ, dims=3), ["θ_x, log-scale", "θ_y, log-scale"], sys_kpp)
save(joinpath(@__DIR__, "figs/kpp_theta1.png"), fig_theta)

# %%
pd_sol = PlotData2D(mean(itp_hlocenkf, dims=3)[:, :], sys_kpp.semi)
plot(pd_sol)

# %%
pd_sol = PlotData2D(itp_hlocenkf_θ[:, :, 2], sys_kpp.semi)
plot(pd_sol)

# %%
pd_sol = PlotData2D(mean(itp_locenkf, dims=3)[:, :], sys_kpp.semi)
plot(pd_sol)