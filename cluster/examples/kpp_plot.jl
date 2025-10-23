using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using CairoMakie
using HierarchicalDA
using JLD2
using Trixi
using StaticArrays
using Statistics

# %%
function most_recent_subdir(path, subdir_start)
    all_subdirs = readdir(path)
    subdirs = eachindex(all_subdirs)[startswith.(all_subdirs, (subdir_start,))]
    all_subdirs_join = readdir(path, join=true)[subdirs]
    most_recent_subdir = argmax(mtime, all_subdirs_join)
    most_recent_subdir
end

function get_file_path(
    algo::AbstractString,
    data_path::AbstractString,
    time_idx::Union{Int,Nothing}=nothing
)
    subdir_path = most_recent_subdir(data_path, algo)
    all_data = readdir(subdir_path)
    int_idxs = map(s -> parse(Int, s[2:end-5]), all_data)
    time_idx_file = nothing
    if isnothing(time_idx)
        time_idx_file = sortperm(int_idxs)[end]
        time_idx = maximum(int_idxs)
    else
        time_idx_file = findfirst(==(time_idx), int_idxs)
    end
    joinpath(subdir_path, all_data[time_idx_file]), time_idx
end

# %%
data_path = joinpath(@__DIR__, "data")
sim_subdir = most_recent_subdir(data_path, "sim")
hloc_file_path, time_idx = get_file_path("HLocEnKF", sim_subdir)
loc_file_path, _ = get_file_path("LocEnKF", sim_subdir, time_idx)
@info "" sim_subdir time_idx hloc_file_path loc_file_path
@load joinpath(sim_subdir, "data.jld2") data polydeg Ncells_dim
sys_kpp = setup_kpp(polydeg, Ncells_dim);

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
itp_data = ensemble_to_itp(data.xt[:, time_idx:time_idx], sys_kpp)
itp_hlocenkf = ensemble_to_itp(X_hlocenkf, sys_kpp)
# itp_hlocenkf_θ = ensemble_to_itp(reshape(log10.(θ_hlocenkf), :, 2), sys_kpp)
itp_locenkf = ensemble_to_itp(X_locenkf, sys_kpp)

# %%
fig_samples = trixiheatmaps(map(x -> x[:, :, 1], [itp_hlocenkf, itp_locenkf, itp_data]), ["GSBL-DA sample", "EnKF sample", "Solution"], sys_kpp)
display(fig_samples)
save(joinpath(@__DIR__, "figs/kpp_samples1.png"), fig_samples)

# %%
mean_hlocenkf, mean_locenkf = map(x -> mean(x, dims=3)[:, :], (itp_hlocenkf, itp_locenkf))
fig_means = trixiheatmaps([mean_hlocenkf - itp_data, mean_locenkf - itp_data], ["GSBL-DA sample", "LocEnKF sample"], sys_kpp)
display(fig_means)
# save(joinpath(@__DIR__, "figs/kpp_means1.png"), fig_means)

# %%
fig_theta = trixiheatmaps([eachslice(itp_hlocenkf_θ, dims=3)..., itp_data], ["θ_x, log-scale", "θ_y, log-scale", "Solution"], sys_kpp)
display(fig_theta)
# save(joinpath(@__DIR__, "figs/kpp_theta1.png"), fig_theta)

# %%
num_obs_dim = Int(sqrt(size(data.yt, 1)))
xq = sort(unique(x -> round(x, digits=5), vec(sys_kpp.mesh.md.xq)))
yq = sort(unique(x -> round(x, digits=5), vec(sys_kpp.mesh.md.yq)))
delta_obs = Int(length(xq) / num_obs_dim)
heatmap(xq[1+((delta_obs-1)÷2):delta_obs:end], yq[1+((delta_obs-1)÷2):delta_obs:end], reshape(data.yt[:, time_idx], num_obs_dim, :), axis=(aspect=1., limits=(-2, 2, -2, 2), title="Observation"))