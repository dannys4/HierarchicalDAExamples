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
# loc_file_path, _ = get_file_path("LocEnKF", sim_subdir, time_idx)
@info "" sim_subdir time_idx hloc_file_path # loc_file_path
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
# itp_data = repeat(itp_data, 1, 1, size(X_hlocenkf, 2))
itp_hlocenkf = ensemble_to_itp(X_hlocenkf, sys_kpp)
itp_hlocenkf_θ = ensemble_to_itp(reshape(log10.(θ_hlocenkf), :, 2), sys_kpp)
itp_locenkf = ensemble_to_itp(X_locenkf, sys_kpp)

# %%
fig_samples, _ = trixiheatmaps(map(x -> x[:, :, 1], [itp_hlocenkf, itp_locenkf, itp_data]), ["GSBL-DA sample", "EnKF sample", "Solution"], sys_kpp)
display(fig_samples)
save(joinpath(@__DIR__, "figs", "kpp", "samples_t$(time_idx).png"), fig_samples)

# %%
mean_hlocenkf, mean_locenkf = map(x -> (mean(x, dims=3)-itp_data[:, :, 1])[:, :], (itp_hlocenkf, itp_locenkf))
mean_hlocenkf, mean_locenkf = [map(x -> log10.(abs.(x)), mm) for mm in (mean_hlocenkf, mean_locenkf)]
fig_means, _ = trixiheatmaps([mean_hlocenkf, mean_locenkf], ["GSBL-DA mean", "LocEnKF mean"], sys_kpp)
display(fig_means)
save(joinpath(@__DIR__, "figs", "kpp", "mean_error_t$(time_idx).png"), fig_means)

# %%
fig_theta, _ = trixiheatmaps([eachslice(itp_hlocenkf_θ, dims=3)..., itp_data[:, :, 1]], ["θ_x, log-scale", "θ_y, log-scale", "Solution"], sys_kpp)
display(fig_theta)
save(joinpath(@__DIR__, "figs", "kpp", "theta_t$(time_idx).png"), fig_theta)

# %%
itp_data_full = ensemble_to_itp(data.xt, sys_kpp)
plot_ts = [1, 20, 40]
plot_data = [itp_data_full[:, :, t] for t in plot_ts]
fig_data, axs_data = trixiheatmaps(plot_data, [L"u(x,%$t)" for t in data.tt[plot_ts]], sys_kpp)
# display(fig_data)

num_obs_dim = Int(sqrt(size(data.yt, 1)))
xq = sort(unique(x -> round(x, digits=5), vec(sys_kpp.mesh.md.xq)))
yq = sort(unique(x -> round(x, digits=5), vec(sys_kpp.mesh.md.yq)))
delta_obs = Int(length(xq) / num_obs_dim)
x_obs, y_obs = [qq[((delta_obs+1)÷2):delta_obs:end] for qq in (xq, yq)]
y_obs = repeat(y_obs, 1, length(y_obs))
x_obs = repeat(x_obs', length(x_obs), 1)
for (j, t) in enumerate(plot_ts)
    scatter!(axs_data[j], vec(y_obs) .- 5e-2, vec(x_obs) .- 5e-2, color=data.yt[:, t])
end
display(fig_data)
# save(joinpath(@__DIR__, "figs", "kpp", "soln.pdf"), fig_data)
save(joinpath(@__DIR__, "figs", "kpp", "soln.png"), fig_data)
