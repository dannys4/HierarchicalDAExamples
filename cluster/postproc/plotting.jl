# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Julia 1.11.5
#     language: julia
#     name: julia-1.11
# ---

# %%
using Pkg;
Pkg.activate(joinpath(@__DIR__, "..", ".."))

# %%
using DataFrames, JLD2, Distributions, ProgressMeter, CairoMakie

# %%
data_path = joinpath(@__DIR__, "data")
example_name = "burgers"
filename = example_name * ".jld2"

# %%
@load joinpath(data_path, filename) df_truth df_algs df_params

# %%
df = leftjoin(df_algs, df_params, on=:id)
leftjoin!(df, df_truth, on=:id)

##
with_theme(theme_minimal(), linewidth=3) do
    cols = Makie.wong_colors()
    for (truth_sym, qoi_sym, ylab) in [(:true_tv_norm, :tv, "TV norm"), (:true_mass, :mass, "Mass"), (:true_entropy, :entropy, "Entropy")]
        truth_sym in propertynames(df) || continue
        qoi_true = vec(df[1, truth_sym])
        qoi_gsbl = Vector{Float64}[]
        qoi_enkf = Vector{Float64}[]
        for row in eachrow(df)
            push_val = row[qoi_sym]
            if push_val isa AbstractVector{<:AbstractVector}
                # Get ensemble average
                push_val = map(mean, push_val)
            end
            which_push = row[:algorithm] == "locenkf" ? qoi_enkf : qoi_gsbl
            push!(which_push, push_val)
        end
        qoi_enkf_mat = reduce(hcat, qoi_enkf)
        qoi_enkf_mat[isnan.(qoi_enkf_mat)] .= Inf
        quants_enkf = quantile.(eachrow(qoi_enkf_mat), ([0.5, 0.1, 0.9],))
        medians_enkf = first.(quants_enkf)[2:end]
        bars_enkf = map(x -> (x[2], x[3]), quants_enkf)[2:end]

        qoi_gsbl_mat = reduce(hcat, qoi_gsbl)
        qoi_gsbl_mat[isnan.(qoi_gsbl_mat)] .= Inf
        quants_gsbl = quantile.(eachrow(qoi_gsbl_mat), ([0.5, 0.1, 0.9],))
        medians_gsbl = first.(quants_gsbl)[2:end]
        bars_gsbl = map(x -> (x[2], x[3]), quants_gsbl)[2:end]
        times = 1:size(medians_enkf,1)

        fig = Figure()
        ax = Axis(fig[1, 1], xlabel="Time", ylabel=ylab)
        lines!(times, medians_enkf, color=cols[1])
        rangebars!(times, bars_enkf, linewidth=8, color=(cols[1], 0.4), label="EnKF")
        lines!(times, medians_gsbl, color=cols[2])
        rangebars!(times, bars_gsbl, linewidth=8, color=(cols[2], 0.4), label="GSBL")
        lines!(times, qoi_true, linestyle=:dot, color=cols[3], label="truth")
        axislegend(position=:lt, orientation=:horizontal)
        fig_name = example_name * "_" * join(split(ylab), "_") * ".pdf"
        save(joinpath(@__DIR__, "figs", fig_name), fig)
        display(fig)
    end
end

##
# Filter cols
rm_names = ["random_seed", "id"]
df_cols = names(df)
one_unique = v -> (length(unique(v)) == 1)

keep_cols = filter(axes(df, 2)) do j
    !(df_cols[j] in rm_names || one_unique(df[!, j]))
end

df_cut = select(df, keep_cols)

# %%
function plot_convergence!(ax, df, fixed_key, fixed_val, x_axis, y_axis, marker_diffs, line_style, cols=Makie.wong_colors())
    line_styles = (:dash, :solid, :dot) # Should only need 2
    marker_color = ((:+, cols[1]), (:rect, cols[2]), (:star5, cols[3]))
    metric_quantile_levels = (0.05, 0.5, 0.95)

    df_filter = subset(df, fixed_key => v -> v .== fixed_val)
    styles, markers = [unique(df_filter[!, col]) for col in [line_style, marker_diffs]]
    line_styles = Dict(val => line_styles[idx] for (idx, val) in enumerate(styles))
    marker_color = Dict(val => marker_color[idx] for (idx, val) in enumerate(markers))

    g_df_filter = groupby(df_filter, [line_style, marker_diffs, x_axis])
    df_filter_metrics = combine(g_df_filter,
        [y_axis] =>
            ((v,) -> NamedTuple(zip([:lo, :mid, :hi], quantile(map(getindex, v), metric_quantile_levels)))) =>
                AsTable
    )
    sort!(df_filter_metrics, x_axis)
    df_metrics_g = groupby(df_filter_metrics, [marker_diffs, line_style])

    plots_v, labels_v = [], []
    for (key, subdf) in pairs(df_metrics_g)
        md, ls = getproperty.((key,), [marker_diffs, line_style])

        marker, color = marker_color[md]
        linestyle = line_styles[ls]
        label = "$marker_diffs = $md, $ls"
        li = lines!(ax, subdf[!, x_axis], subdf[!, :mid], linewidth=3; linestyle, color)
        sc = scatter!(ax, subdf[!, x_axis], subdf[!, :mid], markersize=18; marker, color)
        err_lo = subdf[!, :mid] - subdf[!, :lo]
        err_hi = subdf[!, :hi] - subdf[!, :mid]
        errorbars!(ax, subdf[!, x_axis], subdf[!, :mid], err_lo, err_hi; color, whiskerwidth=10)
        push!(labels_v, label)
        push!(plots_v, [li, sc])
    end
    plots_v, labels_v
end

# %%
fixed_key = :Ne
x_axis = :sigma_y
y_axis = :crps2
marker_diffs = :delta_y
line_style = :algorithm

fixed_vals = filter(val -> sum(df_cut[!, fixed_key] .== val) > 50, unique(df_cut[!, fixed_key]))
fixed_val = fixed_vals[2]
n_row = floor(Int, sqrt(length(fixed_vals)) + sqrt(eps()))
fixed_vals_row = [mod1(idx, n_row) for idx in eachindex(fixed_vals)]
fixed_vals_col = [((idx - 1) ÷ n_row) + 1 for idx in eachindex(fixed_vals)]
n_col = maximum(fixed_vals_col)

function get_ylabel(y_axis)
    metric = match(r"^[a-z]*", string(y_axis)).match
    upper_fcn = metric in ["crps", "rmse"] ? uppercase : uppercasefirst
    upper_fcn(metric) * ", log-scale"
end

# %%
xticks = [(val, string(val)) for val in [0.01, 0.025, 0.05, 0.1]]
xminorticks = 0.01 * (1:0.5:10)
with_theme(theme_latexfonts(), figure_padding = 0., Axis=(
    aspect=1, xlabel=L"$\sigma_y$, log-scale", ylabel=get_ylabel(y_axis),
    xscale=log10, yscale=log10,
    xticks=first.(xticks),
    xticklabels=last.(xticks),
    xminorticks=xminorticks, xminorticksvisible=true)) do

    fig = Figure(size=(300*n_col, 300*n_row + 50))
    gl_plot = fig[1:2, 1:2] = GridLayout()
    gl_plot_rows = [(gl_plot[j, 1] = GridLayout()) for j in 1:n_row]
    gl_legend = fig[3, 1:2] = GridLayout()
    # ax = Axis(fig[1, 1], aspect=1., xlabel=string(x_axis), ylabel=string(y_axis), title="$fixed_key = $fixed_val")
    plots_v = labels_v = nothing
    for (idx, val) in enumerate(fixed_vals)
        row, col = fixed_vals_row[idx], fixed_vals_col[idx]
        gl_row = gl_plot_rows[row]
        ax = Axis(gl_row[1, col], title="Ensemble size $val", aspect=1.)
        ax.xminorticks = xminorticks
        plots_v, labels_v = plot_convergence!(ax, df_cut, fixed_key, val, x_axis, y_axis, marker_diffs, line_style)
    end
    function proc_label(label)
        _, _, delta, alg = split(label)
        alg_name = alg == "locenkf" ? "EnKF" : "GSBL"
        delta = delta[1:end-1]
        L"%$alg_name, $\Delta y$ = %$delta"
    end
    leg = Legend(
        gl_legend[1, 1:2], plots_v, proc_label.(labels_v),
        patchsize=(50, 20), orientation=:horizontal)
    leg.nbanks = 2
    fig_name = join(vcat(split(filename, "_")[1:end-1], string(y_axis), "convergence.pdf"), "_")
    save(joinpath(@__DIR__, "figs", fig_name), fig)
    fig
end

# %%