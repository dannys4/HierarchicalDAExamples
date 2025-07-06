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
using DataFrames, JLD2, Distributions, ProgressMeter, CairoMakie

# %%
data_path = joinpath(@__DIR__, "data")
filename = "burgers_df.jld2"

# %%
@load joinpath(data_path, filename) df


# %%
df_cut = select(df, filter(j -> names(df)[j] != "random_seed" && length(unique(df[:, j])) > 1, axes(df, 2)))

# %%
function plot_convergence!(ax, df, fixed_key, fixed_val, x_axis, y_axis, marker_diffs, line_style, cols=Makie.wong_colors())
    line_styles = (:dash, :solid, :dot) # Should only need 2
    marker_color = ((:+, cols[1]), (:rect, cols[2]), (:star5, cols[3]))
    metric_quantile_levels = (0.1, 0.5, 0.9)

    df_filter = subset(df, fixed_key => v -> v .== fixed_val)
    styles, markers = [unique(df_filter[!, col]) for col in [line_style, marker_diffs]]
    line_styles = Dict(val => line_styles[idx] for (idx, val) in enumerate(styles))
    marker_color = Dict(val => marker_color[idx] for (idx, val) in enumerate(markers))

    g_df_filter = groupby(df_filter, [line_style, marker_diffs, x_axis])
    df_filter_metrics = combine(g_df_filter,
        [y_axis] =>
            ((v,) -> NamedTuple(zip([:lo, :mid, :hi], quantile(v, metric_quantile_levels)))) =>
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
x_axis = :sigma_x_filter
y_axis = :rmse2
marker_diffs = :delta_y
line_style = :algorithm

fixed_vals = filter(val -> sum(df_cut[!, fixed_key] .== val) > 50, unique(df_cut[!, fixed_key]))
fixed_val = fixed_vals[2]
n_row = floor(Int, sqrt(length(fixed_vals)))
fixed_val_pos = [val => (((idx - 1) ÷ n_row) + 1, mod1(idx, n_row)) for (idx, val) in enumerate(fixed_vals)]
function get_ylabel(y_axis)
    metric = match(r"^[a-z]*", string(y_axis)).match
    upper_fcn = metric in ["crps", "rmse"] ? uppercase : uppercasefirst
    upper_fcn(metric) * ", log-scale"
end

# %%
xticks = [(val, string(val)) for val in [0.01, 0.025, 0.05, 0.1]]
xminorticks = 0.01 * (1:0.5:10)
with_theme(theme_latexfonts(), Axis=(
    aspect=1, xlabel=L"$\sigma_x$, log-scale", ylabel=get_ylabel(y_axis),
    xscale=log10, yscale=log10,
    xticks=first.(xticks),
    xticklabels=last.(xticks),
    xminorticks=xminorticks, xminorticksvisible=true)) do

    fig = Figure(size=(800, 900))
    gl_plot = fig[1:2, 1:2] = GridLayout()
    gl_legend = fig[3, 1:2] = GridLayout()
    # ax = Axis(fig[1, 1], aspect=1., xlabel=string(x_axis), ylabel=string(y_axis), title="$fixed_key = $fixed_val")
    plots_v = labels_v = nothing
    for (fixed_val, ax_pos) in fixed_val_pos
        ax = Axis(gl_plot[ax_pos...], title="Ensemble size $fixed_val")
        ax.xminorticks = xminorticks
        plots_v, labels_v = plot_convergence!(ax, df_cut, fixed_key, fixed_val, x_axis, y_axis, marker_diffs, line_style)
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
    save(joinpath(@__DIR__, "figs", "$(y_axis)_convergence.pdf"), fig)
    fig
end

# %%