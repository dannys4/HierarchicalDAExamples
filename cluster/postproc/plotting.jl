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
names(df)

# %%
df_cut = select(df, filter(j -> names(df)[j] != "random_seed" && length(unique(df[:, j])) > 1, axes(df, 2)))

# %%
df_g = groupby(df_cut, ["Ne", "delta_y", "algorithm", "sigma_x_filter"])

# %%
unique(df[!, :Ne])

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
        push!(labels_v, label)
        push!(plots_v, [li, sc])
    end
    plots_v, labels_v
end

cols = Makie.wong_colors()

fixed_key = :Ne
x_axis = :sigma_x_filter
y_axis = :rmse2
marker_diffs = :delta_y
line_style = :algorithm

fixed_vals = unique(df_cut[!, fixed_key])
n_row = floor(Int, sqrt(length(fixed_vals)))

fig = Figure(size=(800, 500))
ax = Axis(fig[1, 1], aspect=1., xlabel=string(x_axis), ylabel=string(y_axis), title="$fixed_key = $fixed_val")
plots_v, labels_v = plot_convergence!(ax, df_cut, fixed_key, fixed_val, x_axis, y_axis, marker_diffs, line_style)
Legend(fig[1, 2], plots_v, labels_v, patchsize=(35, 35))
fig

# %%