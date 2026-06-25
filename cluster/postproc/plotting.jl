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
example_name = "sod_618"
filename = example_name * "_df.jld2"

# %%
@load joinpath(data_path, filename) df_truth df_algs df_params

# %%
df = leftjoin(df_algs, df_params, on=:id)
leftjoin!(df, df_truth, on=:id)

##
false && with_theme(theme_minimal(), linewidth=3) do
    cols = Makie.wong_colors()
    for (truth_sym, qoi_sym, ylab) in [(:true_tv_norm, :tv, "TV norm"), (:true_mass, :mass, "Mass"), (:true_entropy, :entropy, "Entropy")]
        qoi_true = df[1, truth_sym]
        qoi_gsbl = []
        qoi_enkf = []
        for row in eachrow(df)
            which_push = row[:algorithm] == "locenkf" ? qoi_enkf : qoi_gsbl
            push!(which_push, row[qoi_sym])
        end
        push!(Main.AA, (qoi_true, qoi_gsbl, qoi_enkf))
        qoi_enkf_mat = reduce(vcat, qoi_enkf)
        qoi_enkf_mat[isnan.(qoi_enkf_mat)] .= Inf
        quants_enkf = quantile.(eachcol(qoi_enkf_mat), ([0.5, 0.1, 0.9],))
        medians_enkf = first.(quants_enkf)[2:end]
        bars_enkf = map(x -> (x[2], x[3]), quants_enkf)[2:end]

        qoi_gsbl_mat = reduce(vcat, qoi_gsbl)
        qoi_gsbl_mat[isnan.(qoi_gsbl_mat)] .= Inf
        quants_gsbl = quantile.(eachcol(qoi_gsbl_mat), ([0.5, 0.1, 0.9],))
        medians_gsbl = first.(quants_gsbl)[2:end]
        bars_gsbl = map(x -> (x[2], x[3]), quants_gsbl)[2:end]
        times = 2 * (1 .+ (1:length(bars_gsbl))) / (length(bars_gsbl) + 1)
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
false && begin
    fixed_key = :Ne
    x_axis = :sigma_x_filter
    y_axis = :mass_err
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
        fig_name = join(vcat(split(filename, "_")[1:end-1], string(y_axis), "convergence.pdf"), "_")
        save(joinpath(@__DIR__, "figs", fig_name), fig)
        fig
    end
end

# %%
with_theme(merge(theme_minimal(),theme_latexfonts()), Axis=(
    xlabel="Ensemble Size, log-scale", xscale = log10,
    xticks=[25, 38, 56, 84, 127]
)) do
    df = df_cut
    group_col = :algorithm
    x_col = :Ne
    y_col = :rmse2
    groups = unique(df[!,group_col])
    n_group = length(groups)
    abscessa_vals = Int.(sort(unique(df[!,x_col])))
    GROUP_INC = 0.15
    JITTER = 0.01
    alg_names = Dict(["locenkf" => "EnKF", "hlocenkf" => "GSBL-EnKF"])
    alg_markers = Dict(["locenkf" => :circle, "hlocenkf" => :+])
    fig = Figure(size=(1000,300))
    for state in 1:3
        ax = Axis(fig[1,state], ylabel= state == 1 ? "RMSE" : "")
        xlims!(ax, (20, 150))
        for (j,gdf) in enumerate(groupby(df, group_col))
            x_inc = 2*(j - ((n_group + 1)÷2))/n_group - 0.5
            group_inc = GROUP_INC * gdf[!, x_col] * x_inc
            jitter = JITTER * randn(length(gdf[!,x_col])) .* gdf[!, x_col]
            x_jit = gdf[!,x_col] + jitter + group_inc
            y = map(x->x[state], gdf[!,y_col])
            alg = first(gdf[!,group_col])
            scatter!(ax, x_jit, y, label=alg_names[alg], markersize=18, marker=alg_markers[alg], alpha=0.6)
        end
        for (j,gdf) in enumerate(groupby(df, group_col))
            x_inc = 2*(j - ((n_group + 1)÷2))/n_group - 0.5
            # group_inc = GROUP_INC * ab * x_inc
            y = map(x->x[state], gdf[!,y_col])
            med_error = [median(y[gdf[!,x_col] .== n]) for n in abscessa_vals]
            alg = first(gdf[!,group_col])
            scatterlines!(ax, abscessa_vals .* (1 .+ GROUP_INC * x_inc), med_error, linewidth=3, markersize=18, strokecolor=:black, strokewidth=3, marker=alg_markers[alg])
        end
        if state == 3
            axislegend()
        end
    end
    save(joinpath(@__DIR__, "figs", "rmse_sod.pdf"), fig)
    fig
end