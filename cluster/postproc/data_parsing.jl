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
Pkg.activate(joinpath(@__DIR__, ".."));

# %%
using DataFrames, JLD2, Distributions, ProgressMeter, HierarchicalDA, UUIDs, Random

# %%
if length(ARGS) < 2
    ArgumentError("Call script with `julia data_parsing [path to data] [serialization filename]`")
end

data_path, filename = ARGS[1:2]

if !ispath(data_path) || !isabspath(data_path)
    ArgumentError("Expected valid absolute path argument, got $data_path")
end

file_path = joinpath(@__DIR__, "data", strip(filename))
file_prepath = joinpath(split(file_path, "/")[1:end-1]...)
if !ispath(file_prepath)
    ArgumentError("Expected valid filename path, got $filename")
end

@info "Loading from $data_path, saving to $file_path"
ff = jldopen(file_path, "w")
ff["df"] = DataFrame()
close(ff)

# %%
function flatten_properties(::Type{AcceptableType},
    obj,
    obj_name::String,
    properties::Vector=[]) where {AcceptableType}

    all_names = propertynames(obj)
    if length(all_names) < 1
        (obj isa AcceptableType) || @error "Could not flatten object $obj_name"
        push!(properties, Symbol(obj_name) => obj)
    else
        for name in all_names
            obj_prop = getproperty(obj, name)
            prop_name = obj_name * "_" * string(name)
            flatten_properties(AcceptableType, obj_prop, prop_name, properties)
        end
    end
    return properties
end

DEFAULT_KEEP_PARAMS = ["data_parameters", "filter_parameters", "GSBL_parameters"]

function process_file(file::JLD2.JLDFile,
    ::Type{AcceptableType}=Union{<:Real,<:AbstractString,<:AbstractArray{<:Real}};
    keep_params=DEFAULT_KEEP_PARAMS,
    metrics_group="metrics",
    preproc!::Function=Returns(nothing)) where {AcceptableType}

    uuid = string(uuid4(Random.GLOBAL_RNG))
    PairT = Pair{Symbol,AcceptableType}
    params_row = PairT[:id=>uuid]
    preproc!(params_row, file)
    for group_str in keep_params
        group = file[group_str]
        for key in keys(group)
            val = group[key]
            if val isa AcceptableType
                push!(params_row, Symbol(key) => group[key])
            else
                append!(params_row, flatten_properties(AcceptableType, group[key], string(key)))
            end
        end
    end
    truth_row = PairT[:id=>uuid]
    alg_rows = Vector{PairT}[]
    metrics = file[metrics_group]

    for metric_key in keys(metrics)
        metric_val = metrics[metric_key]
        if metric_val isa JLD2.Group
            row_alg = Pair{Symbol,AcceptableType}[:id=>uuid, :algorithm=>metric_key]
            for metric in keys(metric_val)
                metric_name = split(metric, "_")[1] # Some things are (metric)_(alg_name)
                @info "" metric_name metric_val[metric]
                if isa(metric_val[metric], AcceptableType)
                    push!(row_alg, Symbol(metric_name) => metric_val[metric])
                end
            end
            push!(alg_rows, row_alg)
        else
            push!(truth_row, Symbol(metric_key) => metric_val)
        end
    end
    NamedTuple(truth_row), NamedTuple.(alg_rows), NamedTuple(params_row)
end

# %%
df_truth, df_algs, df_params = DataFrame(), DataFrame(), DataFrame()
@showprogress "Collating files..." for filename in readdir(data_path)
    local truth_row, alg_rows, params_row
    try
        jldopen(joinpath(data_path, filename), "r") do file
            truth_row, alg_rows, params_row = process_file(file)
        end
    catch e
        if e isa JLD2.InvalidDataException || e isa EOFError
            @warn "Problematic file $filename, ignoring..."
        else
            rethrow(e)
        end
    else
        push!(df_truth, truth_row)
        try
            append!(df_algs, alg_rows)
        catch e
            @info "" propertynames.(alg_rows)
            rethrow(e)
        end
        push!(df_params, params_row)
    end
end

@save file_path df_truth df_algs df_params
