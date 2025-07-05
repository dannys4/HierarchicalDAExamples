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
using DataFrames, JLD2, Distributions, ProgressMeter, CairoMakie, HierarchicalDA

# %%
if length(ARGS) < 2
    ArgumentError("Call script with `julia data_parsing [path to data] [serialization filename]`")
end

data_path, filename = ARGS[1:2]

if !ispath(data_path) || !isabspath(data_path)
    ArgumentError("Expected valid absolute path argument, got $data_path")
end

# %%
function add_delta_y!(row::Vector{<:Pair{Symbol}}, file::JLD2.JLDFile)
    data = file["data"]
    push!(row, :delta_y => size(data["xt"], 1) ÷ size(data["yt"], 1))
end

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

function create_rows(file::JLD2.JLDFile,
    ::Type{AcceptableType}=Union{<:Real,<:AbstractString};
    keep_params=DEFAULT_KEEP_PARAMS,
    metrics_group="metrics",
    preproc!::Function=Returns(nothing)) where {AcceptableType}

    row = Pair{Symbol,AcceptableType}[]
    preproc!(row, file)
    for group_str in keep_params
        group = file[group_str]
        for key in keys(group)
            val = group[key]
            if val isa AcceptableType
                push!(row, Symbol(key) => group[key])
            else
                append!(row, flatten_properties(AcceptableType, group[key], string(key)))
            end
        end
    end
    rows = NamedTuple[]
    metrics = file[metrics_group]
    for algorithm in keys(metrics)
        row_alg = deepcopy(row)
        push!(row_alg, :algorithm => algorithm)
        for metric in keys(metrics[algorithm])
            push!(row_alg, Symbol(metric) => metrics[algorithm][metric])
        end
        push!(rows, NamedTuple(row_alg))
    end
    rows
end

# %%
rows = create_rows(ex_load; (preproc!)=add_delta_y!)

# %%
df = DataFrame()
@showprogress "Collating files..." for filename in readdir(data_path)
    local rows
    rows_curry = file -> create_rows(file; (preproc!)=add_delta_y!)
    try
        rows = jldopen(rows_curry, joinpath(data_path, filename), "r")
    catch e
        if e isa JLD2.InvalidDataException
            @warn "Problematic file $filename, ignoring..."
        else
            rethrow(e)
        end
    else
        append!(df, rows)
    end
end

@save joinpath(@__DIR__, "data", filename) df