# HierarchicalDAExamples
This code accompanies the publication

> *Regularity-preserving data assimilation for hyperbolic conservation laws using generalized sparse Bayesian learning*, J. Glaubitz, D. Sharp, M. Le Provost, and Y. Marzouk. 2025.

## Installation
If you are in the home directory, i.e., the directory containing the `README.md` file, open a Julia REPL and invoke `using Pkg; Pkg.activate("."); Pkg.instantiate();`. This should install everything as needed.

## Usage
### Notebooks
- First, make sure to install [`jupyterlab`](https://jupyterlab.readthedocs.io/en/latest/getting_started/installation.html) and [`jupytext`](https://jupytext.readthedocs.io/en/latest/install.html). We recommend `conda` for this.
- Once you've installed these, make sure the `jupytext` [extension](https://jupytext.readthedocs.io/en/latest/jupyterlab-extension.html) is installed in `jupyterlab`. To check if it's installed, click the `File` menu in the top left of a `jupyterlab` window, and `Jupytext` should be one of the options.
- Once you boot up `jupyterlab`, open the `notebooks` directory in the file explorer and navigate to the subdirectory with the example that you are interested in.
- There will be a `.jl` file here (for example, a path might be `HierarchicalDAExamples/notebooks/burgers/burgers_example.jl`); right click on this and press `Open with > notebook`. This will open the script as a notebook using `jupytext`.
- Use this as you would a normal `jupyter` notebook.

> <span style="color:red;">WARNING: Any changes to an opened notebook will be recorded in the associated script file.</span>

### Cluster submission
For parameter ablation and numerical experiments, we provide a different directory, `cluster`, which is more amenable, though fundamentally the same. One can open any `.jl` files in `HierarchicalDAExamples/cluster/examples` with notebooks as well for editing etc. See the above instructions for this.

For running these examples for experiments, use the `cluster/run_example` helper script, which will automate the process of running an example many times over many different parameters. Change into the `cluster` directory and invoke `./run_example -h` for instructions.

See `cluster/ex_submit.sh` for an example submission script for SLURM systems. Since `run_example` does most of the work, the submission script should be rather minimal (and thus it is easier to adapt to other submission systems).

By default, these numerical examples will record troves of data for each simulation, with many metrics pre-calculated. To pare down the amount of content and post-process, we provide helper scripts in the `cluster/postproc/` subdirectory. To pare the data down and nust store the metrics and parameters in a Julia `DataFrame`, call `julia data_parsing.jl`. To process this `DataFrame` and make plots (e.g., for convergence), use the code provided in `plotting.jl`---due to the highly customizable nature of plotting, this is better suited for running as a notebook.

## Original examples for using HierarchicalDA library
Examples based on code originally provided by Mathieu Le Provost. See branch `mleprovo_version` for the original examples.