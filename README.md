# CES-Chores-CE

Research code for competitive equilibrium experiments in chores markets with convex CES disutilities. The experiments compare additive and multiplicative price updates and search for stepsizes using synthetic, bidding, and Spliddit data.

## Setup

Use Python 3.10 or newer:

```sh
git clone https://github.com/CoffeeAndConvexity/CES-Chores-CE.git
cd CES-Chores-CE
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencies are NumPy, pandas, Matplotlib, and Pillow. The numerical example can be run with:

```sh
python code/ces_disutility.py
```

## Repository layout

```text
code/         Numerical routines, experiment runners, and plotting scripts
scripts/      Batch experiment commands
data/         Input datasets and saved reference results
figures/      Reference figures
archive/      Historical scripts, experiment results, and earlier figures
tests/        Numerical and workflow regression tests
outputs/      New results and figures (generated locally; ignored by Git)
```

See [data/README.md](data/README.md) for dataset details and [archive/README.md](archive/README.md) for historical files.

## Reproduce the reference plots

```sh
python code/plot_adaptive_stepsize.py
python code/plot_adaptive_stepsize_bidding.py
python code/plot_adaptive_stepsize_spliddit.py
python code/plot_spliddit_ratio_relations.py
```

Plots read the saved results in `data/results/` and write to `outputs/figures/`. Use `--help` to see plot options. The ratio plot uses the saved ratio table by default, preserving the historical instance mapping.

## Run experiments

Start with a small synthetic experiment:

```sh
python code/adaptive_stepsize.py \
  --n-agents 3 --n-resources 4 --rho 2 \
  --num-instances 1 --max-steps 5 \
  --value-distribution truncated_normal
```

Run a small bidding or Spliddit experiment:

```sh
python code/adaptive_stepsize_bidding.py \
  --N 3 --M 4 --rho 2 --num-instances 1 --max-steps 5

python code/adaptive_stepsize_spliddit.py \
  --rho 2 --first-instances 1 --max-steps 5
```

Runners append CSV results to `outputs/results/`. Use `--output-dir` to keep experiment configurations in separate directories. Defaults for inputs and outputs are resolved relative to the repository, so scripts also work when invoked by absolute path from another directory.

To plot new bidding results:

```sh
python code/plot_adaptive_stepsize_bidding.py \
  --data-dir outputs/results --output-dir outputs/figures/new-run \
  --missing-iterations 5
```

Set `--missing-iterations` to the experiment's `--max-steps` when using a different iteration budget. The default reference budgets are 10,000 for bidding and 60,000 for Spliddit.

The batch scripts run larger experiment grids and can take substantially longer:

```sh
bash scripts/batchrun_adaptive_stepsize.sh
bash scripts/batchrun_adaptive_stepsize_bidding.sh
bash scripts/batchrun_adaptive_stepsize_spliddit.sh
```

Additional command-line arguments are forwarded to the runner. Set `PYTHON_BIN` to select a Python executable when needed.

New Spliddit results record matrix filenames as well as numeric instance indices. Historical numeric indices depended on the original directory order; use filenames or the saved ratio table when relating results to matrices.

## Validation

```sh
python -m unittest discover -s tests -v
```

Tests cover numerical invariants and the refactored experiment and plotting workflows. Dependencies are not version-pinned; validation was performed with Python 3.12, NumPy 1.26.4, pandas 2.2.2, Matplotlib 3.8.4, and Pillow 10.3.0.

## Citation

If you use this code, cite the repository:

```bibtex
@misc{coffeeandconvexity2026ceschores,
  author       = {{CoffeeAndConvexity}},
  title        = {{CES-Chores-CE}: Competitive Equilibrium Experiments with Convex {CES} Disutilities},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/CoffeeAndConvexity/CES-Chores-CE}
}
```
