# CES-Chores-CE

Research code for competitive equilibrium experiments in chores markets with convex CES disutilities. The experiments compare additive and multiplicative price updates and search for stepsizes using synthetic, bidding, and Spliddit data.

## Setup

Use Python 3.10 or newer. Run commands from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The plotting scripts currently import SQLAlchemy and wandb, so these are included alongside the numerical and plotting dependencies.

## Contents

- `code/ces_disutility.py`: CES disutility, demand, and excess-supply calculations.
- `code/adaptive_stepsize*.py`: synthetic, bidding, and Spliddit experiment runners.
- `code/batchrun_*.sh`: batch experiment commands.
- `code/plot_*.py`: plots of recorded experiment results.
- `code/bidding-data.csv`: input for the bidding experiments.
- `data/`: recorded results and Spliddit valuation matrices.
- `Spliddit_Data/`: Spliddit SQL and CSV source data.
- `figures/` and root-level PNG/PDF files: saved figures.
- `old files/`: earlier experiment scripts and results.

## Usage

Run the small numerical example:

```sh
python code/ces_disutility.py
```

Run a small synthetic experiment:

```sh
python code/adaptive_stepsize.py \
  --n-agents 3 --n-resources 4 --rho 2 \
  --num-instances 1 --max-steps 5 \
  --value-distribution truncated_normal
```

Regenerate the bidding figure from the recorded results in `data/`:

```sh
python code/plot_adaptive_stepsize_bidding.py
```

This writes `bidding.pdf` in the repository root. The other adaptive-stepsize plotting scripts also read the saved CSVs in `data/`.

Experiment runners append CSV results in the current working directory. Copy or move the intended results into `data/` before plotting new runs; some plotting scripts expect consolidated filenames rather than the parameter-specific filenames emitted by the runners.

The Spliddit runner expects matrices at `Spliddit_Data/distribute_tasks_valuation_matrices`, while this snapshot stores them at `data/distribute_tasks_valuation_matrices`. Before running it on macOS or Linux, create the expected link:

```sh
ln -s ../data/distribute_tasks_valuation_matrices \
  Spliddit_Data/distribute_tasks_valuation_matrices
python code/adaptive_stepsize_spliddit.py --rho 2 --first-instances 1
```

For `code/plot_spliddit_ratio_relations.py`, the same location can instead be supplied with `--matrix-dir data/distribute_tasks_valuation_matrices`.

This repository is a snapshot of the research scripts, input data, recorded results, and figures. Dependencies are not version-pinned.
