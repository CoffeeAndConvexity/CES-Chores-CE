# Data

- `bidding-data.csv`: anonymized bidder IDs, submission IDs, and categorical preferences used by the bidding runner.
- `distribute_tasks_valuation_matrices/`: Spliddit valuation matrices; `_index.csv` records their instance IDs, filenames, and dimensions.
- `raw/spliddit/`: the original Spliddit SQL export and extracted CSV tables.
- `results/`: saved experiment results used to reproduce the reference figures, including the cached Spliddit ratio analysis.

Saved data is kept separate from new runs. Experiment scripts write new results to `outputs/results/` by default, and plotting scripts read the reference results here unless `--data-dir` is supplied.

The historical Spliddit result CSVs identify instances by `instance_idx`, which depended on the original directory listing order. The saved ratio table also records matrix filenames. New Spliddit runs use the matrix index order and record filenames explicitly; do not assume historical and new numeric instance indices refer to the same matrices.
