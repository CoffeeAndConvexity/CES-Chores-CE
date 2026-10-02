from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ces_disutility import compute_excess_supply

import argparse
# rho, max_steps
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search for largest stepsize that doesn't cause blowup.")
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--first-instances", type=int, default=20, help="Number of instances to run (starting from the first instance).")
    return parser.parse_args()

def _compute_update(method: str, prices: np.ndarray, excess_supply: np.ndarray) -> np.ndarray:
    if method == "additive":
        return excess_supply - excess_supply.mean()
    if method == "multiplicative":
        u = prices * excess_supply
        return u - u.mean()
    if method == "quadratic":
        u = (prices ** 2) * excess_supply
        return u - u.mean()
    raise ValueError(f"Unknown method: {method}")

def run(instance, method: str, eta: float, max_steps: int) -> None: 
    args = _parse_args()
    D, B = instance
    n_agents, n_resources = D.shape
    rho = args.rho
    
    eps_list = [0.1, 0.05, 0.01, 0.005, 0.001]

    p0 = np.full(n_resources, B.sum() / n_resources, dtype=float)
    p = p0.copy()

    hit_iterations = {eps: None for eps in eps_list}
    for t in range(max_steps):
        y = compute_excess_supply(p, D, B, rho)
        update = _compute_update(method, p, y)
        p = p + eta * update

        # decrease eta if any price goes negative or if any price becomes non-finite (indicating blowup)
        if np.any(p < 0):
            return False, "negative prices"
        if not np.all(np.isfinite(p)):
            return False, "blowup - nonfinite prices"
        
        norm_y = np.linalg.norm(y)

        # check if we hit any epsilons, and record the iteration of the first time we hit each epsilon
        for eps in eps_list:
            if norm_y <= eps and hit_iterations[eps] is None:
                hit_iterations[eps] = t

        # increase eta if we have hit all epsilons and haven't had any issues, to speed up convergence
        if all(hit_iterations[eps] is not None for eps in eps_list):
            data = {**hit_iterations}
            break
        if t == max_steps - 1:
            data = {**hit_iterations}

    return True, data

def search_largest_stepsize(instance, method="multiplicative", max_steps: int = 10000) -> np.ndarray:
    # procedure: 
    # search for 10 different eta values with equally spaced values between 1e-6 and eta_upper_bound (starting at 10)
    # update eta_upper_bound to be the smallest eta that causes negative prices or blowup
    # update best_eta to be the eta that has the best convergence (e.g. smallest average iterations to hit the smallest epsilon)

    eta = 1
    eta_upper_bound = 10
    best_eta = None
    second_best_eta = None
    third_best_eta = None
    while True:
        eta_list = []
        eta_ok_list = []
        eta_convergence_list = []
        for i in range(9):
            eta = eta_upper_bound * (i + 1) / 10
            ok, data = run(instance, method=method, eta=eta, max_steps=max_steps)
            eta_list.append(eta)
            eta_ok_list.append(ok)
            if ok:
                eta_convergence_list.append(data[0.001] if data[0.001] is not None else max_steps + 1)
            else:
                eta_convergence_list.append(2 * max_steps + 1)
        best_eta = eta_list[np.argsort(eta_convergence_list)[0]]
        second_best_eta = eta_list[np.argsort(eta_convergence_list)[1]]
        third_best_eta = eta_list[np.argsort(eta_convergence_list)[2]]
        if any(not ok for ok in eta_ok_list):
            eta_upper_bound = min(eta for eta, ok in zip(eta_list, eta_ok_list) if not ok)
        else:
            break

    ok, data = run(instance=instance, method=method, eta=best_eta, max_steps=max_steps)
    print(data)
    return best_eta, data

# test search_largest_stepsize
if __name__ == "__main__":
    # load *all* Spliddit instances from "Spliddit_Data/distribute_tasks_valuation_matrices"
    current_dir = Path(__file__).resolve().parent
    
    path = f"{current_dir}/../Spliddit_Data/distribute_tasks_valuation_matrices"
    import os
    instance_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    instances = []
    for instance_file in instance_files:
        if instance_file[0] == "_":
            continue
        print(f"Loading instance file: {instance_file}")
        df = pd.read_csv(os.path.join(path, instance_file), index_col=0)
        D = df.values
        B = np.ones(D.shape[0])
        instances.append((D, B))

    for instance in instances[-5:]:
        print(f"Testing instance file: {instance}")

    args = _parse_args()
    rho = args.rho
    max_steps = args.max_steps
    if args.first_instances == -1:
        first_instances = len(instances)
    else:
        first_instances = args.first_instances
    for method in ["multiplicative", "additive"]:
        for instance_idx, instance in enumerate(instances[:first_instances]):
            # run the search for largest stepsize on this instance
            best_eta, data = search_largest_stepsize(instance=instance, method=method, max_steps=max_steps)

            # write to a common CSV file for later plotting (without overwriting previous rows for other methods)
            df = pd.DataFrame([{"method": method, 
                                "instance_idx": instance_idx,
                                "rho": rho, 
                                "eta": best_eta, 
                                **data}])
            
            df.to_csv(f"batch_adaptive_stepsize_{method}_spliddit.csv", index=False, mode='a', 
                      header=not pd.io.common.file_exists(f"batch_adaptive_stepsize_{method}_spliddit.csv"))