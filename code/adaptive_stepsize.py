import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ces_disutility import compute_excess_supply

import argparse
# n_agents, n_resources, rho
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search for largest stepsize that doesn't cause blowup.")
    parser.add_argument("--n-agents", type=int, default=100)
    parser.add_argument("--n-resources", type=int, default=200)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--num-instances", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--value-distribution", type=str, default="lognormal")
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

def run(method: str, eta: float, max_steps: int, instance_seed: int) -> None: 
    args = _parse_args()
    n_agents = args.n_agents
    n_resources = args.n_resources
    rho = args.rho
    
    eps_list = [0.1, 0.05, 0.01, 0.005, 0.001]

    np.random.seed(instance_seed)
    if args.value_distribution == "lognormal":
        D = np.random.lognormal(mean=0.0, sigma=1.0, size=(n_agents, n_resources))  # location parameter = 0.0, scale parameter=1.0
    elif args.value_distribution == "uniform":
        D = np.random.uniform(low=0.0, high=1.0, size=(n_agents, n_resources))
    elif args.value_distribution == "integer_uniform":
        D = np.random.randint(low=1, high=20, size=(n_agents, n_resources))
    elif args.value_distribution == "exponential":
        D = np.random.exponential(scale=1.0, size=(n_agents, n_resources))
    elif args.value_distribution == "truncated_normal":
        D = np.random.normal(loc=0.5, scale=0.2, size=(n_agents, n_resources))  # mean=0.5, std=0.2, truncated to be positive
        D = np.clip(D, a_min=0.01, a_max=None)  # truncate to be positive
    else:
        raise ValueError(f"Unknown value distribution: {args.value_distribution}")
    B = np.ones(n_agents)

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
            data = {"instance": instance_seed, **hit_iterations}
            break
        if t == max_steps - 1:
            data = {"instance": instance_seed, **hit_iterations}

    return True, data

def search_largest_stepsize(method="multiplicative", max_steps: int = 10000, instance_seed: int = 0) -> np.ndarray:
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
            ok, data = run(method=method, eta=eta, max_steps=max_steps, instance_seed=instance_seed)
            eta_list.append(eta)
            eta_ok_list.append(ok)
            if ok:
                eta_convergence_list.append(data[0.001] if data[0.001] is not None else max_steps + 1)
            else:
                eta_convergence_list.append(2 * max_steps + 1)
        print(eta_list, eta_convergence_list)
        best_eta = eta_list[np.argsort(eta_convergence_list)[0]]
        second_best_eta = eta_list[np.argsort(eta_convergence_list)[1]]
        third_best_eta = eta_list[np.argsort(eta_convergence_list)[2]]
        if any(not ok for ok in eta_ok_list):
            eta_upper_bound = min(eta for eta, ok in zip(eta_list, eta_ok_list) if not ok)
        else:
            break

    ok, data = run(method=method, eta=best_eta, max_steps=max_steps, instance_seed=instance_seed)
    print(data)
    return best_eta, data

# test search_largest_stepsize
if __name__ == "__main__":
    args = _parse_args()
    n_agents = args.n_agents
    n_resources = args.n_resources
    rho = args.rho
    max_steps = args.max_steps
    for method in ["multiplicative", "additive"]:
        for instance_seed in range(args.num_instances):
            best_eta, data = search_largest_stepsize(method=method, max_steps=max_steps, instance_seed=instance_seed)

            # write to a common CSV file for later plotting (without overwriting previous rows for other methods)
            df = pd.DataFrame([{"method": method, 
                                "n_agents": n_agents, 
                                "n_resources": n_resources, 
                                "rho": rho, 
                                "instance": instance_seed, 
                                "eta": best_eta, 
                                **data}])
            print(df)
            df.to_csv(f"batch_adaptive_stepsize_{method}_{n_agents}_{n_resources}_{args.value_distribution}.csv", index=False, mode='a', 
                      header=not pd.io.common.file_exists(f"batch_adaptive_stepsize_{method}_{n_agents}_{n_resources}_{args.value_distribution}.csv"))

        
                    
