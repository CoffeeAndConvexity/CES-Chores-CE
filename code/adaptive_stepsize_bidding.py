import numpy as np
import pandas as pd

from ces_disutility import compute_excess_supply

import argparse
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search for largest stepsize that doesn't cause blowup.")
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--num-instances", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--noise-level", type=float, default=1)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--M", type=int, default=200)
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

def run(D_instance, method: str, eta: float, max_steps: int, instance_seed: int) -> None: 
    args = _parse_args()
    rho = args.rho
    
    n_agents, n_resources = D_instance.shape
    B = np.ones(n_agents)

    eps_list = [0.1, 0.05, 0.01, 0.005, 0.001]
    p0 = np.full(n_resources, B.sum() / n_resources, dtype=float)
    p = p0.copy()

    hit_iterations = {eps: None for eps in eps_list}
    for t in range(max_steps):
        y = compute_excess_supply(p, D_instance, B, rho)
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

def search_largest_stepsize(D_instance, method="multiplicative", max_steps: int = 10000, instance_seed: int = 0) -> np.ndarray:
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
            ok, data = run(D_instance, method=method, eta=eta, max_steps=max_steps, instance_seed=instance_seed)
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

    ok, data = run(D_instance, method=method, eta=best_eta, max_steps=max_steps, instance_seed=instance_seed)
    print(data)
    return best_eta, data

def distance_matrix_among_papers(D): 

    M = D.shape[1]
    X = D.T 
    distance_matrix = np.zeros((M, M))
    for j in range(M): 
        for j_ in range(M): 
            distance_matrix[j][j_] = sum((X[j] - X[j_]) ** 2)

    return distance_matrix 

if __name__ == "__main__":

    print("Loading bidding data...")
    df = pd.read_csv('./code/bidding-data.csv')

    args = _parse_args()
    print(f"Arguments: rho={args.rho}, num_instances={args.num_instances}, max_steps={args.max_steps}, noise_level={args.noise_level}, N={args.N}, M={args.M}")

    dict_bidder_index = dict()
    i = 0
    for bidder in df['Bidder']:
        if bidder not in dict_bidder_index.keys():
            dict_bidder_index[bidder] = i
            i += 1

    N, M = len(np.unique(df['Bidder'])), max(df['Submission'])
    print(f"N = {N}, M = {M}")
    B = np.ones(shape=N)

    dict_pref_value = {
        'yes': 1,
        'maybe': 3,
        'no response': 5,
        'no': 7,
        'conflict': 7 * args.M + 1  # optimal price bound?
    }

    # create D matrix based on the bidding data, where D[i][j] is the value of reviewer i for paper j
    D = dict_pref_value['no response'] * np.ones(shape=(N, M))
    for row in list(df.itertuples(index=False, name=None)):
        D[dict_bidder_index[row[0]]][row[1] - 1] = dict_pref_value[row[2]]

    distance_matrix = distance_matrix_among_papers(D)

    num_instances = args.num_instances
    seeds = range(num_instances)  # set how many instances we want to try for one size
    
    N, M = args.N, args.M
    
    for instance_seed in seeds:
        np.random.seed(instance_seed)

        while True:  # loop utill we obtain a valid instance
            # randomly pick a paper 
            j = np.random.randint(D.shape[1])

            # find M-nearest papers
            M_nearest_neighbors = np.argsort(distance_matrix[j])[:M]

            # select reviewers with most responses 
            reviewer_num_response = np.ones(D.shape[0], dtype=int)
            for i in range(D.shape[0]):
                for j in M_nearest_neighbors: 
                    if D[i][j] < 5 - 1e-3: 
                        reviewer_num_response[i] += 1 
            N_top_response_reviewers = np.flip(np.argsort(reviewer_num_response))[:N] 

            # attain sampled D
            D_sampled = D[np.ix_(N_top_response_reviewers, M_nearest_neighbors)] 

            if max(np.amin(D_sampled, axis=1)) < 6: 
                break

        print(f"Testing instance {instance_seed} with shape {D_sampled.shape}")

        # add noise to D_sampled based on noise level argument
        noise_level = args.noise_level
        if noise_level > 0:
            np.random.seed(instance_seed)
            noise = noise_level * np.random.normal(size=(N, M)) 
            D_sampled = np.maximum(D_sampled + noise, 1)

        best_eta, data = search_largest_stepsize(D_sampled, method="multiplicative", max_steps=args.max_steps, instance_seed=instance_seed)

        # write to a common CSV file for later plotting (without overwriting previous rows for other methods)
        df = pd.DataFrame([{"method": "multiplicative", 
                            "n_agents": N, 
                            "n_resources": M, 
                            "rho": args.rho, 
                            "instance": instance_seed, 
                            "eta": best_eta, 
                            **data}])
        print(df)
        df.to_csv(f"batch_adaptive_stepsize_multiplicative_bidding.csv", index=False, mode='a', 
                    header=not pd.io.common.file_exists(f"batch_adaptive_stepsize_multiplicative_bidding.csv"))
        
        best_eta, data = search_largest_stepsize(D_sampled, method="additive", max_steps=args.max_steps, instance_seed=instance_seed)

        # write to a common CSV file for later plotting (without overwriting previous rows for other methods)
        df = pd.DataFrame([{"method": "additive", 
                            "n_agents": N, 
                            "n_resources": M, 
                            "rho": args.rho, 
                            "instance": instance_seed, 
                            "eta": best_eta, 
                            **data}])
        print(df)
        df.to_csv(f"batch_adaptive_stepsize_additive_bidding.csv", index=False, mode='a', 
                    header=not pd.io.common.file_exists(f"batch_adaptive_stepsize_additive_bidding.csv"))





