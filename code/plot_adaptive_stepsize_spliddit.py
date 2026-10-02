from matplotlib import pyplot as plt
import pandas as pd
from sqlalchemy import values
from wandb import summary
import numpy as np


# import the following colors by RGB values
COLOR_MAP = {
    "additive": "#b3480e",
    "multiplicative": "#0e5195",
}

SHAPE_MAP = {
    "additive": "s",
    "multiplicative": "^",
}

FONTSIZE = 24
MARKER_SIZE = 12

def main() -> None:
    plt.figure(figsize=(20, 6))
    # create 3 subplots in a 1x3 grid for each rho value
    rho_values = [1.2, 2.0, 5.0]
    for idx, rho in enumerate(rho_values):
        plt.subplot(1, 3, idx + 1)
        IF_LABELED = {"additive": False, "multiplicative": False}
        for method in ["additive", "multiplicative", ]:
            csv_path = f"data/batch_adaptive_stepsize_{method}_spliddit.csv"
            df = pd.read_csv(csv_path)
            df_rho = df.loc[df["rho"] == rho].copy()
            # df_rho -> dict of {column_name: value} for the row with the largest eta
            if df_rho.empty:
                print(f"No data for rho={rho} in {csv_path}")
                continue
            
            # average iterations for each epsilon value (group by epsilon and take the mean of iterations) over 20 instances                
            # compute the error bars as the standard deviation of iterations for each epsilon value
            # for each instance, there is a columne called 0.1, 0.05, 0.01, 0.005, 0.001 which is the number of iterations to reach that epsilon value (or max_steps + 1 if not reached)
            # there is no column named "epsilon", so we need to iterate over the columns that are epsilon values
            inv_eps_list = []
            iter_mean = []
            iter_std = []
            for epsilon in [0.1, 0.05, 0.01, 0.005, 0.001]:
                # if an entry is None, fill it with max_steps = 60000
                iterations = df_rho[str(epsilon)].fillna(60000)
                mean, std = iterations.mean(), iterations.std()
                print(f"rho={rho}, method={method}, epsilon={epsilon}, mean={mean}, std={std}")
                inv_eps_list.append(1 / epsilon)
                iter_mean.append(mean)
                iter_std.append(std)
            if not IF_LABELED[method]:
                plt.errorbar(inv_eps_list, iter_mean, yerr=0.1*np.array(iter_std), label=f"{method}", 
                                marker=SHAPE_MAP[method], capsize=3, markersize=MARKER_SIZE, linewidth=3,
                                color=COLOR_MAP[method])
                plt.scatter(inv_eps_list, iter_mean, marker=SHAPE_MAP[method], color=COLOR_MAP[method])
                IF_LABELED[method] = True
            else:
                plt.errorbar(inv_eps_list, iter_mean, yerr=0.1*np.array(iter_std), 
                                marker=SHAPE_MAP[method], capsize=3, markersize=MARKER_SIZE, linewidth=3,
                                color=COLOR_MAP[method])
                plt.scatter(inv_eps_list, iter_mean, marker=SHAPE_MAP[method], color=COLOR_MAP[method])

        plt.xscale("log")
        plt.xlabel(rf"$1 / \epsilon$", fontsize=FONTSIZE)
        plt.ylabel("Average iterations", fontsize=FONTSIZE)
        # share y labels across subplots
        if idx > 0:
            plt.ylabel("")
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE, rotation=90)
        # xticks and yticks number of ticks should be at most 5
        plt.locator_params(axis='y', nbins=5)
        plt.title(rf"$\rho={rho}$", fontsize=FONTSIZE)  # Latex style for rho
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=FONTSIZE)
    # plt.suptitle(f"Additive vs Multiplicative\nSimulations by {value_distribution}", fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(f"spliddit.pdf", 
                facecolor="white",
                transparent=False,
                format="pdf",
                bbox_inches="tight",
                pad_inches=0.03,
                dpi=170)

if __name__ == "__main__":
    main()
