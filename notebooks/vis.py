import matplotlib.pyplot as plt
import os, json, argparse, ast
import numpy as np

def plot_per_digit_accuracy(per_digit_acc, title="Per-digit accuracy", ax=None, cmap="viridis"):
    keys = [ast.literal_eval(k) if isinstance(k, str) else k for k in per_digit_acc.keys()]
    max_i = max(k[0] for k in keys)
    max_j = max(k[1] for k in keys)

    grid = np.full((max_i, max_j), np.nan)
    for (i, j), v in zip(keys, per_digit_acc.values()):
        grid[i - 1, j - 1] = v

    if ax is None:
        fig, ax = plt.subplots(figsize=(max_j * 0.6 + 2, max_i * 0.6 + 2))

    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, origin="upper")

    for i in range(max_i):
        for j in range(max_j):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        color="white" if grid[i, j] < 0.5 else "black", fontsize=8)

    ax.set_xticks(range(max_j), [str(j + 1) for j in range(max_j)])
    ax.set_yticks(range(max_i), [str(i + 1) for i in range(max_i)])
    ax.set_xlabel("operand 2 digits")
    ax.set_ylabel("operand 1 digits")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if ax is None:
        return fig, ax
    else:
        return ax
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="../artifacts/runs/self_improvement_multiplication/")
    args = parser.parse_args()
    
    folder_names = sorted(os.listdir(args.output_dir))
    rounds = [f for f in folder_names if f.startswith("round_")]
    num_rounds = len(rounds)
    
    # per digit accuracy over rounds
    fig, ax = plt.subplots(1, num_rounds, figsize=(num_rounds * 8, 6), dpi=200)
    ax.flatten()
    eval_accu_list = []
    for i, round_folder in enumerate(rounds):
        with open(os.path.join(args.output_dir, round_folder, "metrics.json"), "r") as f:
            round_i_metadata = json.load(f)
        eval_accu_list.append(round_i_metadata["eval_accuracy"])
        plot_per_digit_accuracy(round_i_metadata["per_digit_accuracy"], title=f"Round {i+1} per-digit accuracy", ax=ax[i])
        fig.savefig("per_digit_accuracy_over_rounds.png", dpi=200, bbox_inches="tight")
    
    # plot avg eval accuracy over rounds
    fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
    ax.plot(range(1, num_rounds + 1), eval_accu_list, marker="o")
    ax.set_xlabel("Round")
    ax.set_xticks(range(1, num_rounds + 1))
    ax.set_ylabel("Avg Evaluation Accuracy")
    fig.savefig("eval_accuracy_over_rounds.png", dpi=200, bbox_inches="tight")
    