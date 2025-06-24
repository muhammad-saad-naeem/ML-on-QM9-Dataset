import matplotlib.pyplot as plt
import numpy as np
import pathlib
from sklearn.metrics import mean_absolute_error, mean_squared_error

proc_dir = pathlib.Path("data/processed")
models = {
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "gnn": "SchNet (GNN)"
}

for key, label in models.items():
    try:
        y_true = np.load(proc_dir / f"{key}_y_true.npy").flatten()
        y_pred = np.load(proc_dir / f"{key}_y_pred.npy").flatten()
    except FileNotFoundError:
        print(f"Skipping {label} — predictions not found.")
        continue

    if y_true.shape != y_pred.shape:
        print(f"Shape mismatch in {label}: {y_true.shape} vs {y_pred.shape}")
        continue

    # Compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", linewidths=0.4, s=40)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "--", color="black", lw=2)
    plt.xlabel("True Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title(f"Parity Plot: {label}", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Add MAE and RMSE in a styled text box
    metrics_text = f"MAE  = {mae:.3f}\nRMSE = {rmse:.3f}"
    props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.95)
    plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                   fontsize=11, verticalalignment="top", bbox=props)

    # Save
    outpath = proc_dir / f"parity_{key}.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved: {outpath}")
