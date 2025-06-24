import numpy as np
import pandas as pd
import pathlib
import joblib
import yaml
from sklearn.model_selection import train_test_split

cfg = yaml.safe_load(open("config.yaml"))
proc_dir = pathlib.Path(cfg["dataset"]["processed_dir"])

# Load target
y = pd.read_csv(proc_dir / "targets.csv")[cfg["dataset"]["target_property"]].values

# Load features (same combo as training script)
X = np.hstack([
    np.load(proc_dir / "X_desc.npy"),
    np.load(proc_dir / "X_cm.npy"),
])

# Recreate splits
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=cfg["training"]["val_size"] + cfg["training"]["test_size"],
    random_state=cfg["training"]["seed"]
)
val_ratio = cfg["training"]["val_size"] / (cfg["training"]["val_size"] + cfg["training"]["test_size"])
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=val_ratio,
    random_state=cfg["training"]["seed"]
)

# For each model, reload and predict on test set
for model_name in ["ridge", "random_forest", "xgboost"]:
    model_path = proc_dir / f"{model_name}.joblib"
    if not model_path.exists():
        print(f"Skipping {model_name} — model not found.")
        continue
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    np.save(proc_dir / f"{model_name}_y_true.npy", y_test)
    np.save(proc_dir / f"{model_name}_y_pred.npy", y_pred)
    print(f"Saved predictions for {model_name}")
