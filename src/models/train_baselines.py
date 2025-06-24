"""
Train a suite of classical regressors on QM9 fingerprints/descriptors.

Usage
-----
python -m src.models.train_baselines --config config.yaml
"""
import yaml, pathlib, joblib, argparse, json
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODELS = {
    "ridge": lambda cfg: Ridge(alpha=cfg["alpha"]),
    "random_forest": lambda cfg: RandomForestRegressor(**cfg),
    "xgboost": lambda cfg: XGBRegressor(
        objective="reg:squarederror", tree_method="hist", **cfg
    ),
}

def metrics(y_true, y_pred):
    return dict(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )

def main(config):
    cfg = yaml.safe_load(open(config))
    proc_dir = pathlib.Path(cfg["dataset"]["processed_dir"])
    y = pd.read_csv(proc_dir / "targets.csv")[cfg["dataset"]["target_property"]].values

    # Use RDKit descriptors + Coulomb matrix concatenated
    X = np.hstack(
        [
            np.load(proc_dir / "X_desc.npy"),
            np.load(proc_dir / "X_cm.npy"),
        ]
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=cfg["training"]["test_size"] + cfg["training"]["val_size"], random_state=cfg["training"]["seed"]
    )
    val_ratio = cfg["training"]["val_size"] / (
        cfg["training"]["test_size"] + cfg["training"]["val_size"]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=cfg["training"]["seed"]
    )

    results = {}
    for model_name, model_cfg in cfg["models"]["baselines"].items():
        print(f"\nTraining {model_name} …")
        model = MODELS[model_name](model_cfg)
        model.fit(X_train, y_train)
        preds = {
            "train": model.predict(X_train),
            "val": model.predict(X_val),
            "test": model.predict(X_test),
        }
        results[model_name] = {split: metrics(y_true, preds[split]) for split, y_true in
                               zip(["train", "val", "test"], [y_train, y_val, y_test])}
        joblib.dump(model, proc_dir / f"{model_name}.joblib")

    summary_path = proc_dir / "baseline_metrics.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print("Metrics saved to", summary_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(**vars(parser.parse_args()))
