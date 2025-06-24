"""
Aggregate baseline & GNN metrics, print markdown table.

python -m src.models.evaluate --config config.yaml
"""
import yaml, pathlib, json, argparse, pandas as pd, numpy as np

def main(config):
    cfg = yaml.safe_load(open(config))
    proc_dir = pathlib.Path(cfg["dataset"]["processed_dir"])

    metrics = {}
    for name in ["baseline_metrics.json", "gnn_metrics.json"]:
        path = proc_dir / name
        if path.exists():
            metrics[name] = json.loads(path.read_text())

    # Build markdown
    rows = []
    if "baseline_metrics.json" in metrics:
        for model_name, vals in metrics["baseline_metrics.json"].items():
            rows.append({
                "model": model_name,
                "MAE_test": vals["test"]["mae"],
                "RMSE_test": vals["test"]["rmse"],
            })
    if "gnn_metrics.json" in metrics:
        rows.append({
            "model": "SchNet (GNN)",
            "MAE_test": metrics["gnn_metrics.json"]["test"]["mae"],
            "RMSE_test": metrics["gnn_metrics.json"]["test"]["rmse"],
        })

    df = pd.DataFrame(rows).sort_values("MAE_test")
    md = "### Test set performance\n\n" + df.to_markdown(index=False)
    report_path = pathlib.Path("reports/report.md")
    report_path.write_text(md)
    print(md)
    print(f"\nWrote {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(**vars(parser.parse_args()))
