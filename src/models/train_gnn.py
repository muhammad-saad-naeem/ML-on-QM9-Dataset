"""
Simple SchNet‑like GNN using PyTorch Geometric.

Usage
-----
python -m src.models.train_gnn --config config.yaml
"""
import yaml, argparse, pathlib, json, math, random, os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, DataLoader, Data
from torch_geometric.nn import SchNet
from torch_geometric.transforms import Distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

class QM9Dataset(InMemoryDataset):
    def __init__(self, root, target, transform=None, pre_transform=None):
        self.target = target
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["gdb9.sdf"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        # Assuming raw data already present (handled by download_qm9.py)
        pass

    def process(self):
        import rdkit.Chem as Chem

        sdf_path = pathlib.Path(self.raw_dir) / "gdb9.sdf"
        target_csv = pathlib.Path(self.raw_dir).parent / "targets.csv"
        targets = pd.read_csv(target_csv)[self.target].values

        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)

        data_list = []
        transform = Distance(norm=False)

        for idx, mol in enumerate(suppl):
            if mol is None:
                continue
            conf = mol.GetConformer()
            pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
            z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
            y = torch.tensor([targets[idx]], dtype=torch.float)

            edge_index = torch.combinations(torch.arange(z.size(0)), r=2).t()
            data = Data(z=z, pos=pos, y=y, edge_index=edge_index)
            data = transform(data)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.z, batch.pos, batch.batch)
        loss = torch.nn.functional.l1_loss(pred.squeeze(), batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.z, batch.pos, batch.batch)
            y_true.append(batch.y.cpu())
            y_pred.append(pred.cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return dict(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(math.sqrt(mean_squared_error(y_true, y_pred))),
    )

def main(config):
    cfg = yaml.safe_load(open(config))
    raw_dir = pathlib.Path(cfg["dataset"]["raw_dir"])
    processed_dir = pathlib.Path(cfg["dataset"]["processed_dir"])
    target = cfg["dataset"]["target_property"]
    dataset = QM9Dataset(root=str(raw_dir), target=target)

    # split
    idx = list(range(len(dataset)))
    train_idx, tmp_idx = train_test_split(idx, test_size=cfg["training"]["test_size"] + cfg["training"]["val_size"],
                                          random_state=cfg["training"]["seed"])
    val_ratio = cfg["training"]["val_size"] / (cfg["training"]["val_size"] + cfg["training"]["test_size"])
    val_idx, test_idx = train_test_split(tmp_idx, test_size=val_ratio,
                                         random_state=cfg["training"]["seed"])

    train_ds = dataset[train_idx]
    val_ds = dataset[val_idx]
    test_ds = dataset[test_idx]

    train_loader = DataLoader(train_ds, batch_size=cfg["models"]["gnn"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)
    test_loader = DataLoader(test_ds, batch_size=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SchNet(hidden_channels=cfg["models"]["gnn"]["hidden_dim"],
                   num_filters=cfg["models"]["gnn"]["hidden_dim"],
                   num_interactions=cfg["models"]["gnn"]["num_layers"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["models"]["gnn"]["lr"])

    best_val = float("inf")
    for epoch in range(cfg["models"]["gnn"]["epochs"]):
        train_loss = train(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["mae"] < best_val:
            best_val = val_metrics["mae"]
            torch.save(model.state_dict(), processed_dir / "best_schnet.pt")
        if epoch % 10 == 0:
            print(f"[{epoch:03d}] train_loss={train_loss:.4f}  val_mae={val_metrics['mae']:.4f}")

    # test
    model.load_state_dict(torch.load(processed_dir / "best_schnet.pt"))
    test_metrics = evaluate(model, test_loader, device)

    json_path = processed_dir / "gnn_metrics.json"
    json_path.write_text(json.dumps(dict(val=val_metrics, test=test_metrics), indent=2))
    print("Test metrics", test_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(**vars(parser.parse_args()))
