"""
Convert QM9 molecules to 2D molecular fingerprints and Coulomb matrices.

Example
-------
python -m src.features.featurize --raw data/raw --out data/processed
"""
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np, pandas as pd, argparse, pathlib, tqdm, joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def morgan_fp(mol, n_bits=2048, radius=2):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

def rdkit_desc(mol):
   # return np.array([f(mol) for f in Descriptors.descList])  # len ~200
    return np.array([func(mol) for name, func in Descriptors.descList])


def coulomb_matrix(coords, charges, max_atoms=29):
    """Very simple Coulomb matrix (no sorting)."""
    n = len(charges)
    mat = np.zeros((max_atoms, max_atoms))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                mat[i, i] = 0.5 * charges[i] ** 2.4
            else:
                dist = np.linalg.norm(coords[i] - coords[j])
                mat[i, j] = mat[j, i] = charges[i] * charges[j] / dist
    return mat.flatten()

def main(raw, out):
    raw = pathlib.Path(raw)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # Load targets
    targets = pd.read_csv(raw / "gdb9.sdf.csv")
    sdf_path = raw / "gdb9.sdf"

    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    fps, descs, cms, valid_idx = [], [], [], []

    for idx, mol in enumerate(tqdm.tqdm(suppl, desc="Featurizing")):
        if mol is None:
            continue
        fps.append(np.asarray(morgan_fp(mol)))
        descs.append(rdkit_desc(mol))
        conf = mol.GetConformer()
        coords = np.array(conf.GetPositions())
        charges = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        cms.append(coulomb_matrix(coords, charges))
        valid_idx.append(idx)

    # Stack and save
    X_fps = np.vstack(fps)
    X_desc = np.vstack(descs)
    X_cm = np.vstack(cms)

    scaler = StandardScaler().fit(X_desc)
    X_desc = scaler.transform(X_desc)

    # Optional PCA to 100 dims for CM
    pca = PCA(n_components=100).fit(X_cm)
    X_cm = pca.transform(X_cm)

    np.save(out / "X_fps.npy", X_fps)
    np.save(out / "X_desc.npy", X_desc)
    np.save(out / "X_cm.npy", X_cm)

    targets.iloc[valid_idx].to_csv(out / "targets.csv", index=False)

    joblib.dump({"scaler": scaler, "pca": pca}, out / "preprocessors.pkl")
    print("Saved processed features to", out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    main(**vars(parser.parse_args()))
