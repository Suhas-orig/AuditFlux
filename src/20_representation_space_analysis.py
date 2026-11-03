from __future__ import annotations

"""AuditFlux Program 20: Representation-space analysis on BFW.

Purpose
-------
Extract an independent 512-D face representation for each unique BFW image and
examine (i) within-identity compactness, (ii) demographic information
recoverable from identity-level representations, and (iii) how representation
similarity relates to held-out verification errors.

Representation model
--------------------
facenet-pytorch InceptionResnetV1 pretrained on CASIA-WebFace. This is an
additional exploratory model, NOT one of the VGG16/ResNet50/SENet50 models
whose scores were used in Programs 1-19. CASIA-WebFace is selected rather than
VGGFace2 because BFW images originate from VGGFace2; this reduces the obvious
training-source overlap concern, but does not prove that no identities overlap.

Input
-----
The existing BFW pair CSV plus the BFW cropped-face image directory or ZIP.
The script searches for bfw-faces-cropped.zip beside the configured BFW CSV,
or accepts --image-source / BFW_IMAGE_ROOT / BFW_CROPPED_ZIP.

Important
---------
This script does not reconstruct the original VGG16/ResNet50/SENet50 embeddings.
It extracts a separate FaceNet representation to add a mechanistic probe.
All demographic-probe results are descriptive representation-leakage measures,
not evidence of causal bias or unfairness by themselves.
"""

import argparse
import hashlib
import io
import json
import os
import sys
import time
import warnings
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
from config import BFW_DATA_PATH, OUTPUT_DIR  # noqa: E402

PREFIX = "20_representation_space"
GROUPS = ["AF", "AM", "BF", "BM", "IF", "IM", "WF", "WM"]
MODEL_NAME = "InceptionResnetV1-CASIA-WebFace"
EMBEDDING_DIM = 512
IMAGE_SIZE = 160
RANDOM_SEED = 20261120
BOOTSTRAP_REPS = 1000
PAIR_CHUNK_SIZE = 100_000

REQUIRED_COLUMNS = [
    "fold", "p1", "p2", "label", "id1", "id2", "a1", "a2",
    "g1", "g2", "e1", "e2", "vgg16", "resnet50", "senet50",
]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def resolve_image_source(explicit: str | None) -> Path:
    """Resolve an image directory or archive without silently choosing raw faces."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    for env_name in ("BFW_IMAGE_ROOT", "BFW_CROPPED_ZIP"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser())
    try:
        import config as project_config  # type: ignore
        for attr in ("BFW_IMAGE_ROOT", "BFW_CROPPED_ZIP"):
            value = getattr(project_config, attr, None)
            if value:
                candidates.append(Path(value))
    except Exception:
        pass

    data_parent = Path(BFW_DATA_PATH).expanduser().resolve().parent
    candidates.extend([
        data_parent / "bfw-faces-cropped.zip",
        data_parent / "faces-cropped.zip",
        data_parent / "bfw_faces_cropped.zip",
        data_parent / "bfw-faces-cropped",
        data_parent / "faces-cropped",
        data_parent / "cropped",
    ])

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file() and candidate.suffix.lower() == ".zip":
            return candidate.resolve()
        if candidate.is_dir():
            # Accept only a directory that visibly contains BFW subgroup folders.
            if any((candidate / group_name).exists() for group_name in [
                "asian_females", "asian_males", "black_females", "black_males",
                "indian_females", "indian_males", "white_females", "white_males",
            ]):
                return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate the BFW cropped-face images. Put bfw-faces-cropped.zip "
        "beside the BFW CSV, extract it into a directory containing the eight "
        "demographic folders, set BFW_CROPPED_ZIP/BFW_IMAGE_ROOT, or pass "
        "--image-source. Do not substitute the non-cropped archive without "
        "reviewing its preprocessing requirements."
    )


class ImageResolver:
    def __init__(self, source: Path):
        self.source = source
        self.zip_file: zipfile.ZipFile | None = None
        self.member_by_suffix: dict[str, list[str]] = {}
        self.member_by_norm: dict[str, str] = {}
        if source.is_file():
            self.zip_file = zipfile.ZipFile(source, "r")
            for member in self.zip_file.namelist():
                if member.endswith("/"):
                    continue
                norm = member.replace("\\", "/").lstrip("./").lower()
                self.member_by_norm[norm] = member
                parts = PurePosixPath(norm).parts
                if len(parts) >= 3:
                    suffix = "/".join(parts[-3:]).lower()
                    self.member_by_suffix.setdefault(suffix, []).append(member)
        else:
            self.root = source

    def read(self, relative_path: str) -> bytes:
        rel = relative_path.replace("\\", "/").lstrip("./")
        if self.zip_file is None:
            path = self.source.joinpath(*PurePosixPath(rel).parts)
            if not path.is_file():
                raise FileNotFoundError(f"Missing BFW image: {path}")
            return path.read_bytes()

        # First try an exact normalized archive member.
        normalized_lookup = rel.lower()
        exact = self.member_by_norm.get(normalized_lookup)
        if exact is not None:
            return self.zip_file.read(exact)
        suffix = "/".join(PurePosixPath(rel).parts[-3:]).lower()
        matches = self.member_by_suffix.get(suffix, [])
        if len(matches) == 1:
            return self.zip_file.read(matches[0])
        if len(matches) > 1:
            # Prefer an unambiguous path under a cropped-face folder.
            cropped = [m for m in matches if "crop" in m.lower()]
            if len(cropped) == 1:
                return self.zip_file.read(cropped[0])
            raise RuntimeError(
                f"Ambiguous archive path for {relative_path}: {matches[:5]}"
            )
        raise FileNotFoundError(f"Image path not found in archive: {relative_path}")

    def close(self) -> None:
        if self.zip_file is not None:
            self.zip_file.close()


def build_image_metadata(df: pd.DataFrame) -> pd.DataFrame:
    left = df[["p1", "id1", "fold", "a1", "g1", "e1"]].rename(
        columns={"p1": "image_path", "id1": "identity", "a1": "group", "g1": "gender", "e1": "ethnicity"}
    )
    right = df[["p2", "id2", "fold", "a2", "g2", "e2"]].rename(
        columns={"p2": "image_path", "id2": "identity", "a2": "group", "g2": "gender", "e2": "ethnicity"}
    )
    images = pd.concat([left, right], ignore_index=True)
    images["image_path"] = images["image_path"].astype(str).str.replace("\\", "/", regex=False)
    images = images.drop_duplicates()
    if images["image_path"].duplicated().any():
        bad = images.loc[images["image_path"].duplicated(keep=False)].head(10)
        raise ValueError(f"Image paths map to conflicting identity metadata:\n{bad}")
    images = images.sort_values("image_path").reset_index(drop=True)
    if images["image_path"].nunique() != len(images):
        raise ValueError("Image metadata contains duplicate paths.")
    if images["identity"].nunique() != 800:
        raise ValueError(f"Expected 800 identities; found {images['identity'].nunique()}.")
    counts = images.groupby("identity").size()
    if not (counts == 25).all():
        raise ValueError("Expected exactly 25 unique images per identity.")
    if set(images["group"].astype(str).unique()) != set(GROUPS):
        raise ValueError("Unexpected demographic group labels in image metadata.")
    identity_cells = images[["identity", "fold", "group"]].drop_duplicates().groupby(["fold", "group"]).size()
    if len(identity_cells) != 40 or not (identity_cells == 20).all():
        raise ValueError("Expected exactly 20 identities in each of the 5 fold x 8 group cells.")
    return images


def load_face_model(device: str):
    try:
        import torch
        from facenet_pytorch import InceptionResnetV1
    except ImportError as exc:
        raise ImportError(
            "Missing embedding dependencies. Install them in the project environment: "
            "pip install torch torchvision facenet-pytorch"
        ) from exc
    resolved_device = device
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA GPU.")
    # CASIA-WebFace is deliberately selected to avoid using a VGGFace2-trained
    # embedding model on BFW, whose source imagery is VGGFace2.
    model = InceptionResnetV1(pretrained="casia-webface", classify=False).eval().to(resolved_device)
    return torch, model, resolved_device


def image_to_tensor(image_bytes: bytes, torch):
    with Image.open(io.BytesIO(image_bytes)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        width, height = im.size
        if min(width, height) < 20:
            raise ValueError(f"Image is unexpectedly small: {im.size}")
        # Preserve the crop aspect ratio; pad to a square rather than stretch
        # faces differently in width and height.
        im = ImageOps.pad(
            im, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.BILINEAR,
            color=(127, 127, 127), centering=(0.5, 0.5)
        )
        arr = np.asarray(im, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    return tensor, width, height


def extract_embeddings(images: pd.DataFrame, resolver: ImageResolver, device: str, batch_size: int,
                       max_images: int | None = None) -> tuple[np.ndarray, pd.DataFrame, dict]:
    torch, model, resolved_device = load_face_model(device)
    work = images.copy()
    if max_images is not None:
        work = work.head(max_images).copy()
    vectors: list[np.ndarray] = []
    diagnostics: list[dict] = []
    batch_tensors: list = []
    batch_meta: list[dict] = []
    started = time.time()

    def flush_batch():
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors, dim=0).to(resolved_device)
        with torch.inference_mode():
            z = model(x)
            z = torch.nn.functional.normalize(z, p=2, dim=1)
        out = z.detach().cpu().numpy().astype(np.float32)
        vectors.extend(out)
        diagnostics.extend(batch_meta)
        batch_tensors.clear()
        batch_meta.clear()

    for idx, row in enumerate(work.itertuples(index=False), start=1):
        try:
            raw = resolver.read(row.image_path)
            tensor, width, height = image_to_tensor(raw, torch)
        except Exception as exc:
            raise RuntimeError(f"Failed image {row.image_path}: {exc}") from exc
        batch_tensors.append(tensor)
        batch_meta.append({
            "image_path": row.image_path,
            "source_width": width,
            "source_height": height,
            "source_aspect_ratio": width / height,
        })
        if len(batch_tensors) >= batch_size:
            flush_batch()
            if idx % (batch_size * 10) == 0:
                print(f"  Embedded {idx:,}/{len(work):,} images...")
    flush_batch()
    embeddings = np.vstack(vectors).astype(np.float32)
    diag = pd.DataFrame(diagnostics)
    diag = work.reset_index(drop=True).merge(diag, on="image_path", validate="one_to_one")
    meta = {
        "model": MODEL_NAME,
        "pretraining_dataset": "CASIA-WebFace",
        "embedding_dimension": int(embeddings.shape[1]),
        "image_size": IMAGE_SIZE,
        "device": resolved_device,
        "images_embedded": int(len(embeddings)),
        "elapsed_seconds": round(time.time() - started, 2),
        "embedding_norm_min": float(np.linalg.norm(embeddings, axis=1).min()),
        "embedding_norm_max": float(np.linalg.norm(embeddings, axis=1).max()),
        "embedding_norm_mean": float(np.linalg.norm(embeddings, axis=1).mean()),
    }
    return embeddings, diag, meta


def within_identity_compactness(images: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    rows = []
    for identity, part in images.groupby("identity", sort=False):
        idx = part.index.to_numpy()
        z = embeddings[idx]
        sim = z @ z.T
        upper = sim[np.triu_indices(len(z), k=1)]
        rows.append({
            "identity": identity,
            "fold": int(part["fold"].iloc[0]),
            "group": str(part["group"].iloc[0]),
            "gender": str(part["gender"].iloc[0]),
            "ethnicity": str(part["ethnicity"].iloc[0]),
            "n_images": int(len(part)),
            "n_genuine_image_pairs": int(len(upper)),
            "within_identity_cosine_mean": float(np.mean(upper)),
            "within_identity_cosine_median": float(np.median(upper)),
            "within_identity_cosine_sd": float(np.std(upper, ddof=1)),
            "within_identity_cosine_q10": float(np.quantile(upper, .10)),
            "within_identity_cosine_q90": float(np.quantile(upper, .90)),
            "within_identity_cosine_min": float(np.min(upper)),
        })
    return pd.DataFrame(rows).sort_values(["group", "fold", "identity"]).reset_index(drop=True)


def stratified_identity_bootstrap_mean(values: np.ndarray, folds: np.ndarray,
                                       reps: int, rng: np.random.Generator) -> tuple[float, float]:
    fold_values = {f: values[folds == f] for f in np.unique(folds)}
    draws = np.empty(reps, dtype=float)
    for b in range(reps):
        sample_parts = []
        for arr in fold_values.values():
            sample_parts.append(rng.choice(arr, size=len(arr), replace=True))
        draws[b] = np.mean(np.concatenate(sample_parts))
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def summarize_compactness(identity_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for group, part in identity_df.groupby("group", sort=True):
        values = part["within_identity_cosine_mean"].to_numpy(float)
        folds = part["fold"].to_numpy()
        lo, hi = stratified_identity_bootstrap_mean(values, folds, BOOTSTRAP_REPS, rng)
        rows.append({
            "group": group,
            "n_identities": int(len(part)),
            "mean_identity_compactness": float(values.mean()),
            "sd_between_identity_compactness": float(values.std(ddof=1)),
            "median_identity_compactness": float(np.median(values)),
            "p2_5_stratified_identity_bootstrap": lo,
            "p97_5_stratified_identity_bootstrap": hi,
        })
    return pd.DataFrame(rows)


def demographic_probe(identity_df: pd.DataFrame, embeddings: np.ndarray):
    # One normalized mean embedding per identity prevents 25 images/person from
    # masquerading as 25 independent demographic observations.
    id_vectors = []
    id_rows = []
    for identity, part in identity_df.groupby("identity", sort=False):
        idx = part.index.to_numpy()
        mean_vec = embeddings[idx].mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        id_vectors.append(mean_vec.astype(np.float32))
        id_rows.append({
            "identity": identity,
            "fold": int(part["fold"].iloc[0]),
            "group": str(part["group"].iloc[0]),
            "ethnicity": str(part["ethnicity"].iloc[0]),
            "gender": str(part["gender"].iloc[0]),
        })
    X = np.vstack(id_vectors)
    meta = pd.DataFrame(id_rows)
    tasks = {"intersectional_group_8way": "group", "ethnicity_4way": "ethnicity", "gender_2way": "gender"}
    prediction_rows = []
    metric_rows = []
    for task, target_col in tasks.items():
        y = meta[target_col].astype(str).to_numpy()
        oof = np.empty(len(y), dtype=object)
        for fold in sorted(meta["fold"].unique()):
            train = meta["fold"].to_numpy() != fold
            test = ~train
            scaler = StandardScaler()
            x_train = scaler.fit_transform(X[train])
            x_test = scaler.transform(X[test])
            clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", solver="lbfgs")
            clf.fit(x_train, y[train])
            pred = clf.predict(x_test)
            oof[test] = pred
            metric_rows.append({
                "task": task,
                "held_out_fold": int(fold),
                "n_train_identities": int(train.sum()),
                "n_test_identities": int(test.sum()),
                "accuracy": float(accuracy_score(y[test], pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
                "macro_f1": float(f1_score(y[test], pred, average="macro", zero_division=0)),
                "chance_accuracy": 1.0 / len(np.unique(y)),
            })
        for i, row in meta.iterrows():
            prediction_rows.append({
                "task": task,
                "identity": row["identity"],
                "fold": row["fold"],
                "true_label": y[i],
                "predicted_label": oof[i],
                "correct": int(y[i] == oof[i]),
            })
        metric_rows.append({
            "task": task,
            "held_out_fold": "POOLED_OOF",
            "n_train_identities": np.nan,
            "n_test_identities": int(len(y)),
            "accuracy": float(accuracy_score(y, oof)),
            "balanced_accuracy": float(balanced_accuracy_score(y, oof)),
            "macro_f1": float(f1_score(y, oof, average="macro", zero_division=0)),
            "chance_accuracy": 1.0 / len(np.unique(y)),
        })
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows), meta, X


def pair_scores_and_metrics(df: pd.DataFrame, images: pd.DataFrame, embeddings: np.ndarray):
    path_to_idx = pd.Series(images.index.to_numpy(), index=images["image_path"]).to_dict()
    mapped1 = df["p1"].map(path_to_idx)
    mapped2 = df["p2"].map(path_to_idx)
    if mapped1.isna().any() or mapped2.isna().any():
        raise ValueError("Some pair endpoints do not have extracted embeddings.")
    i1 = mapped1.to_numpy(dtype=np.int32)
    i2 = mapped2.to_numpy(dtype=np.int32)
    scores = np.empty(len(df), dtype=np.float32)
    for start in range(0, len(df), PAIR_CHUNK_SIZE):
        end = min(start + PAIR_CHUNK_SIZE, len(df))
        scores[start:end] = np.einsum("ij,ij->i", embeddings[i1[start:end]], embeddings[i2[start:end]])

    scored = df[["fold", "label", "id1", "id2", "a1", "a2", "vgg16", "resnet50", "senet50"]].copy()
    scored.insert(0, "row_index", np.arange(len(scored), dtype=np.int64))
    scored["facenet_casia_cosine"] = scores

    distribution_rows = []
    for (group, label), part in scored.groupby(["a1", "label"], sort=True):
        vals = part["facenet_casia_cosine"].to_numpy(float)
        distribution_rows.append({
            "group": group,
            "label": int(label),
            "n_pairs": int(len(vals)),
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)),
            "q10": float(np.quantile(vals, .10)),
            "q25": float(np.quantile(vals, .25)),
            "median": float(np.median(vals)),
            "q75": float(np.quantile(vals, .75)),
            "q90": float(np.quantile(vals, .90)),
        })
    return scored, pd.DataFrame(distribution_rows), scores


def group_auc_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, part in scored.groupby("a1", sort=True):
        auc = roc_auc_score(part["label"], part["facenet_casia_cosine"])
        rows.append({
            "group": group,
            "n_pairs": int(len(part)),
            "genuine_pairs": int((part["label"] == 1).sum()),
            "impostor_pairs": int((part["label"] == 0).sum()),
            "roc_auc": float(auc),
        })
    rows.append({
        "group": "ALL",
        "n_pairs": int(len(scored)),
        "genuine_pairs": int((scored["label"] == 1).sum()),
        "impostor_pairs": int((scored["label"] == 0).sum()),
        "roc_auc": float(roc_auc_score(scored["label"], scored["facenet_casia_cosine"])),
    })
    return pd.DataFrame(rows)


def cross_validated_verification(df: pd.DataFrame, scores: np.ndarray):
    rows = []
    threshold_rows = []
    labels = df["label"].to_numpy(dtype=np.int8)
    folds = df["fold"].to_numpy()
    groups = df["a1"].astype(str).to_numpy()
    for heldout in sorted(np.unique(folds)):
        train = folds != heldout
        test = ~train
        train_impostor = scores[train & (labels == 0)]
        # Higher quantile threshold; calibration is done only on other folds.
        threshold = float(np.quantile(train_impostor, .99, method="higher"))
        threshold_rows.append({
            "held_out_fold": int(heldout),
            "target_training_fpr": 0.01,
            "calibration_rows": int(train.sum()),
            "calibration_impostor_pairs": int(len(train_impostor)),
            "threshold": threshold,
            "observed_training_fpr": float(np.mean(train_impostor >= threshold)),
        })
        pred = scores[test] >= threshold
        y = labels[test]
        g = groups[test]
        for group in GROUPS:
            mask = g == group
            positive = mask & (y == 1)
            negative = mask & (y == 0)
            fnr = float(np.mean(~pred[positive])) if positive.any() else np.nan
            fpr = float(np.mean(pred[negative])) if negative.any() else np.nan
            rows.append({
                "held_out_fold": int(heldout),
                "group": group,
                "threshold_calibrated_on_other_folds": threshold,
                "n_genuine": int(positive.sum()),
                "n_impostor": int(negative.sum()),
                "FNR": fnr,
                "FPR": fpr,
            })
    return pd.DataFrame(threshold_rows), pd.DataFrame(rows)


def compactness_error_association(df: pd.DataFrame, scores: np.ndarray,
                                  identity_compactness: pd.DataFrame,
                                  thresholds: pd.DataFrame) -> pd.DataFrame:
    threshold_map = thresholds.set_index("held_out_fold")["threshold"].to_dict()
    genuine = df[df["label"].to_numpy() == 1].copy()
    genuine_scores = scores[df["label"].to_numpy() == 1]
    genuine["score"] = genuine_scores
    genuine["threshold"] = genuine["fold"].map(threshold_map)
    genuine["false_reject"] = (genuine["score"] < genuine["threshold"]).astype(np.int8)
    per_identity = genuine.groupby(["fold", "id1", "a1"], as_index=False).agg(
        identity_FNR=("false_reject", "mean"),
        genuine_pairs=("false_reject", "size"),
    ).rename(columns={"id1": "identity", "a1": "group"})
    merged = per_identity.merge(
        identity_compactness[["identity", "within_identity_cosine_mean"]],
        on="identity", validate="one_to_one"
    )
    rows = []
    for name, part in [("ALL", merged), *list(merged.groupby("group"))]:
        if len(part) >= 10 and part["within_identity_cosine_mean"].nunique() > 1 and part["identity_FNR"].nunique() > 1:
            rho = spearmanr(part["within_identity_cosine_mean"], part["identity_FNR"]).statistic
        else:
            rho = np.nan
        rows.append({
            "group": name,
            "n_identities": int(len(part)),
            "spearman_compactness_vs_identity_FNR": float(rho) if np.isfinite(rho) else np.nan,
        })
    return pd.DataFrame(rows)


def score_correlations(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subset_name, part in [("all_pairs", scored), ("genuine_only", scored[scored.label == 1]),
                              ("impostor_only", scored[scored.label == 0])]:
        for model in ["vgg16", "resnet50", "senet50"]:
            if part["facenet_casia_cosine"].nunique() > 1 and part[model].nunique() > 1:
                rho = spearmanr(part["facenet_casia_cosine"], part[model]).statistic
            else:
                rho = np.nan
            rows.append({
                "subset": subset_name,
                "existing_model_score": model,
                "n_pairs": int(len(part)),
                "spearman_correlation": float(rho) if np.isfinite(rho) else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="AuditFlux Program 20 representation-space analysis")
    parser.add_argument("--image-source", default=None, help="BFW cropped-image directory or ZIP archive")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=None, help="Optional small smoke-test limit; do not use for final results")
    parser.add_argument("--force-embeddings", action="store_true", help="Recompute the embedding cache")
    args = parser.parse_args()

    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = Path(BFW_DATA_PATH)
    if not data_path.is_file():
        raise FileNotFoundError(f"BFW CSV not found: {data_path}")
    print(f"Loading BFW table: {data_path}")
    df = pd.read_csv(data_path, usecols=REQUIRED_COLUMNS)
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Missing values found in required BFW columns.")
    if len(df) != 923_898:
        warnings.warn(f"Expected 923,898 rows from the established BFW table; found {len(df):,}.")
    if set(df["label"].unique()) != {0, 1}:
        raise ValueError("Expected binary BFW labels 0 and 1.")
    if not (df.loc[df.label == 1, "id1"].to_numpy() == df.loc[df.label == 1, "id2"].to_numpy()).all():
        raise ValueError("Genuine rows must connect the same identity at both endpoints.")
    if not (df.loc[df.label == 0, "id1"].to_numpy() != df.loc[df.label == 0, "id2"].to_numpy()).all():
        raise ValueError("Impostor rows must connect different identities.")

    image_meta = build_image_metadata(df)
    image_source = resolve_image_source(args.image_source)
    print(f"Image source: {image_source}")
    print(f"Unique images: {len(image_meta):,}; identities: {image_meta.identity.nunique():,}")

    cache_path = OUTPUT_DIR / f"{PREFIX}_facenet_casia_embeddings.npz"
    image_meta_path = OUTPUT_DIR / f"{PREFIX}_image_metadata.csv"
    embedding_meta = {}
    if cache_path.exists() and image_meta_path.exists() and not args.force_embeddings and args.max_images is None:
        print(f"Loading cached embeddings: {cache_path}")
        cached = np.load(cache_path, allow_pickle=False)
        embeddings = cached["embeddings"].astype(np.float32)
        cached_meta = pd.read_csv(image_meta_path)
        if cached_meta["image_path"].tolist() != image_meta["image_path"].tolist():
            raise ValueError("Embedding cache image order does not match current BFW metadata. Use --force-embeddings.")
        image_meta = cached_meta
        embedding_meta = json.loads(str(cached["metadata"].item()))
        if embeddings.shape != (len(image_meta), EMBEDDING_DIM):
            raise ValueError("Cached embedding matrix has unexpected shape.")
    else:
        resolver = ImageResolver(image_source)
        try:
            embeddings, image_meta, embedding_meta = extract_embeddings(
                image_meta, resolver, args.device, args.batch_size, args.max_images
            )
        finally:
            resolver.close()
        if args.max_images is None:
            np.savez_compressed(cache_path, embeddings=embeddings, metadata=np.array(json.dumps(embedding_meta)))
            image_meta.to_csv(image_meta_path, index=False)
            print(f"Saved embedding cache: {cache_path}")
        else:
            print("SMOKE TEST MODE: embeddings were not saved as a final cache.")

    if args.max_images is not None:
        print(
            f"SMOKE TEST PASSED: extracted {len(embeddings)} images with shape {embeddings.shape}. "
            "No final analysis outputs were written. Run again without --max-images for the full study."
        )
        return

    if embeddings.shape != (20_000, EMBEDDING_DIM):
        raise ValueError(f"Expected 20,000 x 512 embeddings; found {embeddings.shape}.")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(embeddings).all() or not np.allclose(norms, 1.0, atol=2e-3):
        raise ValueError("Embeddings contain non-finite values or are not L2-normalized.")

    print("Computing within-identity compactness...")
    identity_compactness = within_identity_compactness(image_meta, embeddings)
    compactness_summary = summarize_compactness(identity_compactness)

    print("Running identity-disjoint demographic probes...")
    probe_metrics, probe_predictions, identity_meta, identity_vectors = demographic_probe(image_meta, embeddings)

    print("Scoring all BFW comparison pairs...")
    scored, score_distributions, pair_scores = pair_scores_and_metrics(df, image_meta, embeddings)
    group_auc = group_auc_summary(scored)
    threshold_df, heldout_audit = cross_validated_verification(df, pair_scores)
    association = compactness_error_association(df, pair_scores, identity_compactness, threshold_df)
    correlations = score_correlations(scored)

    outputs = {
        "image_metadata": f"{PREFIX}_image_metadata.csv",
        "identity_compactness": f"{PREFIX}_identity_compactness.csv",
        "group_compactness_summary": f"{PREFIX}_group_compactness_summary.csv",
        "demographic_probe_metrics": f"{PREFIX}_demographic_probe_metrics.csv",
        "demographic_probe_oof_predictions": f"{PREFIX}_demographic_probe_oof_predictions.csv",
        "pair_score_distributions": f"{PREFIX}_pair_score_distributions.csv",
        "group_auc": f"{PREFIX}_group_auc.csv",
        "heldout_verification_thresholds": f"{PREFIX}_heldout_verification_thresholds.csv",
        "heldout_verification_audit": f"{PREFIX}_heldout_verification_audit.csv",
        "compactness_error_association": f"{PREFIX}_compactness_error_association.csv",
        "score_correlations": f"{PREFIX}_score_correlations.csv",
    }
    identity_compactness.to_csv(OUTPUT_DIR / outputs["identity_compactness"], index=False)
    compactness_summary.to_csv(OUTPUT_DIR / outputs["group_compactness_summary"], index=False)
    probe_metrics.to_csv(OUTPUT_DIR / outputs["demographic_probe_metrics"], index=False)
    probe_predictions.to_csv(OUTPUT_DIR / outputs["demographic_probe_oof_predictions"], index=False)
    score_distributions.to_csv(OUTPUT_DIR / outputs["pair_score_distributions"], index=False)
    group_auc.to_csv(OUTPUT_DIR / outputs["group_auc"], index=False)
    threshold_df.to_csv(OUTPUT_DIR / outputs["heldout_verification_thresholds"], index=False)
    heldout_audit.to_csv(OUTPUT_DIR / outputs["heldout_verification_audit"], index=False)
    association.to_csv(OUTPUT_DIR / outputs["compactness_error_association"], index=False)
    correlations.to_csv(OUTPUT_DIR / outputs["score_correlations"], index=False)

    manifest = {
        "experiment": "Program 20 - Representation-Space Analysis",
        "source_bfw_csv": str(data_path),
        "source_bfw_rows": int(len(df)),
        "source_bfw_sha256": sha256_file(data_path),
        "image_source": str(image_source),
        "unique_images": int(len(image_meta)),
        "unique_identities": int(image_meta.identity.nunique()),
        "images_per_identity_expected": 25,
        "embedding_model": MODEL_NAME,
        "embedding_pretraining_dataset": "CASIA-WebFace",
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_image_size": IMAGE_SIZE,
        "embedding_normalization": "L2-normalized 512-dimensional FaceNet embeddings",
        "preprocessing": "RGB conversion, EXIF orientation, aspect-ratio-preserving pad to 160x160, tensor normalization to [-1,1]; no additional MTCNN alignment",
        "identity_compactness": "All 300 unordered image pairs per identity; identity is the resampling unit for group bootstrap summaries",
        "demographic_probe": "One normalized mean embedding per identity; five-fold leave-one-original-BFW-fold-out logistic regression; no identity appears in both train and test",
        "verification_thresholding": "For each held-out BFW fold, global impostor 99th-percentile threshold calibrated on the other four folds; descriptive group FNR/FPR on held-out fold",
        "audited_group_convention": "a1 first endpoint, consistent with AuditFlux Programs 1-19",
        "limitations": [
            "This is a separate FaceNet-CASIA representation probe, not the original VGG16, ResNet50, or SENet50 embedding vectors.",
            "BFW source imagery is VGGFace2; CASIA-WebFace reduces direct source-dataset overlap compared with a VGGFace2-pretrained probe, but identity overlap cannot be ruled out without a name-level overlap audit.",
            "No MTCNN alignment is applied; supplied BFW cropped images are resized directly to 160x160. Crop geometry and preprocessing can affect embeddings.",
            "Demographic predictability measures information recoverable from the representation; it is not by itself evidence of unfairness, causality, or harmful decisions.",
            "The five-fold held-out verification audit is descriptive and conditional on the observed BFW pair protocol; pairwise observations share identities.",
            "The BFW impostor comparison graph is incomplete; no missing comparisons are imputed.",
            "Group bootstrap percentiles are identity-resampled descriptive intervals, not universal population guarantees.",
            "Results concern one additional pretrained representation model and do not generalize automatically to all face-recognition architectures."
        ],
        "embedding_metadata": embedding_meta,
        "random_seed": RANDOM_SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "outputs": outputs,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    with (OUTPUT_DIR / f"{PREFIX}_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nPROGRAM 20 COMPLETE")
    print(f"Images embedded: {len(image_meta):,}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Outputs: {OUTPUT_DIR}")
    print(f"Elapsed: {time.time() - start:.1f}s")
    print("Review manifest limitations before interpreting the results.")


if __name__ == "__main__":
    main()
