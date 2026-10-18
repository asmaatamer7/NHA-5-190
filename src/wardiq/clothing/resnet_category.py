"""ResNet-50 garment category classifier: frozen-backbone baseline and bounded fine-tuning.

Rules enforced here (from the M2 task pages):
- fit on `train` only; choose epochs/checkpoint on `validation` only; `test` is never read unless
  `--final-test` is passed explicitly after the checkpoint is frozen
- labels come from classification_manifest.csv (resolved by source id in
build_classification_manifest.py)
- every run writes seed, config, class map, item-keyed predictions, metrics, confusion matrix
and cost
Nothing here fabricates results: metrics are computed from the predictions this code produces.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import zipfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms  # type: ignore[import-untyped]

IMNET_MEAN, IMNET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def eval_tf(size: Any) -> Any:
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMNET_MEAN, IMNET_STD),
        ]
    )


def train_tf(size: Any) -> Any:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMNET_MEAN, IMNET_STD),
        ]
    )


def _resolve(ref: str, image_root: str) -> Path:
    """
    Manifest refs may already start with 'data/raw/...' (repo-relative) or be relative to
    image_root.
    """
    for cand in (Path(image_root) / ref, Path(ref)):
        if cand.exists():
            return cand
    return Path(image_root) / ref  # let the caller raise a clear FileNotFoundError


_MNIST: dict[str, Any] = {}


def _mnist_image(ref: str, image_root: str) -> Image.Image:
    """
    'fashion_mnist://<source_split>/<source_index>' -> image from the official IDX release
    (downloaded once).
    """
    from torchvision.datasets import FashionMNIST  # type: ignore[import-untyped]

    split, idx = ref[len("fashion_mnist://") :].split("/")
    if split not in _MNIST:
        _MNIST[split] = FashionMNIST(
            str(Path(image_root) / "data" / "raw" / "fashion_mnist"),
            train=(split == "train"),
            download=True,
        )
    return cast(Image.Image, _MNIST[split][int(idx)][0].convert("RGB"))


def open_image(ref: str, image_root: str) -> Image.Image:
    """`ref` is a path, 'archive.zip::member' (Fashionpedia) or 'fashion_mnist://split/index'."""
    if ref.startswith("fashion_mnist://"):
        return _mnist_image(ref, image_root)
    if "::" in ref:
        zpath, member = ref.split("::", 1)
        with zipfile.ZipFile(_resolve(zpath, image_root)) as z, z.open(member) as f:
            return Image.open(f).convert("RGB")
    return Image.open(_resolve(ref, image_root)).convert("RGB")


class ClothingDataset(Dataset[Any]):
    def __init__(
        self, df: pd.DataFrame, class_to_idx: dict[str, Any], image_root: str, tf: Any
    ) -> None:
        self.df, self.c2i, self.root, self.tf = (
            df.reset_index(drop=True),
            class_to_idx,
            image_root,
            tf,
        )

    def __len__(self) -> Any:
        return len(self.df)

    def __getitem__(self, i: Any) -> Any:
        r = self.df.iloc[i]
        return self.tf(open_image(r.image_ref, self.root)), self.c2i[r.label], r.item_id


def check_image_coverage(df: pd.DataFrame, image_root: str) -> pd.DataFrame:
    """Metadata counts do not prove training readiness: test that every image actually opens."""
    ok = []
    for r in df.itertuples():
        try:
            open_image(cast(str, r.image_ref), image_root).verify() if "::" not in cast(
                str, r.image_ref
            ) else open_image(cast(str, r.image_ref), image_root)
            ok.append(True)
        except Exception:
            ok.append(False)
    return df.assign(image_ok=ok)


def build_model(num_classes: int, unfreeze: list[str], pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.resnet50(weights=weights)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, num_classes)  # new head is always trainable
    for name in unfreeze:  # e.g. ["layer4"] for bounded fine-tuning
        for p in getattr(m, name).parameters():
            p.requires_grad = True
    return cast(nn.Module, m)


@torch.no_grad()
def predict(model: Any, loader: Any, device: Any) -> Any:
    model.eval()
    ids, ys, ps, confs = [], [], [], []
    for x, y, item in loader:
        pr = torch.softmax(model(x.to(device)), 1).cpu()
        c, p = pr.max(1)
        ids += list(item)
        ys += y.tolist()
        ps += p.tolist()
        confs += c.tolist()
    return ids, np.array(ys), np.array(ps), np.array(confs)


def metrics(y: Any, p: Any, classes: Any) -> Any:
    pr, rc, f1, sup = precision_recall_fscore_support(
        y, p, labels=range(len(classes)), zero_division=0
    )
    return {
        "n_items": int(len(y)),
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(
            f1_score(y, p, average="macro", labels=range(len(classes)), zero_division=0)
        ),
        "per_class": {
            c: {
                "support": int(sup[i]),
                "precision": float(pr[i]),
                "recall": float(rc[i]),
                "f1": float(f1[i]),
            }
            for i, c in enumerate(classes)
        },
    }


def save_predictions(
    path: Any, ids: Any, y: Any, p: Any, conf: Any, classes: Any, extra: Any = None
) -> None:
    df = pd.DataFrame(
        {
            "item_id": ids,
            "true": [classes[i] for i in y],
            "pred": [classes[i] for i in p],
            "confidence": conf,
        }
    )
    if extra is not None:
        df = df.merge(extra, on="item_id", how="left")
    df.to_csv(path, index=False)


def run(cfg: dict[str, Any], final_test: bool = False) -> Any:
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(cfg["classification_manifest"], dtype=str, keep_default_na=False)
    if cfg.get("check_image_coverage", True):
        df = check_image_coverage(df, cfg["image_root"])
        cov = df.groupby(["split", "image_ok"]).size().unstack(fill_value=0)
        cov.to_csv(out / "image_coverage.csv")
        print(cov)
        df = df[df.image_ok].drop(columns="image_ok")
    if df.image_ref.str.startswith(
        "fashion_mnist://"
    ).any():  # download once in the main process, before DataLoader workers start
        for sp in ("train", "test"):
            _mnist_image(f"fashion_mnist://{sp}/0", cfg["image_root"])
    if cfg.get(
        "max_items_per_split"
    ):  # quick CPU check only: seeded random subset per split, never a deliverable
        n = int(cfg["max_items_per_split"])
        df = pd.concat(
            [g.sample(min(len(g), n), random_state=cfg["seed"]) for _, g in df.groupby("split")]
        )
    classes = sorted(df[df.split == "train"].label.unique())
    c2i = {c: i for i, c in enumerate(classes)}
    df = df[df.label.isin(classes)]  # a class unseen in train cannot be evaluated fairly
    (out / "class_to_idx.json").write_text(json.dumps(c2i, indent=1))
    sz, bs, nw = cfg["image_size"], cfg["batch_size"], cfg.get("num_workers", 2)

    def mk(split: str, tf: Any, shuf: bool) -> DataLoader[Any]:
        return DataLoader(
            ClothingDataset(df[df.split == split], c2i, cfg["image_root"], tf),
            batch_size=bs,
            shuffle=shuf,
            num_workers=nw,
        )

    tr, va = mk("train", train_tf(sz), True), mk("validation", eval_tf(sz), False)

    model = build_model(len(classes), cfg.get("unfreeze", [])).to(device)
    if cfg.get("init_checkpoint"):  # fine-tuning starts from the unchanged baseline
        model.load_state_dict(torch.load(cfg["init_checkpoint"], map_location=device)["state_dict"])
        for name in cfg.get("unfreeze", []):
            for p in getattr(model, name).parameters():
                p.requires_grad = True
    groups = [
        {
            "params": [
                p for n, p in model.named_parameters() if p.requires_grad and n.startswith("fc")
            ],
            "lr": cfg["lr_head"],
        }
    ]
    if cfg.get("unfreeze"):
        groups.append(
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith("fc")
                ],
                "lr": cfg["lr_backbone"],
            }
        )
    opt = torch.optim.AdamW(groups, weight_decay=cfg.get("weight_decay", 1e-4))
    lossf = nn.CrossEntropyLoss()

    t0, best, bad, hist = time.time(), -1.0, 0, []
    for ep in range(cfg["epochs"]):
        model.train()
        tot = 0.0
        for x, y, _ in tr:
            opt.zero_grad()
            loss = lossf(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        ids, y, p, c = predict(model, va, device)
        m = metrics(y, p, classes)
        hist.append(
            {
                "epoch": ep + 1,
                "train_loss": tot / len(cast(ClothingDataset, tr.dataset)),
                "val_acc": m["accuracy"],
                "val_macro_f1": m["macro_f1"],
            }
        )
        print(hist[-1])
        if m["macro_f1"] > best:  # checkpoint chosen on VALIDATION macro-F1 only
            best, bad = m["macro_f1"], 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "config": cfg,
                    "epoch": ep + 1,
                    "arch": "resnet50",
                },
                out / "checkpoint.pt",
            )
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break  # stopping rule (documented in the config)
    cost = {
        "seconds": round(time.time() - t0, 1),
        "device": device,
        "epochs_run": len(hist),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }

    ck = torch.load(out / "checkpoint.pt", map_location=device)
    model.load_state_dict(ck["state_dict"])
    ids, y, p, c = predict(model, va, device)
    res = metrics(y, p, classes)
    res.update(
        {
            "best_epoch": ck["epoch"],
            "cost": cost,
            "history": hist,
            "seed": cfg["seed"],
            "split_source": sorted(df.split_source.unique().tolist()),
        }
    )
    save_predictions(
        out / "val_predictions.csv", ids, y, p, c, classes, df[["item_id", "source_dataset"]]
    )
    pd.DataFrame(
        confusion_matrix(y, p, labels=range(len(classes))), index=classes, columns=classes
    ).to_csv(out / "val_confusion_matrix.csv")
    (out / "val_metrics.json").write_text(json.dumps(res, indent=1))
    if final_test:  # only after the checkpoint is frozen; never used for any choice above
        te = mk("test", eval_tf(sz), False)
        ids, y, p, c = predict(model, te, device)
        (out / "test_metrics.json").write_text(json.dumps(metrics(y, p, classes), indent=1))
        save_predictions(out / "test_predictions.csv", ids, y, p, c, classes)
    reload_check(out, cfg, df, classes, device)
    return res


def load_for_inference(ckpt_path: str, device: str = "cpu") -> Any:
    ck = torch.load(ckpt_path, map_location=device)
    m = build_model(len(ck["classes"]), [], pretrained=False)
    m.load_state_dict(ck["state_dict"])
    m.to(device).eval()
    return m, ck["classes"], ck["config"]["image_size"]


@torch.no_grad()
def predict_item(
    model: Any, classes: Any, image: Image.Image, size: int, item_id: str, device: str = "cpu"
) -> dict[str, Any]:
    """
    Inference interface: returns the `category` block fields of the item-output schema
    (confidence in [0,1]).
    """
    pr = torch.softmax(model(eval_tf(size)(image.convert("RGB")).unsqueeze(0).to(device)), 1)[
        0
    ].cpu()
    c, i = pr.max(0)
    return {
        "item_id": item_id,
        "category": {"common_category": classes[int(i)], "confidence": float(c)},
    }


def reload_check(out: Any, cfg: Any, df: Any, classes: Any, device: Any) -> None:
    """Acceptance check: the saved checkpoint reloads and reproduces the saved predictions."""
    model, cls, size = load_for_inference(out / "checkpoint.pt", device)
    saved = pd.read_csv(out / "val_predictions.csv", dtype={"item_id": str, "pred": str}).head(20)
    sub = df.set_index("item_id").loc[saved.item_id]
    same = 0
    for iid, r in sub.iterrows():
        pred = predict_item(
            model, cls, open_image(r.image_ref, cfg["image_root"]), size, iid, device
        )["category"]["common_category"]
        same += int(pred == saved.set_index("item_id").loc[iid, "pred"])
    (out / "reload_check.json").write_text(json.dumps({"checked": len(saved), "matching": same}))
    print(f"reload check: {same}/{len(saved)} sample predictions reproduced")
    if same != len(saved):
        raise ValueError("Reloaded checkpoint predictions differ from saved validation predictions")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--final-test", action="store_true")
    a = ap.parse_args()
    run(yaml.safe_load(open(a.config)), a.final_test)
